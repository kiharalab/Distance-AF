from os import rename
import numpy as np

from Model.ipa_openfold import *
from typing import Dict, Optional, Tuple
from Loss.backbone_loss import backbone_loss
from Loss.sidechain_loss import sidechain_loss_dis
from Loss.distance_loss import compute_fape_dis, compute_fape, compute_sidechain_dis

from train_utils.feats import atom14_to_atom37, pseudo_beta_fn

import pickle
import os
import itertools


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins=50, c_in=384, c_hidden=128):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s

class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s=384, c_out=37, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
class Dist_AF_IPA(nn.Module):
    def __init__(self, args):
        super(Dist_AF_IPA, self).__init__()
        self.structure_module = StructureModule(trans_scale_factor=args.point_scale, no_blocks=args.ipa_depth, no_heads_ipa=12, c_ipa=16) #no_heads_ipa=24, c_ipa=64
        self.plddt =  PerResidueLDDTCaPredictor()
        self.experimentally_resolved = ExperimentallyResolvedHead()
        self.args = args

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        #self.pair_project = nn.Linear(128 + 128, 128)
    def forward(self, embedding, single_repr, aatype, batch_gt_frames):

        output_bb, translation, outputs = self.structure_module(single_repr, embedding, f=aatype, mask=batch_gt_frames['seq_mask'])
        pred_frames = torch.stack(output_bb)
        lddt = self.plddt(outputs['single'])
        experimentally_resolved_logits = self.experimentally_resolved(outputs['single'])
        return translation, outputs, pred_frames

        bb_loss, dis_loss = backbone_loss(
            backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
            traj=pred_frames,
            dis_gt=dist_constraint, 
            mask_window=domain_window,
            domain_window=domain_window,
            dis_loss_weight=1.0,
            dis_clamp=self.args.dis_clamp,
            args=self.args,
            local_constraint=self.args.local_constraint
        )

        rename =compute_renamed_ground_truth(batch_gt, outputs['positions'][-1])
       
        sc_loss = sidechain_loss_dis(
            sidechain_frames=outputs['sidechain_frames'],
            sidechain_atom_pos=outputs['positions'],
            rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
            rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
            rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
            renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
            renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
            alt_naming_is_better=rename['alt_naming_is_better'],
            dis_gt=dist_constraint, 
            dist_window=domain_window
        )
        
        angle_loss = supervised_chi_loss(outputs['angles'],
                                        outputs['unnormalized_angles'],
                                        aatype=aatype,
                                        seq_mask=batch_gt_frames['seq_mask'],
                                        chi_mask=batch_gt_frames['chi_mask'],
                                        chi_angles_sin_cos=batch_gt_frames['chi_angles_sin_cos'],
                                        chi_weight=0.5,
                                        angle_norm_weight=0.01,
                                        dist=self.args.dist
                                        )
        
        #print(angle_loss)
        #print(bb_loss)
        #print('angle: ', angle_loss.size())
        fape = 0.5 * bb_loss + 0.5 * sc_loss
        
        #print('fape: ', fape.size())
        vio_loss = 0
        plddt_loss = 0
        if not self.training or self.args.dist:
            batch_gt.update({'aatype': aatype})
            violation = find_structural_violations(batch_gt, outputs['positions'][-1],
                                                violation_tolerance_factor=12,
                                                clash_overlap_tolerance=1.5)
            violation_loss_ = violation_loss(violation, batch_gt['atom14_atom_exists'])
            vio_loss = torch.mean(violation_loss_)
            #print(violation_loss_)
            fape += 1 * violation_loss_

            #print(plddt_loss)
            experimentally_resolved_logits = self.experimentally_resolved(outputs['single'])
            exp_loss = experimentally_resolved_loss(experimentally_resolved_logits, 
                                                    atom37_atom_exists=batch_gt['atom37_atom_exists'],
                                                    all_atom_mask=batch_gt['all_atom_mask'],
                                                    resolution=resolution)
            fape += 0.01 * exp_loss
            
            lddt = self.plddt(outputs['single'])
            final_position = atom14_to_atom37(outputs['positions'][-1], batch_gt) 
            plddt_loss = lddt_loss(lddt, final_position, 
                                    all_atom_positions=batch_gt['all_atom_positions'], 
                                    all_atom_mask=batch_gt['all_atom_mask'],
                                    resolution=resolution)
        #print('plddt: ', plddt_loss.size())

        fape += 0.01 * plddt_loss
        #print(exp_loss)
        fape = torch.mean(fape)
        #print('fape before:', fape.item())
        fape += 0.5 * angle_loss
        #print('fape after:', fape.item())
       
        seq_len = torch.mean(batch_gt["seq_length"].float())
        crop_len = torch.tensor(aatype.shape[-1]).to(device=aatype.device)
        #print('seq: ', seq_len)
        #print('crop: ', crop_len)
        fape = fape * torch.sqrt(min(seq_len, crop_len))
        #print(fape)

        angle_loss_weight = 0
        if self.args.angle_loss:
            angle_loss_weight = 1
        
        # if self.args.dist and self.args.vector_dist:
        #     fape = (6 * dis_loss + bb_loss+ sc_loss + vio_loss + angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
        if self.args.dist:
            # if self.args.disable_bb_loss and dis_loss < 1.0:
            #     fape = (dis_loss + sc_loss + 1 * vio_loss + 0.01 * plddt_loss) * torch.sqrt(min(seq_len, crop_len))
            
            if self.args.reweight:
                # if dis_loss < 1.0:
                #     fape = 12 * self.args.dist_weight * dis_loss +( bb_loss+ sc_loss + vio_loss + angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
                
                if dis_loss > 10.0:
                    fape = 12 * self.args.dist_weight* dis_loss + (bb_loss+ sc_loss + vio_loss + angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
                elif dis_loss > 5.0 and dis_loss < 10.0:
                    fape = 24 * self.args.dist_weight* dis_loss + ( bb_loss+ sc_loss + vio_loss+ angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
                else:
                    fape = 48 * self.args.dist_weight* dis_loss + (bb_loss + sc_loss + vio_loss+ angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
            else:
                if dis_loss < 1.0:
                    fape = 6 * self.args.dist_weight *dis_loss +( 10 * bb_loss+ 10 * sc_loss + 10 * vio_loss + angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
                elif dis_loss > 10.0:
                    fape = 6 * self.args.dist_weight* dis_loss + (0.5 * bb_loss+ 0.5 * sc_loss + vio_loss + angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
                elif dis_loss > 5.0 and dis_loss < 10.0:
                    fape = 12 * self.args.dist_weight* dis_loss + (0.5 * bb_loss+ 0.5 * sc_loss + vio_loss+ angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))
                else:
                    fape = 24 * self.args.dist_weight* dis_loss + (0.5 * bb_loss + 0.5 * sc_loss + vio_loss+ angle_loss_weight * angle_loss ) * torch.sqrt(min(seq_len, crop_len))

        print("backbone:", bb_loss, "sidechain:", sc_loss, "angle_loss:",angle_loss, "dis_loss:", dis_loss)
        return translation*self.args.point_scale, fape, outputs['positions'], vio_loss, angle_loss, plddt_loss,  \
                torch.mean(bb_loss), torch.mean(sc_loss)

