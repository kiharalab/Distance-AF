from torch.utils.data import DataLoader
from Dataset.dataset import DistAF_Dataset
import torch.optim as optim
import torch
from Model.Dist_AF import Dist_AF_IPA
import os
from Loss.backbone_loss import backbone_loss
from Loss.sidechain_loss import sidechain_loss_dis
from Loss.openfold_loss import compute_renamed_ground_truth, supervised_chi_loss,find_structural_violations,violation_loss
from train_utils.feats import atom14_to_atom37
from protein_utils import protein
from utils import rmsd
import numpy as np
import functools
from utils.set_seed import set_seed
from train_utils.collate import collate_fn
def train(args):
    set_seed(args)
    with open(args.train_targets, 'r') as f:
        target_name = f.read().splitlines()[0]
    target_output_dir = os.path.join(args.output_dir,target_name)
    if not os.path.exists(target_output_dir):
        os.makedirs(target_output_dir)
    train_dataset = DistAF_Dataset(args)
    args.training_examples = len(train_dataset)
    collate = functools.partial(collate_fn, args=args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    model = Dist_AF_IPA(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.to(args.device_id)
    if args.model_dir:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer.load_state_dict(
        #     torch.load(f'{args.model_dir}/optimizer.pt', map_location=f'{device}:{args.device_id}')
        # )
        if os.path.exists(f'{args.model_dir}/checkpoint.pth'):
            checkpoint = torch.load(f'{args.model_dir}/checkpoint.pth', map_location=args.device_id)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            optimizer.load_state_dict(optimizer_state_dict)
            starting_epoch = checkpoint['epoch']
            # rng_state = checkpoint['rng_state']
            # torch.set_rng_state(rng_state)
        else:
            starting_epoch = 0
            model.load_state_dict(torch.load(f'{args.model_dir}/model_state_dict.pt', map_location=args.device_id))
            optimizer_state_dict = torch.load(f'{args.model_dir}/optimizer.pt', map_location=args.device_id)
            for key in optimizer_state_dict.keys():
                optimizer_state_dict[key] = optimizer_state_dict[key].to(args.device_id) if isinstance(optimizer_state_dict[key], torch.Tensor) else optimizer_state_dict[key]
            optimizer.load_state_dict(optimizer_state_dict)
        print(f'Checkpoints (model and optimizer) loaded from {args.model_dir}')
    else:
        starting_epoch = 0
    print("----------------- Starting Training ---------------")
    print("  Num examples = %d" % (int(args.training_examples)))
    print("  Num Epochs = %d" % (int(args.epochs)))
    print("  Batch Size = %d" % (int(args.batch)))

    model.train()

    #casp_results = test(args, model, test_dataloader)
    #results = val(args, model, val_dataloader)
    gt_keys = ['all_atom_positions', 'all_atom_mask', 'atom14_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions', 
                'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', 
                'atom14_atom_is_ambiguous', 'residue_index']
    gt_frames_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames',
                        'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask', 'chi_angles_sin_cos', 'chi_mask', 'seq_mask']
    for epoch in range(starting_epoch + 1,args.epochs+1):
        for step, (batch,target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            embedding = batch['embed']
            domain_window = batch['domain_window'].squeeze(0)
            dist_constraint = batch['dist_constraint'].squeeze(0)
            single_repr_batch = batch['single_representation']
  
            aatype_batch = batch["aatype"]
            batch_gt = {key: batch[key] for key in gt_keys}
            batch_gt_frames = {key: batch[key] for key in gt_frames_keys}

            batch_gt.update({'seq_length': batch['seq_length']})
            resolution = batch['resolution']
            representation = None
            
            if args.cuda:
                embedding = embedding.to(args.device_id)
                resolution = resolution.to(args.device_id)
                for key in batch_gt.keys():
                    batch_gt[key] = batch_gt[key].to(args.device_id)
                for key in batch_gt_frames.keys():
                    batch_gt_frames[key] = batch_gt_frames[key].to(args.device_id)
                single_repr_batch = single_repr_batch.to(args.device_id)
                #coords_batch = coords_batch.cuda(args.device_id)
                #masks_batch = masks_batch.cuda(args.device_id)
                aatype_batch = aatype_batch.to(args.device_id)
                domain_window = domain_window.to(args.device_id)
                dist_constraint = dist_constraint.to(args.device_id)

            translation, outputs, pred_frames = model(embedding, single_repr_batch, aatype_batch, batch_gt_frames)

            #compute all needed loss
            bb_loss, dis_loss = backbone_loss(
                backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
                backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
                traj=pred_frames,
                dis_gt=dist_constraint, 
                mask_window=domain_window,
                domain_window=domain_window,
                dis_clamp=args.dis_clamp,
                args=args
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
                                        aatype=aatype_batch,
                                        seq_mask=batch_gt_frames['seq_mask'],
                                        chi_mask=batch_gt_frames['chi_mask'],
                                        chi_angles_sin_cos=batch_gt_frames['chi_angles_sin_cos'],
                                        chi_weight=0.5,
                                        angle_norm_weight=0.01,
                                        dist=args.dist
                                        )

            batch_gt.update({'aatype': aatype_batch})
            violation = find_structural_violations(batch_gt, outputs['positions'][-1],
                                                violation_tolerance_factor=12,
                                                clash_overlap_tolerance=1.5)
            violation_loss_ = violation_loss(violation, batch_gt['atom14_atom_exists'])
            vio_loss = torch.mean(violation_loss_)
            #print(violation_loss_)
            seq_len = torch.mean(batch_gt["seq_length"].float())
            crop_len = torch.tensor(aatype_batch.shape[-1]).to(device=aatype_batch.device)
            if dis_loss > 10.0:
                fape = 12 * dis_loss * args.dist_weight + (bb_loss+ sc_loss + vio_loss + angle_loss ) * torch.sqrt(min(seq_len, crop_len))
            elif dis_loss > 5.0 and dis_loss < 10.0:
                fape = 24 * dis_loss * args.dist_weight+ ( bb_loss+ sc_loss + vio_loss+ angle_loss ) * torch.sqrt(min(seq_len, crop_len))
            else:
                fape = 48 * dis_loss * args.dist_weight + (bb_loss + sc_loss + vio_loss+ angle_loss ) * torch.sqrt(min(seq_len, crop_len))
            loss = fape
            print(f"Epoch:{epoch}, FAPE loss:{loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            #save model checkpoint
            if args.val_epochs > 0 and epoch % args.val_epochs == 0 and epoch > 0:
                epoch_output_dir = os.path.join(target_output_dir, f"checkpoint-{epoch}-{epoch}")
                if not os.path.exists(epoch_output_dir):
                    os.makedirs(epoch_output_dir)
                checkpoint = {'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'rng_state': torch.get_rng_state()}
                torch.save(checkpoint, os.path.join(epoch_output_dir, "checkpoint.pth"))
                #save predicted pdb for each evaluted epoch
                final_pos = atom14_to_atom37(outputs['positions'][-1], batch_gt)
                final_atom_mask = batch_gt["atom37_atom_exists"]
                initial_pdb = os.path.join(args.msa_transformer_dir, target_name, f'{target_name}_pred_full.pdb')
                with open(initial_pdb,"r") as f:
                    initial_pdb_str = f.read()
                prot_initial = protein.from_pdb_string(initial_pdb_str)
                pred_prot = protein.Protein(
                    aatype=aatype_batch.squeeze(0).cpu().numpy(),
                    atom_positions=final_pos.squeeze(0).detach().cpu().numpy(),
                    atom_mask=final_atom_mask.squeeze(0).cpu().numpy(),
                    residue_index=prot_initial.residue_index,
                    b_factors=prot_initial.b_factors,
                )
                pred_pdb_lines = protein.to_pdb(pred_prot)

                output_dir_pred = os.path.join(epoch_output_dir, f"{target_name}_pred_all.pdb")
                with open(output_dir_pred, 'w') as f:
                    f.write(pred_pdb_lines)
                
                #compute RMSD between initial and epoch predicted pdb
                gt = prot_initial.atom_positions[None, ...][prot_initial.atom_mask[None, ...].astype(bool)]
                pred = pred_prot.atom_positions[None, ...][prot_initial.atom_mask[None, ...].astype(bool)]
                gt = gt.reshape(-1, 3)
                pred = pred.reshape(-1, 3)
                gt -= rmsd.centroid(gt)
                pred -= rmsd.centroid(pred)
                U = rmsd.kabsch(gt, pred)
                A = np.dot(gt, U)
                rmsd_value = rmsd.rmsd(A, pred)
                with open(os.path.join(epoch_output_dir, f"rmsd.txt"), 'w') as file:
                    file.write(f'{rmsd_value}\n')
    return    