from Model.ipa_openfold import *
from Model.other_layers import PerResidueLDDTCaPredictor, ExperimentallyResolvedHead


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
        del lddt, experimentally_resolved_logits
        return translation, outputs, pred_frames

