import torch
from typing import Dict, Optional, Tuple
from protein_utils.affine_utils import T
from Loss.distance_loss import compute_fape, compute_fape_dis

def backbone_loss(
    backbone_affine_tensor: torch.Tensor,
    backbone_affine_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    dis_gt = None,
    mask_window=None,
    fix_window=None,
    domain_window=None,
    dis_clamp=None,
    args=None,
    **kwargs,
) -> torch.Tensor:
    pred_aff = T.from_tensor(traj)
    gt_aff = T.from_tensor(backbone_affine_tensor)
    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_affine_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_affine_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
        dis_gt=dis_gt,
        mask_window=mask_window,
        fix_window=fix_window,
        domain_window=domain_window
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_affine_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_affine_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
            dis_gt=dis_gt,
            mask_window=mask_window,
            fix_window=fix_window,
            domain_window=domain_window
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)
    # return fape_loss
    if dis_gt != None:
        dis_loss = compute_fape_dis(
            pred_frames=pred_aff,
            pred_positions=pred_aff.get_trans(),
            length_scale=loss_unit_distance,
            target_frames=gt_aff[None],
            target_positions=gt_aff[None].get_trans(),
            eps=eps,
            dis_gt=dis_gt,
            mask_window=mask_window,
            fix_window=fix_window,
            domain_window=domain_window,
            dis_clamp=dis_clamp,
            args=args)
    else:
        dis_loss = 0.0


    return fape_loss, dis_loss