import torch
from protein_utils.affine_utils import T
from typing import Dict, Optional, Tuple

def compute_fape_dis(
    pred_frames: T,
    pred_positions: torch.Tensor,
    length_scale: float,
    eps=1e-8,
    dis_gt=None
) -> torch.Tensor:

    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    if dis_gt is not None:      
        dis_mask = torch.where(dis_gt!=0,1,0)
        local_pred_pos_pair = local_pred_pos * dis_mask[None, None, ..., None]

        error_dist_pair = 0.5 * torch.square((dis_gt[None,None,...] - torch.sqrt(
        torch.sum(local_pred_pos_pair ** 2 , dim=-1) + eps)) * dis_mask[None,None, ...])

        error_dist_pair = error_dist_pair / length_scale
        
        loss  = torch.sum(error_dist_pair) / (len(torch.nonzero(dis_mask)) * pred_positions.size(0))
        return loss 


def compute_sidechain_dis(
    pred_frames: T,
    target_frames: T,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    dis_gt=None,
    dist_window=None
) -> torch.Tensor:

    # [*, N_frames, N_pts, 3]

    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    if dis_gt is not None:
        if dist_window is not None:
            dist_window_sidechain = dist_window.repeat_interleave(14,dim=1).repeat_interleave(8,dim=0)
            local_pred_pos = local_pred_pos * dist_window_sidechain[None,...,None]
            local_target_pos = local_target_pos * dist_window_sidechain[None,...,None]

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )
    
    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
    
    normed_error = error_dist / length_scale
    #print(normed_error)
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]
    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter
    # atom14_mask = positions_mask.squeeze(0).repeat_interleave(frames_mask.shape[1]).reshape(-1, positions_mask.shape[1])
    # group8_mask = frames_mask.squeeze(0).repeat_interleave(positions_mask.shape[1]).reshape(-1, frames_mask.shape[1]).transpose_(0,1)
    # print(group8_mask.shape)
    # print(atom14_mask.shape)
    # print(normed_error.shape)

    normed_error = torch.sum(normed_error, dim=-1)

    
    # if dist_window is not None:
    #     normed_error = (
    #         normed_error / (eps + torch.sum(dist_window_sidechain * group8_mask, dim=-1))[None,...]
    #     )
    #     normed_error = torch.sum(normed_error, dim=-1)
    #     normed_error = normed_error / (eps + torch.sum(torch.sum(dist_window_sidechain * atom14_mask, dim=0),dim=0)/atom14_mask.shape[1])
    # else:
    normed_error = (
    normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    
    return normed_error

def compute_fape(
    pred_frames: T,
    target_frames: T,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    dis_gt=None,
    mask_window=None
) -> torch.Tensor:
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )
    error_dist = torch.sqrt(
            torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
        )

    if dis_gt is not None:
        if mask_window is not None :
            error_dist = error_dist * mask_window[None, None, ...]
    
    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
    
    normed_error = error_dist / length_scale
    # print(normed_error.shape)
    # exit(0)
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    return normed_error