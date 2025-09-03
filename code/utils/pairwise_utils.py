import torch
import numpy as np
from utils import geo_utils, dataset_utils
import cv2
from torch.nn import functional as F




def pairwise_epipole_loss(data, Rs_pred, ts_pred, device=None):
        """
        Compute loss between predicted and ground-truth pairwise epipoles.

        Args:
            data: SceneData object with .pairwise_epipoles [N, N, 4] (ground truth)
            Rs_pred: [N, 3, 3] predicted rotation matrices
            ts_pred: [N, 3] predicted translation vectors

        Returns:
            epipole_loss: scalar tensor (mean L2 loss over all valid pairs)
        """
        # Compute predicted pairwise epipoles [N, N, 4]
        pred_epipoles = geo_utils.compute_pairwise_epipoles_from_Rt(Rs_pred, ts_pred)

        # Get ground-truth pairwise epipoles [N, N, 4]
        # Extract Rs_gt and ts_gt from SceneData object
        # Ks_invT = getattr(data, 'Ns_invT', None)
        # Ks = Ks_invT.transpose(1, 2) if Ks_invT is not None else None
        # Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.to(device), Ks.to(device) if Ks is not None else None)
        # gt_epipoles = geo_utils.compute_pairwise_epipoles_from_Rt(Rs_gt, ts_gt)
        gt_epipoles = data.pairwise_epipoles.to(device, dtype=pred_epipoles.dtype)

        # Only consider off-diagonal pairs (i != j)
        N = gt_epipoles.shape[0]
        mask = ~torch.eye(N, dtype=torch.bool, device=gt_epipoles.device)
        mask = mask.unsqueeze(-1).expand(-1, -1, 4)  # [N, N, 4]

        # Compute L2 loss per pair (over 4 epipole coords)
        diff = (pred_epipoles - gt_epipoles)[mask].view(-1, 4)
        loss = torch.norm(diff, dim=-1).mean() if diff.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)

        return loss


def compute_absolute_pose_consistency(Rs_pred, ts_pred, Rs_gt, ts_gt, calibrated=True, device=None):
        """
        Compute absolute pose consistency loss using the same method as evaluation.py.
        This aligns predicted poses with ground truth and computes rotation/translation errors.
        """
        if not calibrated:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Extract predicted poses using the same method as evaluation.py
        # Ps_norm = pred_cam["Ps_norm"]  # [m, 3, 4]
        
        # # Convert to numpy for geo_utils functions (they expect numpy)
        # Ps_norm_np = Ps_norm.detach().cpu().numpy()
        
        # # Decompose camera matrices to get rotations and translations
        # Rs_pred_np, ts_pred_np = geo_utils.decompose_camera_matrix(Ps_norm_np)
        
        # # Get ground truth poses
        # Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()
        # Rs_gt_np, ts_gt_np = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ns_inv)

        # Align predicted poses with ground truth using the same alignment as evaluation.py
        Rs_pred_np = Rs_pred.detach().cpu().numpy()
        Rs_gt_np = Rs_gt.detach().cpu().numpy()
        ts_pred_np = ts_pred.detach().cpu().numpy()
        ts_gt_np = ts_gt.detach().cpu().numpy()
        Rs_fixed_np, ts_fixed_np, _ = geo_utils.align_cameras(
            Rs_pred_np, Rs_gt_np, ts_pred_np, ts_gt_np, return_alignment=True
        )
        
        # Compute rotation and translation errors
        Rs_error_np, ts_error_np = geo_utils.tranlsation_rotation_errors(
            Rs_fixed_np, ts_fixed_np, Rs_gt_np, ts_gt_np
        )
        
        # Convert back to tensors
        Rs_error = torch.from_numpy(Rs_error_np).float().to(device)
        ts_error = torch.from_numpy(ts_error_np).float().to(device)
        
        # Compute mean errors (same as evaluation.py)
        rotation_loss = Rs_error.mean()
        translation_loss = ts_error.mean()
        
        return rotation_loss, translation_loss

def compute_pairwise_pose_consistency(Rs_pred, ts_pred, Rs_gt, ts_gt, device=None, rotation_weight=1.0, translation_weight=1.0):
    """
    Compute pairwise pose consistency between predicted and ground-truth relative poses.

    Returns:
        rotation_loss, translation_loss (scalars)
    """
    
    relative_poses_pred = geo_utils.batch_get_relative_pose(Rs_pred, ts_pred)

    # Ground-truth absolute poses -> relative poses
    # Ks_invT = getattr(data, 'Ns_invT', None)
    # Ks = Ks_invT.transpose(1, 2) if Ks_invT is not None else None
    # Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.to(device), Ks.to(device) if Ks is not None else None)
    relative_poses_gt = geo_utils.batch_get_relative_pose(Rs_gt, ts_gt)

    # Ensure same device/dtype
    relative_poses_pred = relative_poses_pred.to(device)
    relative_poses_gt = relative_poses_gt.to(device)

    rot_loss, trans_loss = relative_pose_matrix_loss(
        relative_poses_pred, relative_poses_gt,
    )

    return rot_loss, trans_loss


def relative_pose_matrix_loss(relative_poses_pred, relative_poses_gt, eps=1e-7):
    """
    Vectorized loss between two relative-pose matrices.
    Args:
        relative_poses_pred: Tensor [n, n, 7] with [tx, ty, tz, w, x, y, z]
        relative_poses_gt:   Tensor [n, n, 7] with same layout
    Returns:
        rot_loss (scalar), trans_loss (scalar)
    """

    # Split
    t_pred, q_pred = relative_poses_pred[..., :3], relative_poses_pred[..., 3:]
    t_gt,   q_gt   = relative_poses_gt[...,   :3], relative_poses_gt[...,   3:]

    # Normalize quaternions
    q_pred = F.normalize(q_pred, dim=-1)
    q_gt   = F.normalize(q_gt,   dim=-1)

    # Rotation geodesic distance: 2*acos(|<q1, q2>|)
    cos_q = torch.sum(q_pred * q_gt, dim=-1).abs()
    cos_q = cos_q.clamp(min=-1.0 + eps, max=1.0 - eps)
    rot_loss = 2.0 * torch.acos(cos_q)

    # Translation direction alignment: 1 - |cos(theta)|
    t_pred_n = F.normalize(t_pred, dim=-1)
    t_gt_n   = F.normalize(t_gt,   dim=-1)
    trans_loss = 1.0 - torch.sum(t_pred_n * t_gt_n, dim=-1).abs()

    # Mask out diagonal (i == j) in one go
    n = relative_poses_pred.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=relative_poses_pred.device)
    mask = mask.view(n, n, 1).expand(-1, -1, 2)  # expand to match [n,n] losses

    # Stack losses, mask, and reduce
    losses = torch.stack([rot_loss, trans_loss], dim=-1)  # [n,n,2]
    losses = losses[mask].view(-1, 2).mean(0)

    rot_loss, trans_loss = losses[0], losses[1]
    return rot_loss, trans_loss
    
