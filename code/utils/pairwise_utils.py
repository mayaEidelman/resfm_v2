import torch
import numpy as np
from utils import geo_utils, dataset_utils
import cv2
from torch.nn import functional as F


def extract_pairwise_matches_from_scene(data, min_matches=8):
    """
    Extract pairwise matches from scene data for computing relative poses.
    
    Args:
        data: SceneData object containing M matrix and camera information
        min_matches: Minimum number of matches required for a pair
        
    Returns:
        matches_dict: Dictionary with keys (i, j) containing match data
    """
    print("Extracting pairwise matches from scene data for computing relative poses")
    matches_dict = {}
    n_cameras = data.y.shape[0]
    
    # Extract 2D points for each camera
    M = data.M  # [2*m, n] where m is number of cameras
    valid_points = data.valid_pts

    for i in range(n_cameras):
        for j in range(i + 1, n_cameras):
            # Get points visible in both cameras
            pts_i = M[2*i:2*i+2, :]  # [2, n]
            pts_j = M[2*j:2*j+2, :]  # [2, n]
            
            # Find points visible in both cameras
            visible_i = pts_i[0, :] > 0
            visible_j = pts_j[0, :] > 0
            visible_both = visible_i & visible_j 
            
            if visible_both.sum() < min_matches:
                continue
            
            # Extract matched points
            pts1 = pts_i[:, visible_both]  # [2, N]
            pts2 = pts_j[:, visible_both]  # [2, N]
            
            matches_dict[(i, j)] = {
                'pts1': pts1,
                'pts2': pts2,
                'num_matches': visible_both.sum()
            }
    
    return matches_dict


def extract_pairwise_matches_from_scene(data, min_matches=8):
    """
    Extract pairwise matches from scene data for computing relative poses.
    
    Args:
        data: SceneData object containing M matrix and camera information
        min_matches: Minimum number of matches required for a pair
        
    Returns:
        matches_dict: Dictionary with keys (i, j) containing match data
    """
    print("Extracting pairwise matches from scene data for computing relative poses - pairwise utils")
    matches_dict = {}
    n_cameras = data.y.shape[0]
    
    # Extract 2D points for each camera
    M = data.M  # [2*m, n] where m is number of cameras
    valid_points = data.valid_pts  # optional if used

    # Precompute visibility mask for each camera
    # visible[i, :] = bool mask of points visible in camera i
    visible = (M[0::2, :] > 0)  # take only x-coordinates, shape [m, n]

    for i in range(n_cameras):
        pts_i = M[2*i:2*i+2, :]  # [2, n]

        # Compare i with all j > i at once
        common_visible = visible[i] & visible[i+1:]  # shape [m-i-1, n]

        # Count number of matches per pair
        num_matches = common_visible.sum(axis=1)

        # Filter pairs with enough matches
        valid_js = np.where(num_matches >= min_matches)[0] + (i+1)

        for idx, j in enumerate(valid_js):
            mask = common_visible[valid_js[idx] - (i+1)]
            pts1 = pts_i[:, mask]
            pts2 = M[2*j:2*j+2, :][:, mask]

            matches_dict[(i, j)] = {
                'pts1': pts1,
                'pts2': pts2,
                'num_matches': pts1.shape[1]
            }

    return matches_dict

def fast_pairwise_matches_from_sparse(data, min_matches=8):
    # Sparse visibility (cameras x points)
    cam_idx = data.x.indices[0]          # [Nobs]
    pt_idx  = data.x.indices[1]          # [Nobs]
    m, n, _ = data.x.mat_shape
    vals = torch.ones_like(cam_idx, dtype=torch.float32)
    A = torch.sparse_coo_tensor(
        torch.stack([cam_idx, pt_idx], dim=0),
        vals, size=(m, n)
    ).coalesce()

    # Shared counts for all pairs
    counts = torch.sparse.mm(A, A.transpose(0, 1)).to_dense()  # [m, m]
    iu = torch.triu(torch.ones_like(counts, dtype=torch.bool), diagonal=1)
    valid_pairs = torch.nonzero((counts >= min_matches) & iu, as_tuple=False)

    # Preindex observations for quick lookup
    # Build per-camera arrays of (point_ids, index_in_sparse_values)
    obs_index = torch.arange(cam_idx.numel(), device=cam_idx.device)
    per_cam_pts = [pt_idx[cam_idx == i].cpu().numpy() for i in range(m)]
    per_cam_obs = [obs_index[cam_idx == i].cpu().numpy() for i in range(m)]

    matches = {}
    for i, j in valid_pairs.cpu().numpy():
        pts_i, obs_i = per_cam_pts[i], per_cam_obs[i]
        pts_j, obs_j = per_cam_pts[j], per_cam_obs[j]

        common_pts, idx_i, idx_j = np.intersect1d(pts_i, pts_j, assume_unique=False, return_indices=True)
        if common_pts.size < min_matches:
            continue

        # Use indices to pick rows in data.x.values (normalized coords)
        v_i = data.x.values[obs_i[idx_i]]  # [k, 2]
        v_j = data.x.values[obs_j[idx_j]]  # [k, 2]

        matches[(i, j)] = {
            'pts1': v_i.T.contiguous(),     # [2, k]
            'pts2': v_j.T.contiguous(),     # [2, k]
            'num_matches': v_i.shape[0],
            'point_ids': torch.from_numpy(common_pts)
        }

    return matches

def compute_relative_poses_for_scene(data, matches_dict=None, calibrated=True):
    """
    Compute relative poses for all camera pairs in a scene.
    
    Args:
        data: SceneData object
        matches_dict: Pre-computed matches dictionary (optional)
        calibrated: Whether cameras are calibrated
        
    Returns:
        relative_poses: Dictionary with relative pose information
    """
    
    if matches_dict is None:
        print("matches_dict is Non", )
        matches_dict = extract_pairwise_matches_from_scene(data)
    
    n_cameras = data.y.shape[0]
    relative_poses = {}
    
    # Get intrinsic matrices if calibrated
    Ks = None
    if calibrated:
        Ks = data.Ns.inverse().cpu().numpy()  # Ns_inv are the intrinsic matrices
    
    for (i, j), matches in matches_dict.items():
        pts1 = matches['pts1']
        pts2 = matches['pts2']
        
        K1 = Ks[i] if Ks is not None else None
        K2 = Ks[j] if Ks is not None else None
        
        # Compute relative pose
        R, t, inliers = geo_utils.compute_relative_pose_from_matches(
            pts1, pts2, K1, K2, method='5pt' if calibrated else '8pt'
        )
        
        if R is not None and t is not None:
            # Validate the relative pose
            valid, error = geo_utils.validate_relative_pose(
                R, t, pts1, pts2, K1, K2, threshold=4.0
            )
            
            if valid:
                relative_poses[(i, j)] = {
                    'R': torch.from_numpy(R).float(),
                    't': torch.from_numpy(t).float(),
                    'inliers': torch.from_numpy(inliers).bool() if inliers is not None else None,
                    'error': error,
                    'num_matches': matches['num_matches']
                }
    
    return relative_poses


def add_pairwise_data_to_scene(data, calibrated=True):
    """
    Add pairwise matches and relative poses to scene data.
    
    Args:
        data: SceneData object
        calibrated: Whether cameras are calibrated
        
    Returns:
        data: Updated SceneData object with pairwise information
    """
    # Compute pairwise matches
    matches_dict = extract_pairwise_matches_from_scene(data)
    
    # Compute relative poses
    relative_poses = compute_relative_poses_for_scene(data, matches_dict, calibrated)
    
    # Add to data object
    data.matches = matches_dict
    # data.relative_poses = relative_poses
    
    return data


def compute_epipolar_constraints(data, pred_cam):
    """
    Compute epipolar constraints for all camera pairs.
    
    Args:
        data: SceneData object with matches
        pred_cam: Predicted camera parameters
        
    Returns:
        epipolar_errors: Dictionary with epipolar errors for each pair
    """
    if not hasattr(data, 'matches'):
        return {}
    
    Ps_pred = pred_cam["Ps_norm"]  # [m, 3, 4] predicted camera matrices
    n_cameras = Ps_pred.shape[0]
    
    # Extract predicted rotations and translations
    Vs_pred = Ps_pred[:, 0:3, 0:3].inverse().transpose(1, 2)  # [m, 3, 3]
    ts_pred = torch.bmm(-Vs_pred.transpose(1, 2), Ps_pred[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()  # [m, 3]
    
    epipolar_errors = {}
    
    for (i, j), matches in data.matches.items():
        if i >= n_cameras or j >= n_cameras:
            continue
            
        pts1 = matches['pts1']
        pts2 = matches['pts2']
        
        # Compute fundamental matrix from predicted poses
        F_pred = geo_utils.get_fundamental_from_V_t(
            Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j]
        )
        
        # Compute epipolar error
        error = geo_utils.compute_epipolar_constraint_error(F_pred, pts1, pts2)
        epipolar_errors[(i, j)] = error
    
    return epipolar_errors


def compute_pose_consistency_errors(data, pred_cam):
    """
    Compute pose consistency errors between predicted and ground truth relative poses.
    
    Args:
        data: SceneData object with relative poses
        pred_cam: Predicted camera parameters
        
    Returns:
        consistency_errors: Dictionary with consistency errors for each pair
    """
    if not hasattr(data, 'relative_poses'):
        return {}
    
    Ps_pred = pred_cam["Ps_norm"]  # [m, 3, 4] predicted camera matrices
    n_cameras = Ps_pred.shape[0]
    
    # Extract predicted rotations and translations
    Vs_pred = Ps_pred[:, 0:3, 0:3].inverse().transpose(1, 2)  # [m, 3, 3]
    ts_pred = torch.bmm(-Vs_pred.transpose(1, 2), Ps_pred[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()  # [m, 3]
    
    consistency_errors = {}
    
    for (i, j), rel_pose in data.relative_poses.items():
        if i >= n_cameras or j >= n_cameras:
            continue
            
        # Get predicted relative pose
        R_rel_pred = torch.bmm(Vs_pred[j], Vs_pred[i].transpose(1, 2))  # R_j * R_i^T
        t_rel_pred = ts_pred[j] - torch.bmm(R_rel_pred, ts_pred[i].unsqueeze(-1)).squeeze()
        
        # Get ground truth relative pose
        R_rel_gt = rel_pose['R']
        t_rel_gt = rel_pose['t']
        
        # Compute rotation error
        rot_error = geo_utils.compare_rotations(
            R_rel_pred.unsqueeze(0), R_rel_gt.unsqueeze(0)
        )[0]
        
        # Compute translation error
        trans_error = torch.norm(t_rel_pred - t_rel_gt, dim=0)
        
        consistency_errors[(i, j)] = {
            'rotation_error': rot_error,
            'translation_error': trans_error.item(),
            'num_matches': rel_pose['num_matches']
        }
    
    return consistency_errors


def evaluate_pairwise_consistency(data, pred_cam):
    """
    Evaluate pairwise consistency for a scene.
    
    Args:
        data: SceneData object
        pred_cam: Predicted camera parameters
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Compute epipolar constraints
    epipolar_errors = compute_epipolar_constraints(data, pred_cam)
    
    # Compute pose consistency
    consistency_errors = compute_pose_consistency_errors(data, pred_cam)
    
    # Aggregate metrics
    metrics = {
        'mean_epipolar_error': np.mean(list(epipolar_errors.values())) if epipolar_errors else float('inf'),
        'mean_rotation_error': np.mean([e['rotation_error'] for e in consistency_errors.values()]) if consistency_errors else float('inf'),
        'mean_translation_error': np.mean([e['translation_error'] for e in consistency_errors.values()]) if consistency_errors else float('inf'),
        'num_pairs': len(consistency_errors),
        'epipolar_errors': epipolar_errors,
        'consistency_errors': consistency_errors
    }
    
    return metrics 

def compute_absolute_pose_consistency(pred_cam, data, calibrated=True, device=None):
        """
        Compute absolute pose consistency loss using the same method as evaluation.py.
        This aligns predicted poses with ground truth and computes rotation/translation errors.
        """
        if not calibrated:
            return torch.tensor(0.0, device=pred_cam["Ps_norm"].device, requires_grad=True)
        
        # Extract predicted poses using the same method as evaluation.py
        Ps_norm = pred_cam["Ps_norm"]  # [m, 3, 4]
        
        # Convert to numpy for geo_utils functions (they expect numpy)
        Ps_norm_np = Ps_norm.detach().cpu().numpy()
        
        # Decompose camera matrices to get rotations and translations
        Rs_pred_np, ts_pred_np = geo_utils.decompose_camera_matrix(Ps_norm_np)
        
        # Get ground truth poses
        Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()
        Rs_gt_np, ts_gt_np = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ns_inv)
        
        # Align predicted poses with ground truth using the same alignment as evaluation.py
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

def compute_epipolar_constraint_error(F, pts1, pts2):
    """
    Compute epipolar constraint error using fundamental matrix.
    
    Args:
        F: Fundamental matrix [3, 3]
        pts1: Points in first image [2, N] or [3, N]
        pts2: Points in second image [2, N] or [3, N]
        
    Returns:
        error: Mean epipolar error
    """
    """Compute symmetric epipolar distance."""
        # Ensure points are in homogeneous coordinates
    pts1 = pts1.to(F.device)
    pts2 = pts2.to(F.device)
    if pts1.shape[0] == 2:
        pts1 = torch.cat([pts1, torch.ones(1, pts1.shape[1], device=F.device)], dim=0)
    if pts2.shape[0] == 2:
        pts2 = torch.cat([pts2, torch.ones(1, pts2.shape[1], device=F.device)], dim=0)
    
    # Compute epipolar lines
    lines1 = F @ pts2  # F * x2
    lines2 = F.T @ pts1  # F^T * x1
    
    # Compute distances
    dist1 = torch.abs(torch.sum(pts1 * lines1, dim=0)) / torch.norm(lines1[:2, :], dim=0)
    dist2 = torch.abs(torch.sum(pts2 * lines2, dim=0)) / torch.norm(lines2[:2, :], dim=0)
    # Symmetric epipolar distance
    return torch.median(dist1 + dist2) # I changed here to median instead of meanto make it more robust to outliers but dont know it its ok :/

def relative_pose_matrix_loss(relative_poses_pred, relative_poses_gt, rotation_weight=0.5, translation_weight=0.5, eps=1e-7):
    """
    Compute loss between two relative-pose matrices produced by `geo_utils.batch_get_relative_pose`.

    Args:
        relative_poses_pred: Tensor [n, n, 7] with [tx, ty, tz, w, x, y, z]
        relative_poses_gt:   Tensor [n, n, 7] with same layout
        rotation_weight:     Scalar weight for rotation loss
        translation_weight:  Scalar weight for translation-direction loss
        eps:                 Small constant for numerical stability

    Returns:
        total_loss, rot_loss_mean, trans_loss_mean (all scalars)
    """
    assert relative_poses_pred.shape == relative_poses_gt.shape, "Pred/GT shapes must match"
    assert relative_poses_pred.shape[-1] == 7, "Relative pose tensors must have 7 channels (t[3] + q[4])"

    n = relative_poses_pred.shape[0]
    device = relative_poses_pred.device

    # Exclude i == j pairs (identity)
    offdiag_mask = ~torch.eye(n, dtype=torch.bool, device=device)

    t_pred = relative_poses_pred[..., :3]
    q_pred = relative_poses_pred[..., 3:]
    t_gt   = relative_poses_gt[..., :3]
    q_gt   = relative_poses_gt[..., 3:]

    # Normalize quaternions and compute geodesic distance: 2*acos(|<q1, q2>|)
    q_pred = F.normalize(q_pred, p=2, dim=-1)
    q_gt   = F.normalize(q_gt,   p=2, dim=-1)
    cos_q  = torch.sum(q_pred * q_gt, dim=-1).abs().clamp(min=-1.0 + eps, max=1.0 - eps)
    rot_loss = 2.0 * torch.acos(cos_q)

    # Translation direction alignment: 1 - |cos(theta)| between directions
    t_pred_n = F.normalize(t_pred, p=2, dim=-1)
    t_gt_n   = F.normalize(t_gt,   p=2, dim=-1)
    trans_loss = 1.0 - torch.sum(t_pred_n * t_gt_n, dim=-1).abs()

    # Apply off-diagonal mask and average
    rot_loss = rot_loss[offdiag_mask].mean()
    trans_loss = trans_loss[offdiag_mask].mean()

    return rot_loss, trans_loss

    
def compute_pairwise_pose_consistency(pred_cam, data, device=None, rotation_weight=1.0, translation_weight=1.0):
    """
    Compute pairwise pose consistency between predicted and ground-truth relative poses.

    Returns:
        rotation_loss, translation_loss (scalars)
    """
    Ps_norm = pred_cam["Ps_norm"]  # [m, 3, 4]
    device = Ps_norm.device if device is None else device

    # Predicted absolute poses -> relative poses (torch branch)
    Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm)
    relative_poses_pred = geo_utils.batch_get_relative_pose(Rs_pred, ts_pred)

    # Ground-truth absolute poses -> relative poses
    Ks_invT = getattr(data, 'Ns_invT', None)
    Ks = Ks_invT.transpose(1, 2) if Ks_invT is not None else None
    Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.to(device), Ks.to(device) if Ks is not None else None)
    relative_poses_gt = geo_utils.batch_get_relative_pose(Rs_gt, ts_gt)

    # Ensure same device/dtype
    relative_poses_pred = relative_poses_pred.to(device)
    relative_poses_gt = relative_poses_gt.to(device)

    rot_loss, trans_loss = relative_pose_matrix_loss(
        relative_poses_pred, relative_poses_gt,
        rotation_weight=rotation_weight,
        translation_weight=translation_weight
    )

    return rot_loss, trans_loss
