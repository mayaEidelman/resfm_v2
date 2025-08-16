import torch
import numpy as np
from utils import geo_utils, dataset_utils
import cv2


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


# def extract_pairwise_matches_from_scene(data, min_matches=8):
#     """
#     Extract pairwise matches from scene data for computing relative poses.
    
#     Args:
#         data: SceneData object containing M matrix and camera information
#         min_matches: Minimum number of matches required for a pair
        
#     Returns:
#         matches_dict: Dictionary with keys (i, j) containing match data
#     """
#     print("Extracting pairwise matches from scene data for computing relative poses - pairwise utils")
#     matches_dict = {}
#     n_cameras = data.y.shape[0]
    
#     # Extract 2D points for each camera
#     M = data.M  # [2*m, n] where m is number of cameras
#     valid_points = data.valid_pts  # optional if used

#     # Precompute visibility mask for each camera
#     # visible[i, :] = bool mask of points visible in camera i
#     visible = (M[0::2, :] > 0)  # take only x-coordinates, shape [m, n]

#     for i in range(n_cameras):
#         pts_i = M[2*i:2*i+2, :]  # [2, n]

#         # Compare i with all j > i at once
#         common_visible = visible[i] & visible[i+1:]  # shape [m-i-1, n]

#         # Count number of matches per pair
#         num_matches = common_visible.sum(axis=1)

#         # Filter pairs with enough matches
#         valid_js = np.where(num_matches >= min_matches)[0] + (i+1)

#         for idx, j in enumerate(valid_js):
#             mask = common_visible[valid_js[idx] - (i+1)]
#             pts1 = pts_i[:, mask]
#             pts2 = M[2*j:2*j+2, :][:, mask]

#             matches_dict[(i, j)] = {
#                 'pts1': pts1,
#                 'pts2': pts2,
#                 'num_matches': pts1.shape[1]
#             }

#     return matches_dict

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
            pts1, pts2, K1, K2, method='8pt' if calibrated else '8pt'
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
    data.relative_poses = relative_poses
    
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