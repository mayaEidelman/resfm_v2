from turtle import shape
import torch
from utils import geo_utils
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np

class ESFMLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, data, epoch=None):
        Ps = pred_cam["Ps_norm"]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)

        return torch.where(projected_points, reproj_err, hinge_loss)[data.valid_pts].mean()

class ESFMLoss_weighted(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.infinity_pts_margin = conf.get_float("loss.infinity_pts_margin")
        self.normalize_grad = conf.get_bool("loss.normalize_grad")

        self.hinge_loss = conf.get_bool("loss.hinge_loss")
        if self.hinge_loss:
            self.hinge_loss_weight = conf.get_float("loss.hinge_loss_weight")
        else:
            self.hinge_loss_weight = 0

    def forward(self, pred_cam, pred_outliers, data, epoch=None):
        Ps = pred_cam["Ps_norm"]  # [m, 3, 4]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, 4] @ [4, n] -> [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            # mark as False points with very small w in homogeneous coordinates, that is, very large u, v in inhomogeneous coordinates
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)  # [m, n], boolean mask #
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        # From homogeneous coordinates to in inhomogeneous coordinates for all projected_points
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)


        # use reprojection error for projected_points, hinge_loss everywhere else
        projected_points = projected_points[data.valid_pts]
        reproj_err = reproj_err[data.valid_pts]
        hinge_loss = hinge_loss[data.valid_pts]

        pred_outliers = pred_outliers.squeeze(dim=-1)
        reproj_err = (1 - pred_outliers) * reproj_err # Outlier-weighted loss
        weightedLoss = torch.where(projected_points, reproj_err, hinge_loss)

        return weightedLoss.mean()

class GT_Loss_Outliers(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, pred_outliers, data, epoch=None):
        gt_outliers = data.outlier_indices[data.x.indices.T[:, 0], data.x.indices.T[:, 1]]

        # Compute class-balanced BCE loss
        outliers_ratio = gt_outliers.sum() / gt_outliers.shape[0]
        weights = (gt_outliers.float() * ((1.0 / outliers_ratio) * 1 - 2)) + 1 #class balancing
        bce_loss = F.binary_cross_entropy(pred_outliers.squeeze(), gt_outliers.float(), weight=weights)

        return bce_loss

class OutliersLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = GT_Loss_Outliers(conf)

    def forward(self, pred_outliers, data):
        loss = self.outliers_loss(pred_outliers, data)
        return loss

class PairwiseConsistencyLoss(nn.Module):
    """
    Unsupervised pairwise consistency loss that enforces geometric consistency 
    between predicted camera poses and pairwise matches.
    """
    def __init__(self, conf):
        super().__init__()
        self.pairwise_weight = conf.get_float("loss.pairwise_weight", default=1.0)
        self.epipolar_weight = conf.get_float("loss.epipolar_weight", default=1.0)
        self.geometric_weight = conf.get_float("loss.geometric_weight", default=0.5)
        self.rotation_weight = conf.get_float("loss.rotation_weight", default=1.0)
        self.translation_weight = conf.get_float("loss.translation_weight", default=1.0)
        self.calibrated = conf.get_bool('dataset.calibrated')
        
    def forward(self, pred_cam, data, epoch=None):
        """
        Compute unsupervised pairwise consistency loss using relative pose consistency.
        
        Args:
            pred_cam: Dictionary containing predicted camera parameters
            data: SceneData object containing matches
            epoch: Current training epoch (optional)
            
        Returns:
            Total pairwise consistency loss
        """
        Ps_pred = pred_cam["Ps_norm"]  # [m, 3, 4] predicted camera matrices
        n_cameras = Ps_pred.shape[0]
        
        # Extract predicted rotations and camera centers
        Vs_pred = Ps_pred[:, 0:3, 0:3].inverse().transpose(1, 2)  # [m, 3, 3]
        ts_pred = torch.bmm(-Vs_pred.transpose(1, 2), Ps_pred[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()  # [m, 3]
        
        # Initialize loss components
        epipolar_loss = torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        geometric_loss = torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        pairwise_loss = torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Check if we have matches available
        has_matches = hasattr(data, 'matches') and len(data.matches) > 0
        
        # If no pairwise data is available, return zero loss
        if not has_matches:
            return torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Get camera intrinsics if available
        Ns = data.Ns.cpu().numpy()
        K = data.Ns_invT.transpose(1, 2).cpu().numpy()

        
        # Compute pairwise consistency for all camera pairs
        pair_count = 0
        
        for i in range(n_cameras):
            for j in range(i + 1, n_cameras):
                if (i, j) in data.matches:
                    matches_ij = data.matches[(i, j)]
                    if matches_ij['num_matches'] > 8:  # Need at least 8 points for essential matrix
                        pts1 = matches_ij['pts1']
                        pts2 = matches_ij['pts2']
                        
                        # 1. Epipolar consistency loss
                        F_pred = geo_utils.get_fundamental_from_V_t(Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j])
                        epipolar_error = self._compute_epipolar_error(F_pred, pts1, pts2)
                        epipolar_loss = epipolar_loss + epipolar_error
                        
                        # 2. Geometric consistency loss (existing)
                        # geometric_error = self._compute_geometric_consistency_from_pred(
                        #     pred_cam, data, i, j
                        # )
                        # geometric_loss = geometric_loss + geometric_error
                        
                        # 3. Relative pose consistency loss
                        # Check if we already computed the relative pose/fundamental matrix for this pair
                        if ("R_gt" not in matches_ij or "t_gt" not in matches_ij) and "F_gt" not in matches_ij:
                            # Compute ground truth relative pose from essential matrix or fundamental matrix
                            try:
                                result1, result2 = self._compute_ground_truth_relative_pose(pts1, pts2, K[i] if K is not None else None)
                                if result1 is not None:
                                    if result2 is not None:
                                        # We got R, t (essential matrix case)
                                        matches_ij["R_gt"] = torch.from_numpy(result1).float()
                                        matches_ij["t_gt"] = torch.from_numpy(result2.ravel()).float()
                                    else:
                                        # We got F, None (fundamental matrix case)
                                        print(f"Computed fundamental matrix for pair ({i}, {j})")
                                        matches_ij["F_gt"] = torch.from_numpy(result1).float()
                                else:
                                    # Skip this pair if computation failed
                                    continue
                            except Exception as e:
                                print(f"Warning: Failed to compute relative pose/fundamental matrix for pair ({i}, {j}): {e}")
                                # If computation fails, skip this pair
                                continue
                        
                        # Handle both relative pose (R,t) and fundamental matrix (F) cases
                        if "R_gt" in matches_ij and "t_gt" in matches_ij:
                            # Relative pose consistency loss (when camera intrinsics available)
                            R_gt = matches_ij["R_gt"].to(Ps_pred.device)
                            t_gt = matches_ij["t_gt"].to(Ps_pred.device)
                            
                            # Compute consistency losses
                            rot_loss, trans_loss = self._compute_pairwise_consistency(
                                Vs_pred[i], Vs_pred[j],
                                ts_pred[i], ts_pred[j],
                                R_gt, t_gt
                            )

                            consistency_loss = (self.rotation_weight * rot_loss + 
                                              self.translation_weight * trans_loss)
                            pairwise_loss = pairwise_loss + consistency_loss
                            
                        elif "F_gt" in matches_ij:
                            print("no R_gt and t_gt then computing loss from F_gt")
                            # Fundamental matrix consistency loss (when no camera intrinsics)
                            F_gt = matches_ij["F_gt"].to(Ps_pred.device)
                            F_pred = geo_utils.get_fundamental_from_V_t(Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j])
                            
                            # Compare ground truth and predicted fundamental matrices
                            # Use Frobenius norm of the difference, normalized by F_gt norm
                            F_diff = F_pred - F_gt
                            fundamental_loss = torch.norm(F_diff, 'fro') / (torch.norm(F_gt, 'fro') + 1e-8)
                            pairwise_loss = pairwise_loss + fundamental_loss
                        
                        pair_count += 1
        
        # Normalize by number of pairs
        if pair_count > 0:
            epipolar_loss = epipolar_loss / pair_count
            pairwise_loss = pairwise_loss / pair_count
            # geometric_loss = geometric_loss / pair_count
        else:
            # If no valid pairs, return zero loss
            return torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Combine losses
        total_loss = (self.epipolar_weight * epipolar_loss + 
                     self.pairwise_weight * pairwise_loss)
        print(f"Pairwise Loss: {total_loss:.6f}, Epipolar: {epipolar_loss:.6f}, "
                  f"Pairwise Consistency: {pairwise_loss:.6f}")
        
        if epoch is not None and epoch % 1000 == 0:
            print(f"Pairwise Loss: {total_loss:.6f}, Epipolar: {epipolar_loss:.6f}, "
                  f"Pairwise Consistency: {pairwise_loss:.6f}")
        
        return total_loss
    
    def _compute_pairwise_consistency(self, Ri, Rj, ti, tj, R_gt, t_gt):
        """
        Compute pairwise consistency loss between predicted cameras (i,j) and gt relative pose.
        Args:
            Ri, Rj: [3,3] predicted rotation matrices
            ti, tj: [3] predicted translations
            R_gt: [3,3] ground truth relative rotation
            t_gt: [3] ground truth relative translation (up to scale)
        Returns:
            rotation loss, translation loss
        """
        # Predicted relative pose
        R_pred = Rj @ Ri.transpose(0, 1)
        t_pred = (tj - ti)

        # Normalize translation
        t_pred = t_pred / (t_pred.norm() + 1e-8)

        # Losses
        rot_loss = self._geodesic_distance(R_pred, R_gt)
        trans_loss = self._translation_direction_loss(t_pred, t_gt)

        return rot_loss, trans_loss
    
    def _geodesic_distance(self, R1, R2, eps=1e-7):
        """
        Geodesic distance between two rotations.
        Args:
            R1, R2: [3, 3] rotation matrices
        Returns:
            scalar geodesic distance in radians
        """
        R = R1 @ R2.transpose(0, 1)
        trace = torch.clamp((torch.trace(R) - 1) / 2, -1 + eps, 1 - eps)
        return torch.acos(trace)
    
    def _translation_direction_loss(self, t1, t2, eps=1e-7):
        """
        Cosine similarity loss for translation directions.
        Args:
            t1, t2: [3] translation vectors
        Returns:
            scalar loss (0 = aligned, 1 = opposite)
        """
        t1 = t1 / (t1.norm() + eps)
        t2 = t2 / (t2.norm() + eps)
        return 1.0 - torch.abs(torch.dot(t1, t2))
    
    def _compute_epipolar_error(self, F, pts1, pts2):
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

    def _compute_geometric_consistency_from_pred(self, pred_cam, data, i, j):
        """
        Compute geometric consistency loss by reprojecting the model's predicted 3D points
        with the model's predicted camera matrices and comparing to observed 2D points.
        This avoids re-triangulating points and re-deriving cameras.
        """
        Ps_pred = pred_cam["Ps_norm"]  # [m, 3, 4]
        pts3D_pred = pred_cam["pts3D"]  # [4, n]

        device = Ps_pred.device

        # Predicted camera matrices for the two views
        P1 = Ps_pred[i]
        P2 = Ps_pred[j]

        # Determine which 3D points are observed in both cameras i and j
        # Use the same visibility criterion as in pairwise match extraction
        M = data.M  # [2*m, n]
        visible_i = M[2 * i, :] > 0
        visible_j = M[2 * j, :] > 0
        visible_both = (visible_i & visible_j).to(device)

        if visible_both.sum().item() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Select corresponding predicted 3D points
        X_3d = pts3D_pred[:, visible_both]  # [4, k]

        # Project to both cameras
        x1_h = P1 @ X_3d  # [3, k]
        x2_h = P2 @ X_3d  # [3, k]

        # Depth positivity check (points in front of both cameras)
        depth1 = x1_h[2, :]
        depth2 = x2_h[2, :]
        behind_camera_penalty = torch.mean(torch.relu(-depth1) + torch.relu(-depth2))

        # Normalize homogeneous coordinates with safe denominators
        eps = 1e-8
        denom1 = torch.where(depth1.abs() > eps, depth1, torch.ones_like(depth1))
        denom2 = torch.where(depth2.abs() > eps, depth2, torch.ones_like(depth2))
        x1 = x1_h[:2, :] / denom1.unsqueeze(0)
        x2 = x2_h[:2, :] / denom2.unsqueeze(0)

        # Observed normalized 2D measurements for these points
        norm_M_device = data.norm_M.to(device)
        obs1 = norm_M_device[2 * i:2 * i + 2, :][:, visible_both]
        obs2 = norm_M_device[2 * j:2 * j + 2, :][:, visible_both]

        # Valid mask: points with positive depth in both views
        valid = (depth1 > eps) & (depth2 > eps)

        if valid.sum() > 0:
            x1 = x1[:, valid]
            x2 = x2[:, valid]
            obs1 = obs1[:, valid]
            obs2 = obs2[:, valid]
            # Reprojection errors
            reproj_error1 = torch.norm(obs1 - x1, dim=0)
            reproj_error2 = torch.norm(obs2 - x2, dim=0)
            reproj_loss = torch.mean(reproj_error1 + reproj_error2)
        else:
            reproj_loss = torch.tensor(0.0, device=device)

        # Total geometric consistency loss
        geometric_loss = reproj_loss + 0.1 * behind_camera_penalty

        return geometric_loss

    def _compute_geometric_consistency(self, V1, V2, t1, t2, pts1, pts2):
        """
        Compute geometric consistency loss based on triangulation and reprojection.
        This enforces that the predicted poses are consistent with the observed matches.
        """
        device = V1.device
        pts1 = pts1.to(device)
        pts2 = pts2.to(device)
        # Create camera matrices for triangulation
        P1 = torch.eye(3, 4, device=device)  # First camera at origin
        P2 = geo_utils.get_camera_matrix_from_Vt(V2, t2)
        
        # Ensure points are in homogeneous coordinates
        if pts1.shape[0] == 2:
            pts1_homo = torch.cat([pts1, torch.ones(1, pts1.shape[1], device=pts1.device)], dim=0)
        else:
            pts1_homo = pts1
            
        if pts2.shape[0] == 2:
            pts2_homo = torch.cat([pts2, torch.ones(1, pts2.shape[1], device=pts2.device)], dim=0)
        else:
            pts2_homo = pts2
        
        # Triangulate 3D points
        X_3d = self._triangulate_points(P1, P2, pts1_homo, pts2_homo)
        # Project back to both cameras
        pts1_proj = P1 @ X_3d
        pts2_proj = P2 @ X_3d
        
        # Normalize homogeneous coordinates
        pts1_proj = pts1_proj[:2, :] / pts1_proj[2, :].unsqueeze(0)
        pts2_proj = pts2_proj[:2, :] / pts2_proj[2, :].unsqueeze(0)
        
        # Compute reprojection errors
        reproj_error1 = torch.norm(pts1[:2, :] - pts1_proj, dim=0)
        reproj_error2 = torch.norm(pts2[:2, :] - pts2_proj, dim=0)
        
        # Check if points are in front of cameras (positive depth)
        depth1 = P1[2:3, :] @ X_3d
        depth2 = P2[2:3, :] @ X_3d
        
        # Penalize points behind cameras
        behind_camera_penalty = torch.mean(torch.relu(-depth1) + torch.relu(-depth2))
        
        # Total geometric consistency loss
        reproj_loss = torch.mean(reproj_error1 + reproj_error2)
        geometric_loss = reproj_loss + 0.1 * behind_camera_penalty
        
        return geometric_loss
    
    def _triangulate_points(self, P1, P2, pts1, pts2):
        """
        Triangulate 3D points from two camera views using DLT method.
        """
        batch_size = pts1.shape[1]
        X_3d = torch.zeros(4, batch_size, device=pts1.device)
        
        for i in range(batch_size):
            # Build the DLT matrix for this point
            A = torch.zeros(4, 4, device=pts1.device)
            
            # First camera constraint
            A[0, :] = pts1[0, i] * P1[2, :] - P1[0, :]
            A[1, :] = pts1[1, i] * P1[2, :] - P1[1, :]
            
            # Second camera constraint
            A[2, :] = pts2[0, i] * P2[2, :] - P2[0, :]
            A[3, :] = pts2[1, i] * P2[2, :] - P2[1, :]
            
            # Solve using SVD
            U, S, Vt = torch.svd(A)
            X_3d[:, i] = Vt[-1, :]  # Last row of V^T is the solution
        
        return X_3d

    def _compute_ground_truth_relative_pose(self, pts1, pts2, K):
        """
        Compute ground truth relative pose from matched points using essential matrix.
        If camera intrinsics are not available, compute fundamental matrix instead.
        
        Args:
            pts1, pts2: Corresponding points [2, n] or [3, n]
            K: Camera intrinsics matrix [3, 3] or None
            
        Returns:
            If K is provided:
                R: Relative rotation [3, 3] or None if computation fails
                t: Relative translation [3, 1] or None if computation fails
            If K is None:
                F: Fundamental matrix [3, 3] or None if computation fails
                None: Second return value is None to maintain consistent return signature
        """
        try:
            # Ensure points are in the right format for OpenCV
            if isinstance(pts1, torch.Tensor):
                pts1_np = pts1.cpu().numpy()
            else:
                pts1_np = pts1
                
            if isinstance(pts2, torch.Tensor):
                pts2_np = pts2.cpu().numpy()
            else:
                pts2_np = pts2
            
            # Ensure points are 2D for essential matrix computation
            if pts1_np.shape[0] == 3:
                pts1_np = pts1_np[:2, :]
            if pts2_np.shape[0] == 3:
                pts2_np = pts2_np[:2, :]
            
            # Transpose to get (n, 2) format for OpenCV
            pts1_np = pts1_np.T
            pts2_np = pts2_np.T
            # If we have camera intrinsics, compute essential matrix
            if K is not None:
                if isinstance(K, torch.Tensor):
                    K_np = K.cpu().numpy()
                else:
                    K_np = K
                
                # Compute essential matrix
                E, mask = cv2.findEssentialMat(pts1_np, pts2_np, K_np)
                if E is not None and E.shape == (3, 3):
                    # Recover pose from essential matrix
                    retval, R, t, mask = cv2.recoverPose(E, pts1_np, pts2_np, K_np)
                    return R, t
                else:
                    return None, None
            else:
                # If no intrinsics, compute fundamental matrix instead
                F, mask = cv2.findFundamentalMat(pts1_np, pts2_np, cv2.FM_RANSAC)
                
                if F is not None and F.shape == (3, 3):
                    # Return fundamental matrix for comparison with predicted F
                    return F, None
                else:
                    return None, None
                    
        except Exception as e:
            # Return None if computation fails
            return None, None
    
    def precompute_relative_poses(self, data):
        """
        Pre-compute all relative poses for efficiency.
        This can be called once before training to avoid repeated computation.
        
        Args:
            data: SceneData object containing matches
        """
        if not hasattr(data, 'matches') or len(data.matches) == 0:
            return
        
        K = getattr(data, 'K', None)
        
        if K is not None:
            print("Pre-computing relative poses for all camera pairs...")
        else:
            print("Pre-computing fundamental matrices for all camera pairs (no intrinsics available)...")
        
        computed_count = 0
        
        for (i, j), matches_ij in data.matches.items():
            if matches_ij['num_matches'] > 8:
                try:
                    # Check if already computed
                    if ("R_gt" not in matches_ij or "t_gt" not in matches_ij) and "F_gt" not in matches_ij:
                        pts1 = matches_ij['pts1']
                        pts2 = matches_ij['pts2']
                        
                        result1, result2 = self._compute_ground_truth_relative_pose(pts1, pts2, K)
                        
                        if result1 is not None:
                            if result2 is not None:
                                # Store computed relative poses (essential matrix case)
                                matches_ij["R_gt"] = torch.from_numpy(result1).float()
                                matches_ij["t_gt"] = torch.from_numpy(result2.ravel()).float()
                                computed_count += 1
                            else:
                                # Store computed fundamental matrix (no intrinsics case)
                                matches_ij["F_gt"] = torch.from_numpy(result1).float()
                                computed_count += 1
                        else:
                            print(f"Warning: Failed to compute relative pose for pair ({i}, {j})")
                except Exception as e:
                    print(f"Error computing relative pose for pair ({i}, {j}): {e}")
        
        print(f"Successfully computed {computed_count} relative poses")
    
    def clear_relative_poses(self, data):
        """
        Clear all stored relative poses from matches data.
        Useful for freeing memory or forcing recomputation.
        
        Args:
            data: SceneData object containing matches
        """
        if not hasattr(data, 'matches'):
            return
        
        cleared_count = 0
        for (i, j), matches_ij in data.matches.items():
            if "R_gt" in matches_ij:
                del matches_ij["R_gt"]
                cleared_count += 1
            if "t_gt" in matches_ij:
                del matches_ij["t_gt"]
                cleared_count += 1
        
        if cleared_count > 0:
            print(f"Cleared {cleared_count} stored relative poses")


class CombinedLoss_Pairwise(nn.Module):
    """
    Combined loss function that includes both the original ESFM loss and pairwise consistency loss.
    """
    def __init__(self, conf):
        super().__init__()
        self.esfm_loss = ESFMLoss(conf)
        self.pairwise_loss = PairwiseConsistencyLoss(conf)
        self.esfm_weight = conf.get_float("loss.esfm_weight", default=1.0)
        self.pairwise_weight = conf.get_float("loss.pairwise_weight", default=1.0)
        
    def forward(self, pred_cam, data, epoch=None):
        esfm_loss = self.esfm_loss(pred_cam, data, epoch)
        pairwise_loss = self.pairwise_loss(pred_cam, data, epoch)
        
        # Check if pairwise loss is effectively zero (no pairwise data available)
        has_pairwise_data = (hasattr(data, 'relative_poses') and len(data.relative_poses) > 0) or \
                           (hasattr(data, 'matches') and len(data.matches) > 0)
        
        # If no pairwise data is available, set pairwise weight to 0
        effective_pairwise_weight = self.pairwise_weight if has_pairwise_data else 0.0
        
        total_loss = self.esfm_weight * esfm_loss + effective_pairwise_weight * pairwise_loss
        
        if epoch is not None and epoch % 1000 == 0:
            print(f"Combined Loss: {total_loss:.6f}, ESFM: {esfm_loss:.6f}, Pairwise: {pairwise_loss:.6f}")
            if not has_pairwise_data:
                print("Warning: No pairwise data available, pairwise loss weight set to 0")
        
        return total_loss

class CombinedLoss_Outliers(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = OutliersLoss(conf)
        self.weighted_ESFM_loss = ESFMLoss_weighted(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        self.beta = conf.get_float('loss.classification_loss_weight')


    def forward(self, pred_cam, pred_outliers, pred_weights_M, data, epoch=None):
        classificationLoss = torch.tensor([0], device=pred_outliers.device)
        ESFMLoss = torch.tensor([0], device=pred_outliers.device)

        # Reprojection loss (geometric loss)
        if self.alpha:
            ESFMLoss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data)

        # Outlier classification loss
        if self.beta:
            classificationLoss = self.outliers_loss(pred_outliers, data)


        loss = self.alpha * ESFMLoss + self.beta * classificationLoss

        return loss

class CombinedLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = OutliersLoss(conf)
        self.weighted_ESFM_loss = ESFMLoss_weighted(conf)
        self.pairwise_loss = PairwiseConsistencyLoss(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        self.beta = conf.get_float('loss.classification_loss_weight')
        self.pairwise_weight = conf.get_float("loss.pairwise_weight", default=1.0)


    def forward(self, pred_cam, pred_outliers, pred_weights_M, data, epoch=None):
        classificationLoss = torch.tensor([0], device=pred_outliers.device)
        ESFMLoss = torch.tensor([0], device=pred_outliers.device)
        pairwiseLoss = torch.tensor([0], device=pred_outliers.device)

        # Check if pairwise loss is effectively zero (no pairwise data available)
        has_pairwise_data = (hasattr(data, 'relative_poses') and len(data.relative_poses) > 0) or \
                           (hasattr(data, 'matches') and len(data.matches) > 0)
        # If no pairwise data is available, set pairwise weight to 0
        effective_pairwise_weight = self.pairwise_weight if has_pairwise_data else 0.0
        # Reprojection loss (geometric loss)
        if self.alpha:
            ESFMLoss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data)

        # Outlier classification loss
        if self.beta:
            classificationLoss = self.outliers_loss(pred_outliers, data)

        # Pairwise loss
        if effective_pairwise_weight:
            pairwiseLoss = self.pairwise_loss(pred_cam, data, epoch)
            print("Pairwise Loss:", pairwiseLoss)


        loss = self.alpha * ESFMLoss + self.beta * classificationLoss + effective_pairwise_weight * pairwiseLoss
        print(loss)

        return loss


class GTLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_cam, data, epoch=None):
        # Get orientation
        Vs_gt = data.y[:, 0:3, 0:3].inverse().transpose(1, 2)
        if self.calibrated:
            Rs_gt = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs_gt).transpose(1, 2))

        # Get Location
        t_gt = -torch.bmm(data.y[:, 0:3, 0:3].inverse(), data.y[:, 0:3, 3].unsqueeze(-1)).squeeze()

        # Normalize scene by points
        # trans = pts3D_gt.mean(dim=1)
        # scale = (pts3D_gt - trans.unsqueeze(1)).norm(p=2, dim=0).mean()

        # Normalize scene by cameras
        trans = t_gt.mean(dim=0)
        scale = (t_gt - trans).norm(p=2, dim=1).mean()

        t_gt = (t_gt - trans)/scale
        new_Ps = geo_utils.batch_get_camera_matrix_from_Vt(Vs_gt, t_gt)

        Vs_invT = pred_cam["Ps_norm"][:, 0:3, 0:3]
        Vs = torch.inverse(Vs_invT).transpose(1, 2)
        ts = torch.bmm(-Vs.transpose(1, 2), pred_cam["Ps"][:, 0:3, 3].unsqueeze(dim=-1)).squeeze()

        # Translation error
        translation_err = (t_gt - ts).norm(p=2, dim=1)

        # Calculate error
        if self.calibrated:
            Rs = geo_utils.rot_to_quat(torch.bmm(data.Ns_invT, Vs).transpose(1, 2))
            orient_err = (Rs - Rs_gt).norm(p=2, dim=1)
        else:
            Vs_gt = Vs_gt / Vs_gt.norm(p='fro', dim=(1, 2), keepdim=True)
            Vs = Vs / Vs.norm(p='fro', dim=(1, 2), keepdim=True)
            orient_err = torch.min((Vs - Vs_gt).norm(p='fro', dim=(1, 2)), (Vs + Vs_gt).norm(p='fro', dim=(1, 2)))

        orient_loss = orient_err.mean()
        tran_loss = translation_err.mean()
        loss = orient_loss + tran_loss

        if epoch is not None and epoch % 1000 == 0:
            # Print loss
            print("loss = {}, orient err = {}, trans err = {}".format(loss, orient_loss, tran_loss))

        return loss
