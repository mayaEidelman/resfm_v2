import torch
from utils import geo_utils
from torch import nn
from torch.nn import functional as F


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
        self.calibrated = conf.get_bool('dataset.calibrated')
        
    def forward(self, pred_cam, data, epoch=None):
        """
        Compute unsupervised pairwise consistency loss using only epipolar constraints
        and geometric consistency.
        
        Args:
            pred_cam: Dictionary containing predicted camera parameters
            data: SceneData object containing matches
            epoch: Current training epoch (optional)
            
        Returns:
            Total pairwise consistency loss
        """
        Ps_pred = pred_cam["Ps_norm"]  # [m, 3, 4] predicted camera matrices
        n_cameras = Ps_pred.shape[0]
        
        # Extract predicted rotations and translations
        Vs_pred = Ps_pred[:, 0:3, 0:3].inverse().transpose(1, 2)  # [m, 3, 3]
        ts_pred = torch.bmm(-Vs_pred.transpose(1, 2), Ps_pred[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()  # [m, 3]
        
        # Initialize loss components
        epipolar_loss = torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        geometric_loss = torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Check if we have matches available
        has_matches = hasattr(data, 'matches') and len(data.matches) > 0
        
        # If no pairwise data is available, return zero loss
        if not has_matches:
            return torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Compute pairwise consistency for all camera pairs
        pair_count = 0
        
        for i in range(n_cameras):
            for j in range(i + 1, n_cameras):
                if (i, j) in data.matches:
                    matches_ij = data.matches[(i, j)]
                    if len(matches_ij['pts1']) > 8:  # Need at least 8 points for fundamental matrix
                        pts1 = matches_ij['pts1']
                        pts2 = matches_ij['pts2']
                        
                        # 1. Epipolar consistency loss
                        F_pred = self._compute_fundamental_matrix(Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j])
                        epipolar_error = self._compute_epipolar_error(F_pred, pts1, pts2)
                        epipolar_loss = epipolar_loss + epipolar_error
                        
                        # 2. Geometric consistency loss (triangulation-based)
                        geometric_error = self._compute_geometric_consistency(
                            Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j], pts1, pts2
                        )
                        geometric_loss = geometric_loss + geometric_error
                        
                        pair_count += 1
        
        # Normalize by number of pairs
        if pair_count > 0:
            epipolar_loss = epipolar_loss / pair_count
            geometric_loss = geometric_loss / pair_count
        else:
            # If no valid pairs, return zero loss
            return torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Combine losses
        total_loss = (self.epipolar_weight * epipolar_loss + 
                     self.geometric_weight * geometric_loss)
        
        if epoch is not None and epoch % 1000 == 0:
            print(f"Pairwise Loss: {total_loss:.6f}, Epipolar: {epipolar_loss:.6f}, Geometric: {geometric_loss:.6f}")
        
        return total_loss
    
    def _compute_fundamental_matrix(self, V1, V2, t1, t2):
        """Compute fundamental matrix from two camera poses."""
        # Get cross product matrices
        t1_cross = torch.tensor([[0, -t1[2], t1[1]], 
                                [t1[2], 0, -t1[0]], 
                                [-t1[1], t1[0], 0]], device=t1.device)
        t2_cross = torch.tensor([[0, -t2[2], t2[1]], 
                                [t2[2], 0, -t2[0]], 
                                [-t2[1], t2[0], 0]], device=t2.device)
        
        # Compute essential matrix
        E = torch.bmm(V1.transpose(1, 2), torch.bmm(t1_cross - t2_cross, V2))
        
        # For calibrated case, fundamental matrix is same as essential matrix
        if self.calibrated:
            return E
        else:
            # For uncalibrated case, need to account for unknown intrinsics
            # This is a simplified version - in practice you'd need the actual K matrices
            return E
    
    def _compute_epipolar_error(self, F, pts1, pts2):
        """Compute symmetric epipolar distance."""
        # Ensure points are in homogeneous coordinates
        if pts1.shape[0] == 2:
            pts1 = torch.cat([pts1, torch.ones(1, pts1.shape[1], device=pts1.device)], dim=0)
        if pts2.shape[0] == 2:
            pts2 = torch.cat([pts2, torch.ones(1, pts2.shape[1], device=pts2.device)], dim=0)
        
        # Compute epipolar lines
        lines1 = torch.bmm(F, pts2)  # F * x2
        lines2 = torch.bmm(F.transpose(1, 2), pts1)  # F^T * x1
        
        # Compute distances
        dist1 = torch.abs(torch.sum(pts1 * lines1, dim=0)) / torch.norm(lines1[:2, :], dim=0)
        dist2 = torch.abs(torch.sum(pts2 * lines2, dim=0)) / torch.norm(lines2[:2, :], dim=0)
        
        # Symmetric epipolar distance
        return torch.mean(dist1 + dist2)
    
    def _compute_geometric_consistency(self, V1, V2, t1, t2, pts1, pts2):
        """
        Compute geometric consistency loss based on triangulation and reprojection.
        This enforces that the predicted poses are consistent with the observed matches.
        """
        # Create camera matrices for triangulation
        P1 = torch.eye(3, 4, device=V1.device)  # First camera at origin
        P2 = torch.cat([V2.transpose(1, 2), -torch.bmm(V2.transpose(1, 2), t2.unsqueeze(-1))], dim=2)
        
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
        pts1_proj = torch.bmm(P1, X_3d)
        pts2_proj = torch.bmm(P2, X_3d)
        
        # Normalize homogeneous coordinates
        pts1_proj = pts1_proj[:2, :] / pts1_proj[2, :].unsqueeze(0)
        pts2_proj = pts2_proj[:2, :] / pts2_proj[2, :].unsqueeze(0)
        
        # Compute reprojection errors
        reproj_error1 = torch.norm(pts1[:2, :] - pts1_proj, dim=0)
        reproj_error2 = torch.norm(pts2[:2, :] - pts2_proj, dim=0)
        
        # Check if points are in front of cameras (positive depth)
        depth1 = torch.bmm(P1[2:3, :], X_3d).squeeze()
        depth2 = torch.bmm(P2[2:3, :], X_3d).squeeze()
        
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
        pairwise_loss = self.pairwise_loss(pred_cam, data, epoch)
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


        loss = self.alpha * ESFMLoss + self.beta * classificationLoss + effective_pairwise_weight * pairwiseLoss

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

