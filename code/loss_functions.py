from turtle import shape
import torch
from utils import geo_utils, pairwise_utils
from torch import nn
from torch.nn import functional as F
import cv2
import numpy as np
import traceback

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

        pred_outliers = torch.nan_to_num(pred_outliers.squeeze(dim=-1), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        reproj_err = (1 - pred_outliers) * reproj_err # Outlier-weighted loss
        weightedLoss = torch.where(projected_points, reproj_err, hinge_loss)

        return weightedLoss.mean()

class GT_Loss_Outliers(nn.Module):
    def __init__(self, conf):
        super().__init__()

    def forward(self, pred_outliers, data, epoch=None):
        gt_outliers = data.outlier_indices[data.x.indices.T[:, 0], data.x.indices.T[:, 1]]

        # # Align devices and types
        device = pred_outliers.device
        gt = torch.nan_to_num(gt_outliers.float().to(device), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        pred = torch.nan_to_num(pred_outliers.squeeze().to(device), nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # Compute class-balanced BCE loss with epsilon guard
        pos_ratio = gt.mean().clamp_min(1e-6)
        weights = (gt * ((1.0 / pos_ratio) - 2.0)) + 1.0

        bce_loss = F.binary_cross_entropy(pred, gt, weight=weights)

                # Compute class-balanced BCE loss
        # outliers_ratio = gt_outliers.sum() / gt_outliers.shape[0]
        # weights = (gt_outliers.float() * ((1.0 / outliers_ratio) * 1 - 2)) + 1 #class balancing
        # bce_loss = F.binary_cross_entropy(pred_outliers.squeeze(), gt_outliers.float(), weight=weights)


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
        self.absolute_pose_weight = conf.get_float("loss.absolute_pose_weight", default=1.0)
        self.pairwise_pose_weight = conf.get_float("loss.pairwise_pose_weight", default=1.0)
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward1(self, pred_cam, data, epoch=None):
        pairwise_epipole_loss = pairwise_utils.pairwise_epipole_loss(data, pred_cam)
        return pairwise_epipole_loss

    def forward(self, pred_cam, data, epoch=None):
        Ps_pred = pred_cam["Ps_norm"]

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_pred)
        Ks_invT = getattr(data, 'Ns_invT', None)
        Ks = Ks_invT.transpose(1, 2) if Ks_invT is not None else None
        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.to(Ps_pred.device), Ks.to(Ps_pred.device) if Ks is not None else None)

        absolute_pose_loss = self.absolut_Rt_loss(Rs_pred, ts_pred, Rs_gt, ts_gt, Ps_pred.device)
        # pairwise_pose_loss = self.pairwise_Rt_loss(Rs_pred, ts_pred, Rs_gt, ts_gt, Ps_pred.device)
        pairwise_epipole_loss = pairwise_utils.pairwise_epipole_loss(data, Rs_pred, ts_pred, Ps_pred.device)

        total_loss = (absolute_pose_loss + pairwise_epipole_loss)/2
        # total_loss = (pairwise_pose_loss )
        # try:
        #     print(f"Pairwise Loss: {total_loss.item():.6f}, Absolute Pose: {absolute_pose_loss.item():.6f}, Pairwise Consistency: {pairwise_pose_loss.item():.6f}")
        # except Exception:
        #     pass
        return total_loss
        
    def forward2(self, pred_cam, data, epoch=None):
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
                        epipolar_error = pairwise_utils.compute_epipolar_constraint_error(F_pred, pts1, pts2)
                        epipolar_loss = epipolar_loss + epipolar_error
                        
                        # 2. Geometric consistency loss (existing)
                        geometric_error = self._compute_geometric_consistency_from_pred(
                            pred_cam, data, i, j
                        )
                        geometric_loss = geometric_loss + geometric_error
                        
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
                            # F_pred = geo_utils.get_fundamental_from_V_t(Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j])
                            
                            # Compare ground truth and predicted fundamental matrices
                            # Use Frobenius norm of the difference, normalized by F_gt norm
                            F_diff = F_pred - F_gt
                            fundamental_loss = torch.norm(F_diff, 'fro') / (torch.norm(F_gt, 'fro') + 1e-8)
                            pairwise_loss = pairwise_loss + fundamental_loss
                        
                        pair_count += 1
        
        # Normalize by number of pairs
        if pair_count > 0:
            epipolar_loss = epipolar_loss / pair_count
            # pairwise_loss = pairwise_loss / pair_count
            geometric_loss = geometric_loss / pair_count
        else:
            # If no valid pairs, return zero loss
            return torch.tensor(0.0, device=Ps_pred.device, requires_grad=True)
        
        # Compute absolute pose consistency loss (same as evaluation.py)
        absolute_pose_loss = self.absolut_Rt_loss(pred_cam, data, Ps_pred.device)

        # Combine losses
        total_loss = (self.epipolar_weight * epipolar_loss + 
                     self.pairwise_weight * pairwise_loss +
                     self.absolute_pose_weight * absolute_pose_loss)
        
        if epoch is not None and epoch % 1000 == 0:
            print(f"Pairwise Loss: {total_loss:.6f}, Epipolar: {epipolar_loss:.6f}, "
                  f"Pairwise Consistency: {pairwise_loss:.6f}, Absolute Pose: {absolute_pose_loss:.6f}")
        
        return total_loss

    def pairwise_Rt_loss(self, Rs_pred, ts_pred, Rs_gt, ts_gt, device=None):
        pairwise_pose_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if self.pairwise_pose_weight > 0.0:
            try:
                rotation_loss, translation_loss = pairwise_utils.compute_pairwise_pose_consistency(Rs_pred, ts_pred, Rs_gt, ts_gt, device)
                pairwise_pose_loss = (rotation_loss + translation_loss)/2
            except Exception as e:
                print(f"Warning: Failed to compute pairwise pose consistency loss: {e}")
                traceback.print_exc()
        
        return pairwise_pose_loss
        

    def absolut_Rt_loss(self, Rs_pred, ts_pred, Rs_gt, ts_gt, device):
        absolute_pose_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if self.absolute_pose_weight > 0.0:
            try:
                rotation_loss, translation_loss = pairwise_utils.compute_absolute_pose_consistency(Rs_pred, ts_pred, Rs_gt, ts_gt, self.calibrated, device)
                absolute_pose_loss = (rotation_loss + translation_loss)/2
            except Exception as e:
                print(f"Warning: Failed to compute absolute pose consistency loss: {e}")
                traceback.print_exc()
        
        return absolute_pose_loss
    
    
    


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
        has_pairwise_data = (hasattr(data, 'matches') and len(data.matches) > 0)
        # If no pairwise data is available, set pairwise weight to 0
        # effective_pairwise_weight = self.pairwise_weight if has_pairwise_data else 0.0
        # Reprojection loss (geometric loss)
        if self.alpha:
            ESFMLoss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data)

        # Outlier classification loss
        if self.beta:
            classificationLoss = self.outliers_loss(pred_outliers, data)

        # Pairwise loss
        if self.pairwise_weight:
            pairwiseLoss = self.pairwise_loss(pred_cam, data, epoch)
            # print("Pairwise Loss:", pairwiseLoss)


        loss = self.alpha * ESFMLoss + self.beta * classificationLoss + 0.1 * pairwiseLoss

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
