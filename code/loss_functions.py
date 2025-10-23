import torch
from utils import geo_utils
from torch import nn
from torch.nn import functional as F
import traceback
from utils import pairwise_utils

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
        Ps = pred_cam["Ps_norm"]  # [m, 3, 4]
        pts_2d = Ps @ pred_cam["pts3D"]  # [m, 3, 4] @ [4, n] -> [m, 3, n]

        # Normalize gradient
        if self.normalize_grad:
            pts_2d.register_hook(lambda grad: F.normalize(grad, dim=1) / data.valid_pts.sum())

        # Get point for reprojection loss
        if self.hinge_loss:
            # mark as False points with very small w in homogeneous coordinates, that is, very large u, v in inhomogeneous coordinates
            projected_points = geo_utils.get_positive_projected_pts_mask(pts_2d, self.infinity_pts_margin)  # [m, n], boolean mask
        else:
            projected_points = geo_utils.get_projected_pts_mask(pts_2d, self.infinity_pts_margin)

        # Calculate hinge Loss
        hinge_loss = (self.infinity_pts_margin - pts_2d[:, 2, :]) * self.hinge_loss_weight

        # Calculate reprojection error
        # From homogeneous coordinates to in inhomogeneous coordinates for all projected_points
        pts_2d = (pts_2d / torch.where(projected_points, pts_2d[:, 2, :], torch.ones_like(projected_points).float()).unsqueeze(dim=1))
        reproj_err = (pts_2d[:, 0:2, :] - data.norm_M.reshape(Ps.shape[0], 2, -1)).norm(dim=1)
        
        # use reprojection error for projected_points, hinge_loss everywhere else
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

class PairwiseConsistencyLoss_v1(nn.Module):
    """
    Unsupervised pairwise consistency loss that enforces geometric consistency 
    between predicted camera poses and pairwise matches.
    """
    def __init__(self, conf):
        super().__init__()
        self.absolute_pose_weight = conf.get_float("loss.absolute_pose_weight", default=0.5)
        self.pairwise_pose_weight = conf.get_float("loss.pairwise_pose_weight", default=0.5)
        self.calibrated = conf.get_bool('dataset.calibrated')

    def forward(self, pred_cam, data, epoch=None):
        Ps_pred = pred_cam["Ps_norm"]
        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_pred)
        pairwise_epipole_loss = pairwise_utils.pairwise_epipole_loss(data, Rs_pred, ts_pred, Ps_pred.device)
        return pairwise_epipole_loss

    def forward1(self, pred_cam, data, epoch=None):
        Ps_pred = pred_cam["Ps_norm"]

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_pred)
        Ks_invT = getattr(data, 'Ns_invT', None)
        Ks = Ks_invT.transpose(1, 2) if Ks_invT is not None else None
        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.to(Ps_pred.device), Ks.to(Ps_pred.device) if Ks is not None else None)

        absolute_pose_loss = self.absolut_Rt_loss(Rs_pred, ts_pred, Rs_gt, ts_gt, Ps_pred.device)
        # pairwise_pose_loss = self.pairwise_Rt_loss(Rs_pred, ts_pred, Rs_gt, ts_gt, Ps_pred.device)
        pairwise_epipole_loss = pairwise_utils.pairwise_epipole_loss(data, Rs_pred, ts_pred, Ps_pred.device)

        total_loss =  self.absolute_pose_weight * absolute_pose_loss + self.pairwise_pose_weight * pairwise_epipole_loss
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


class CombinedLoss_outliers(nn.Module):
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

        # print(self.alpha * ESFMLoss, self.beta * classificationLoss, self.alpha, self.beta)
        loss = self.alpha * ESFMLoss + self.beta * classificationLoss

        return loss

class CombinedLoss_v1(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = OutliersLoss(conf)
        self.weighted_ESFM_loss = ESFMLoss_weighted(conf)
        self.pairwise_loss = PairwiseConsistencyLoss(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        self.beta = conf.get_float('loss.classification_loss_weight')
        self.gamma = conf.get_float('loss.pairwise_loss_weight')


    def forward(self, pred_cam, pred_outliers, pred_weights_M, data, epoch=None):
        classificationLoss = torch.tensor([0], device=pred_outliers.device)
        ESFMLoss = torch.tensor([0], device=pred_outliers.device)
        pairwiseLoss = torch.tensor([0], device=pred_outliers.device)

        # Reprojection loss (geometric loss)
        if self.alpha:
            ESFMLoss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data)

        # Outlier classification loss
        if self.beta:
            classificationLoss = self.outliers_loss(pred_outliers, data)

        # Pairwise loss
        if self.gamma:
            pairwiseLoss = self.pairwise_loss(pred_cam, data)

        # print(self.alpha * ESFMLoss, self.beta * classificationLoss, self.alpha, self.beta)
        loss = self.alpha * ESFMLoss + self.beta * classificationLoss + self.gamma * pairwiseLoss

        return loss
    
class PairwiseRotationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_R_rel, gt_R_rel):
        relative_rot = torch.bmm(pred_R_rel, gt_R_rel.transpose(1, 2))
        trace = relative_rot.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    
        cos_angle = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        loss = (1 - cos_angle).mean()

        return loss

class PairwiseTranslationAngleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_t_rel, gt_t_rel):
        gt_t_rel_norm = F.normalize(gt_t_rel, p=2, dim=1)
        pred_t_rel_norm = F.normalize(pred_t_rel, p=2, dim=1)
        
        cos_sim = torch.clamp((gt_t_rel_norm * pred_t_rel_norm).sum(dim=1), -1.0, 1.0)
        loss = (1 - cos_sim).mean()

        return loss

class PairwiseTranslationMagnitudeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_t_rel, gt_t_rel):
        gt_t_mag = torch.linalg.norm(gt_t_rel, dim=1)
        pred_t_mag = torch.linalg.norm(pred_t_rel, dim=1)
        
        translation_magnitude_error = torch.abs(pred_t_mag - gt_t_mag) / (gt_t_mag + 1e-8)

        return translation_magnitude_error.mean()

class PairwiseConsistencyLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.rotation_weight = conf.get_float("loss.pairwise_rotation_weight")
        self.translation_angle_weight = conf.get_float("loss.pairwise_translation_angle_weight")
        self.translation_magnitude_weight = conf.get_float("loss.pairwise_translation_magnitude_weight")

        self.rotation_loss = PairwiseRotationLoss()
        self.translation_angle_loss = PairwiseTranslationAngleLoss()
        self.translation_magnitude_loss = PairwiseTranslationMagnitudeLoss()

    def forward(self, pred_cam, data, epoch=None):
        pred_Rs, pred_ts = geo_utils.decompose_camera_matrix(pred_cam["Ps_norm"].detach())
        gt_Rs, gt_ts = geo_utils.decompose_camera_matrix(data.y, data.Ns)
        
        gt_R_rel, gt_t_rel = geo_utils.calculate_pairwise_camera_poses(gt_Rs, gt_ts)
        pred_R_rel, pred_t_rel = geo_utils.calculate_pairwise_camera_poses(pred_Rs, pred_ts)

        rotation_loss = self.rotation_loss(pred_R_rel, gt_R_rel)
        translation_angle_loss = self.translation_angle_loss(pred_t_rel, gt_t_rel)
        translation_magnitude_loss = self.translation_magnitude_loss(pred_t_rel, gt_t_rel)

        total_loss = self.rotation_weight * rotation_loss + \
                     self.translation_angle_weight * translation_angle_loss + \
                     self.translation_magnitude_weight * translation_magnitude_loss
                   
        return total_loss


class CombinedLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.outliers_loss = OutliersLoss(conf)
        self.weighted_ESFM_loss = ESFMLoss_weighted(conf)
        self.pairwise_consistency_loss = PairwiseConsistencyLoss(conf)
        self.alpha = conf.get_float('loss.reproj_loss_weight')
        self.beta = conf.get_float('loss.classification_loss_weight')
        self.gamma = conf.get_float('loss.pairwise_consistency_loss_weight')

    def forward(self, pred_cam, pred_outliers, pred_weights_M, data, epoch=None):
        zero = torch.tensor([0], device=pred_outliers.device)

        ESFMLoss = self.weighted_ESFM_loss(pred_cam, pred_outliers, data) if self.alpha else zero
        classificationLoss = self.outliers_loss(pred_outliers, data) if self.beta else zero
        pairwise_consistency_loss = self.pairwise_consistency_loss(pred_cam, data, epoch) if self.gamma else zero

        loss = self.alpha * ESFMLoss + self.beta * classificationLoss + self.gamma * pairwise_consistency_loss

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

class DirectDepthLoss(nn.Module):
    """
    """
    def __init__(self, conf):
        super().__init__()
        assert conf.get_bool('model.depth_head.enabled')
        self.cost_fcn = conf.get_string('loss.cost_fcn')
        assert self.cost_fcn in ['L1', 'L2']
        if not conf.get_bool('dataset.calibrated'):
            # NOTE: Even in the uncalibrated case, the depth should be possible to infer by normalizing the camera matrix such that the third row has unit length...
            raise NotImplementedError

    def forward(self, pred_dict, data, epoch=None):
        depths_pred = geo_utils.extract_specified_depths(
            depths_sparsemat = pred_dict['depths'],
        )
        depths_gt = geo_utils.extract_specified_depths(
            depths_dense = data.depths,
            indices = pred_dict['depths'].indices, # Use the same indices as above, to make sure the predicted & GT depth values are in a aingle consistent order.
        )

        # Determine depth scale
        s_pred = geo_utils.determine_depth_scale(depth_values=depths_pred)
        s_gt = geo_utils.determine_depth_scale(depth_values=depths_gt)
        # NOTE: Currently, the total depth scale is determined by considering the depths of all projections as a collection of independent samples.
        # This is a quite simple approach. In case we e.g. want to average the depths first per each view and then across all views, we would need to pass the original sparse matrix of depths to the above function instead of the extracted specified elements.
        # s_pred = geo_utils.determine_depth_scale(depths_sparsemat=pred_dict['depths'])
        # s_gt = geo_utils.determine_depth_scale(depths_dense=data.depths, indices=pred_dict['depths'].indices)

        # Normalize depths
        depths_pred = depths_pred / s_pred
        depths_gt = depths_gt / s_gt

        # TODO: Add a (small) depth scale regularization, e.g. reg_weight * (s_pred - 1.0)**2

        if self.cost_fcn == 'L1':
            loss = torch.mean(torch.abs(depths_pred - depths_gt))
        elif self.cost_fcn == 'L2':
            loss = torch.mean((depths_pred - depths_gt)**2)
        else:
            assert False

        return loss

class ExpDepthRegularizedOSELoss(nn.Module):
    """
    Implements the combination of Object Space Error (OSE) and an exponential depth regularization term, for pushing scene points in front of the camera.
    The result is a smooth loss function without a barrier at the principal plane.
    For each projected point, the OSE can be seen as a reprojection error computed at a z-shifted image plane, such that its depth equals the predicted depth.
    """
    def __init__(self, conf):
        super().__init__()
        assert conf.get_bool('model.view_head.enabled', default=False)
        assert conf.get_bool('model.scenepoint_head.enabled', default=False)
        self.depth_regul_weight = conf.get_float("loss.depth_regul_weight")

    def forward(self, pred_dict, data, epoch=None):
        Ps = pred_dict["Ps_norm"]
        pts_2d = Ps @ pred_dict["pts3D"]  # [m, 3, n]

        # Calculate exponential depth regularizaiton term
        depth_reg_term = self.depth_regul_weight * torch.exp(-pts_2d[:, 2, :])

        # Calculate OSE
        pts_2d_gt = data.norm_M.reshape(Ps.shape[0], 2, -1) # (m, 2, n)
        ose_err = (pts_2d[:, :2, :] - pts_2d[:, [2], :]*pts_2d_gt).norm(dim=1)

        assert data.valid_pts.is_cuda # If not, we would have to modify the masking below, to avoid an implicit call to pytorch's buggy CPU-implementation of nonzero.
        return (ose_err + depth_reg_term)[data.valid_pts].mean()





