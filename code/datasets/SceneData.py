from pytorch3d.transforms import axis_angle_to_matrix
import copy
import math
import torch
from utils import geo_utils, dataset_utils, sparse_utils
from utils.general_utils import nonzero_safe
from utils.constants import *
from datasets import Projective, Euclidean
import os.path
from pyhocon import ConfigFactory
import numpy as np
import warnings


class SceneData:
    def __init__(self,M, Ns,Ps_gt, scene_name, calibrated = False, store_depth_targets = False, depths = None, dilute_M=False, outliers=None, dict_info=None, nameslist=None, M_original=None
    ):
        """
        If calibrated = True, then N = inv(K), and the camera matrices, as they are normalized with N, are calibrated.
        """
        if M_original is None:
            M_original = M.detach().clone()

        # Dilute M
        if dilute_M:
            M = geo_utils.dilutePoint(M)
        n_images = Ps_gt.shape[0]

        # Determine the device
        self.device = M.device

        # Set attribute
        self.scene_name = scene_name
        self.calibrated = calibrated
        self.store_depth_targets = store_depth_targets
        self.y = Ps_gt
        self._M = M
        self.M_original = M_original
        self.Ns = Ns
        self.outlier_indices = outliers
        self.dict_info = dict_info

        # M to sparse matrix
        self.x = dataset_utils.M2sparse(self.M, normalize=True, Ns=self.Ns, M_original=M_original)

        # Get image list
        if nameslist is None:
            self.img_list = torch.arange(n_images)
        else:
            self.img_list = nameslist

        # Prepare Ns inverse
        self.Ns_invT = torch.transpose(torch.inverse(self.Ns), 1, 2)

        # Get valid points
        self.valid_pts = dataset_utils.get_M_valid_points(self.M)

        # Normalize M
        self._norm_M = geo_utils.normalize_M(self.M, self.Ns, self.valid_pts).transpose(1, 2).reshape(n_images * 2, -1)

        # Triangulate scenepoints and store a target depths for prediction (at a normalized scale)
        if self.store_depth_targets:
            print("Storing depth targets...")
            valid_pts_idx = nonzero_safe(self.valid_pts)
            if depths is not None:
                # Precomputed depths already provided
                self.depths = depths
            else:
                # TODO: Ideally we should store a sparse representation, but such a change is mostly relevant if considering the other dense matrices as well.
                # For the moment we store the depth for every view / scenepoint combination, no matter whether it is visible or not.
                if not calibrated:
                    # NOTE: Even in the uncalibrated case, the depth should be possible to infer by normalizing the camera matrix such that the third row has unit length...
                    raise NotImplementedError
                K_inv = self.Ns
                X = torch.tensor(geo_utils.n_view_triangulation(
                    self.y.numpy(),
                    self.M.numpy(),
                    Ns = K_inv.numpy(),
                ), dtype=torch.float32)
                # X, V_H = geo_utils.n_view_triangulation(
                #     self.y.numpy(),
                #     self.M.numpy(),
                #     Ns = K_inv.numpy(),
                #     return_V_H = True,
                # )
                # X = torch.tensor(X, dtype=torch.float32)
                # S_diag, V_H = zip(*[ (torch.tensor(S_diag, dtype=torch.float32), torch.tensor(curr_V_H, dtype=torch.float32)) for S_diag, curr_V_H in V_H ])

                # # Calculate reprojection error for the triangulation
                # projected_points = (self.y @ X).swapaxes(1, 2)
                # projected_points = projected_points[:, :, :2] / projected_points[:, :, [2]]
                # reprojerr = torch.mean(torch.norm(projected_points[valid_pts_idx[0], valid_pts_idx[1]] - geo_utils.M_to_xs(self._M)[valid_pts_idx[0], valid_pts_idx[1]], dim=1))
                # print(reprojerr) # DrinkingFountainSomewhereInZurich (Euclidean): 0.3211 px
                # assert False

                # Verify certain assumptions required for the depth calculation below
                valid_scenepoint_mask = torch.any(self.valid_pts, dim=0) # Determine which points are visible in enough cameras
                valid_scenepoint_idx = nonzero_safe(valid_scenepoint_mask)[0]
                assert torch.all(torch.isfinite(X[:, valid_scenepoint_idx])) # Verify no NaN in triangulation
                assert torch.all(X[3, valid_scenepoint_idx] == 1) # Verify normalized (pflat)
                assert torch.all(K_inv[:, 2, :] == torch.tensor([0, 0, 1])[None, None, :]) # Verify final row [0, 0, 1] of K_inv
                assert torch.allclose(torch.norm((K_inv @ self.y)[:, :, :3], dim=2), torch.ones((1, 1))) # Verify unit rows of R, as extracted from K_inv*(K*[R  t]).

                self.depths = (K_inv @ self.y @ X)[:, 2, :]
                # NOTE: The scene should be normalized such that the depths have a nominal magnitude.
                #       While we could perform it here, it would require maintaining that scale whenever we transform / subsample the scene data.
                #       Instead, we perform the normalization when computing the loss.
            assert self.depths.shape == (n_images, self.M.shape[1])
            assert torch.all(torch.isfinite(self.depths[valid_pts_idx[0], valid_pts_idx[1]]))
            # # NOTE: Negative depths encountered for AlcatrazCourtyard:
            # print(scene_name)
            # print(self.depths.shape)
            # print(torch.min(self.depths[valid_pts_idx[0], valid_pts_idx[1]]))
            # print(torch.max(self.depths[valid_pts_idx[0], valid_pts_idx[1]]))
            # print(torch.mean(self.depths[valid_pts_idx[0], valid_pts_idx[1]]))
            # print(torch.sum(self.depths[valid_pts_idx[0], valid_pts_idx[1]] <= 0))
            # print(torch.sum(self.depths[valid_pts_idx[0], valid_pts_idx[1]] < 0))
            # neg_depth_mask = (self.depths <= 0) & self.valid_pts
            # neg_depth_idx = nonzero_safe(neg_depth_mask) # 2-tuple, elements with shape (n_neg,)
            # print(neg_depth_idx)
            # print(geo_utils.M_to_xs(self.M)[neg_depth_idx[0], neg_depth_idx[1]])
            # resid = ((self.y @ X)[:, :2, :] / (self.y @ X)[:, [2], :]).swapaxes(1, 2)[neg_depth_idx[0], neg_depth_idx[1]] - geo_utils.M_to_xs(self.M)[neg_depth_idx[0], neg_depth_idx[1]]
            # print(resid)
            # print(torch.norm(resid, dim=1, keepdim=True))
            # neg_depth_scenepoint_idx = torch.unique(neg_depth_idx[1])
            # print(neg_depth_scenepoint_idx)
            # print(torch.sum(self.valid_pts[:, neg_depth_scenepoint_idx], dim=0))
            # print(self.depths[neg_depth_idx[0], neg_depth_idx[1]])
            # # print(self.depths[neg_depth_idx[0], neg_depth_idx[1]][:, None] * (K_inv @ torch.cat([geo_utils.M_to_xs(self.M), torch.ones((self.M.shape[0]//2, 1, 1))]).swapaxes(1, 2)).swapaxes(1, 2)[neg_depth_idx[0], neg_depth_idx[1]])
            # # print((K_inv @ self.y @ X).swapaxes(1, 2)[neg_depth_idx[0], neg_depth_idx[1]])
            # print(V_H[22539][-1, :])
            # print(S_diag[22539])
            # print(S_diag[1000])
            # # print(V_H)
            # import pdb
            # pdb.set_trace()
            assert torch.all(self.depths[valid_pts_idx[0], valid_pts_idx[1]] > 0)
        else:
            self.depths = None

        # Define the graph connectivity for the aggregations from projection features to view features and scenepoint features, respectively.
        self.graph_wrappers = self.create_axial_aggregation_graphs(
            self.x,
        )

    @property
    def M(self):
        if not self._M.device == self.device:
            self._M = self._M.to(self.device)
        return self._M

    @property
    def norm_M(self):
        if not self._norm_M.device == self.device:
            self._norm_M = self._norm_M.to(self.device)
        return self._norm_M

    def create_axial_aggregation_graphs(
        self,
        x, # (m, n, 2) sparse measurement matrix of projections
    ):
        x = x.to_torch_hybrid_sparse_coo()

        m, n = x.shape[0], x.shape[1]
        valid_indices = x.indices()
        device = x.device

        # Define graph for row-wise aggregation.
        # Nodes consist of all valid matrix elements, followed by one node per row.
        graph_wrapper_proj2view = dataset_utils.AxialAggregationGraphWrapper(m, n, 1, valid_indices=valid_indices)

        # Define graph for column-wise aggregation.
        # Nodes consist of all valid matrix elements, followed by one node per column.
        graph_wrapper_proj2scenepoint = dataset_utils.AxialAggregationGraphWrapper(m, n, 0, valid_indices=valid_indices)

        # Define (dense) graph for global (column-wise) aggregation of row nodes (=view nodes).
        # NOTE: If there are views with fewer than 10 points, the scene will be discarded on the basis of dataset_utils.is_valid_sample().
        # Let's be consistent with this number, although since the scene would then be discarded, it is probably not needed to leave out some views from the view2global aggregation.
        valid_view_mask = sparse_utils.get_n_nonempty(x, dim=1, keepdim=True).to_dense() >= MIN_N_POINTS_PER_VIEW
        if valid_view_mask.is_cuda:
            valid_view_indices = torch.nonzero(valid_view_mask).T
        else:
            valid_view_indices = torch.from_numpy(np.array(np.nonzero(valid_view_mask.numpy())))
        graph_wrapper_view2global = dataset_utils.AxialAggregationGraphWrapper(m, 1, 0, valid_indices=valid_view_indices, device=device)

        # Define (dense) graph for global (row-wise) aggregation of column nodes (=scenepoint nodes).
        valid_scenepoint_mask = sparse_utils.get_n_nonempty(x, dim=0, keepdim=True).to_dense() >= MIN_N_VIEWS_PER_POINT # Need visibility in at least 2 views. If < 2, it is probably == 0 due to previous filtering.
        if valid_scenepoint_mask.is_cuda:
            valid_scenepoint_indices = torch.nonzero(valid_scenepoint_mask).T
        else:
            valid_scenepoint_indices = torch.from_numpy(np.array(np.nonzero(valid_scenepoint_mask.numpy())))
        graph_wrapper_scenepoint2global = dataset_utils.AxialAggregationGraphWrapper(1, n, 1, valid_indices=valid_scenepoint_indices, device=device)

        # ######################################################################################
        # # VERIFICATION OF RESULTING GRAPH CONNECTIVITY #######################################
        # ######################################################################################

        # M = self.norm_M.reshape(m, 2, n).permute(0, 2, 1) # (2*m, n) -> (m, 2, n) -> (m, n, 2)

        # n_valid = valid_indices.shape[1]

        # # x_proj2view = graph_wrapper_proj2view.graph.x
        # x_proj2view = graph_wrapper_proj2view.generate_node_features(x)
        # # x_proj2scenepoint = graph_wrapper_proj2scenepoint.graph.x
        # x_proj2scenepoint = graph_wrapper_proj2scenepoint.generate_node_features(x)

        # for view_idx in range(m):
        #     curr_row_mask = valid_indices[0, :] == view_idx
        #     assert curr_row_mask.shape == (n_valid,)

        #     curr_valid_alt1 = M[valid_indices[0, curr_row_mask], valid_indices[1, curr_row_mask], :]
        #     print(curr_valid_alt1.shape, x_proj2view[:n_valid, :][curr_row_mask, :].shape)
        #     assert torch.all(curr_valid_alt1 == x_proj2view[:n_valid, :][curr_row_mask, :])

        #     # Alternative:
        #     curr_valid_alt2 = M[view_idx, valid_indices[1, curr_row_mask], :]
        #     # print(curr_valid_alt2.shape, x_proj2view[:n_valid, :][curr_row_mask, :].shape)
        #     assert torch.all(curr_valid_alt2 == x_proj2view[:n_valid, :][curr_row_mask, :])

        # for scenepoint_idx in range(n):
        #     curr_col_mask = valid_indices[1, :] == scenepoint_idx
        #     assert curr_col_mask.shape == (n_valid,)

        #     curr_valid_alt1 = M[valid_indices[0, curr_col_mask], valid_indices[1, curr_col_mask], :]
        #     # print(curr_valid_alt1.shape, x_proj2scenepoint[:n_valid, :][curr_col_mask, :].shape)
        #     assert torch.all(curr_valid_alt1 == x_proj2scenepoint[:n_valid, :][curr_col_mask, :])

        #     # Alternative:
        #     curr_valid_alt2 = M[valid_indices[0, curr_col_mask], scenepoint_idx, :]
        #     # print(curr_valid_alt2.shape, x_proj2scenepoint[:n_valid, :][curr_col_mask, :].shape)
        #     assert torch.all(curr_valid_alt2 == x_proj2scenepoint[:n_valid, :][curr_col_mask, :])

        # assert False
        # ######################################################################################
        # ######################################################################################

        graph_wrappers = {
            'proj2view': graph_wrapper_proj2view,
            'proj2scenepoint': graph_wrapper_proj2scenepoint,
            'view2global': graph_wrapper_view2global,
            'scenepoint2global': graph_wrapper_scenepoint2global,
        }

        return graph_wrappers

    def to(self, device, *args, dense_on_demand=False, **kwargs):
        def recognized_transferable(x):
            return any([
                isinstance(x, sparse_utils.SparseMat),
                isinstance(x, dataset_utils.AxialAggregationGraphWrapper),
                torch.is_tensor(x),
            ])
        # Start with making a shallow copy of self:
        ret = copy.copy(self)
        # Next, replace all attributes with the corresponding data moved to the device:
        for key in ret.__dict__:
            if key.startswith('__'):
                continue
            if dense_on_demand and key in ['_M', '_norm_M']:
                continue
            attr = getattr(ret, key)
            if recognized_transferable(attr):
                setattr(ret, key, attr.to(device, *args, **kwargs))
            elif isinstance(attr, dict):
                # Only call .to on values that support it
                new_dict = {}
                for dict_key, dict_val in attr.items():
                    if recognized_transferable(dict_val):
                        new_dict[dict_key] = dict_val.to(device, *args, **kwargs)
                    else:
                        new_dict[dict_key] = dict_val
                setattr(ret, key, new_dict)

        ret.device = device

        return ret

    # def to(self, *args, **kwargs):
    #     for key in self.__dict__:
    #         if not key.startswith('__'):
    #             attr = getattr(self, key)
    #             if isinstance(attr, sparse_utils.SparseMat) or torch.is_tensor(attr):
    #                 setattr(self, key, attr.to(*args, **kwargs))

    #     return self


def create_scene_data(
    conf,
    scene = None,
    calibrated = None,
    use_gt = None,
    phase=None
):
    store_depth_targets = conf.get_bool('model.depth_head.enabled', default=False)

    # Optionally override some configuration options:
    scene = scene if scene is not None else conf.get_string('dataset.scene')
    calibrated = calibrated if calibrated is not None else conf.get_bool('dataset.calibrated')
    use_gt = use_gt if use_gt is not None else conf.get_bool('dataset.use_gt')
    dilute_M = conf.get_bool('dataset.diluteM', default=False)
    # Get raw data
    if calibrated:
        M, Ns, Ps_gt, outliers, dict_info, namesList, M_original = Euclidean.get_raw_data(conf,scene, phase)
    else:
        M, Ns, Ps_gt = Projective.get_raw_data(conf, scene, use_gt)

    # if scene in [
    #     'PantheonParis', # NOTE: Some points are visible in 0 views and will be pruned. All other points are visible in 2+ views.
    # ]:
    #     # Point filtering, discarding points that are not visible in at least MIN_N_VIEWS_PER_POINT views.
    #     M_valid_pts_mask = dataset_utils.get_M_valid_points(M)
    #     points_mask = M_valid_pts_mask.any(dim=0) # M_valid_pts_mask is False for the entire column of such points, so we just need to check for which columns of the mask there are True entries.
    #     M = M[:, nonzero_safe(points_mask)[0]]

    scene_data = SceneData(
        M,
        Ns,
        Ps_gt,
        scene,
        calibrated = calibrated,
        store_depth_targets = store_depth_targets,
        dilute_M = dilute_M, 
        outliers=outliers, 
        dict_info=dict_info, 
        nameslist=namesList,
        M_original=M_original
    )
    # assert dataset_utils.is_valid_sample(scene_data)
    return scene_data


def sample_data(data, num_samples, adjacent=True):
    """For a given scene, randomly sample num_samples cameras (rows), adjacent or not.
    Note: when the requested num_samples is more than available cameras, all cameras will be returned"""

    # Get indices
    indices = dataset_utils.sample_indices(len(data.y), num_samples, adjacent=adjacent)
    M_indices = np.sort(np.concatenate((2 * indices, 2 * indices + 1)))

    indices = torch.from_numpy(indices).squeeze()
    M_indices = torch.from_numpy(M_indices).squeeze()

    depths = data.depths
    if data.store_depth_targets:
        assert depths is not None

    # Get sampled data
    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    outlier_indices = data.outlier_indices[indices]
    outlier_indices = outlier_indices[:, (M > 0).sum(dim=0) > 2]
    M = M[:, (M > 0).sum(dim=0) > 2]
    if data.store_depth_targets:
        depths = depths[indices, :]

    # Additional point filtering, discarding points that are not visible in at least MIN_N_VIEWS_PER_POINT views:
    # M_valid_pts_mask = dataset_utils.get_M_valid_points(M)
    # points_mask = M_valid_pts_mask.any(dim=0) # M_valid_pts_mask is False for the entire column of such points, so we just need to check for which columns of the mask there are True entries.
    # if M.is_cuda:
    #     M = M[:, points_mask]
    #     if data.store_depth_targets:
    #         depths = depths[:, points_mask]
    # else:
    #     # NOTE: Workaround for bug in nonzero_out_cpu(), internally called by pytorch during advanced indexing operation.
    #     idx = np.nonzero(points_mask.numpy())[0]
    #     assert len(idx.shape) == 1, 'Expected 1D-array, but encountered idx.shape == {}'.format(idx.shape)
    #     M = M[:, torch.from_numpy(idx)]
    #     if data.store_depth_targets:
    #         depths = depths[:, torch.from_numpy(idx)]

    sampled_data = SceneData(
        M,
        Ns,
        y,
        data.scene_name,
        calibrated = data.calibrated,
        store_depth_targets = data.store_depth_targets,
        depths = depths,
        outliers=outlier_indices, 
        nameslist=data.img_list[indices]
    )
    if (sampled_data.x.pts_per_cam == 0).any():
        warnings.warn('Cameras with no points for dataset '+ data.scene_name)

    return sampled_data

def create_scene_data_from_list(scene_names_list, conf):
    data_list = []
    for scene_name in scene_names_list:
        conf["dataset"]["scene"] = scene_name
        data = create_scene_data(conf, scene=scene_name)
        data_list.append(data)

    return data_list


def test_dataset():
    # Prepare configuration
    dataset_dict = {"images_path": "/home/labs/waic/hodaya/PycharmProjects/GNN-for-SFM/datasets/images/",
                    "normalize_pts": True,
                    "normalize_f": True,
                    "use_gt": False,
                    "calibrated": False,
                    "scene": "Alcatraz Courtyard",
                    "edge_min_inliers": 30,
                    "use_all_edges": True,
                    }

    train_dict = {"infinity_pts_margin": 1e-4,
                  "hinge_loss_weight": 1,
                  }
    loss_dict = {"infinity_pts_margin": 1e-4,
    "pts_grad_equalization_pre_perspective_divide": False,
    "hinge_loss": True,
    "hinge_loss_weight" : 1
    }
    conf_dict = {"dataset": dataset_dict, "loss":loss_dict}

    print("Test projective")
    conf = ConfigFactory.from_dict(conf_dict)
    data = create_scene_data(conf)
    test_data(data, conf)

    print('Test move to device')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    new_data = data.to(device)

    print(os.linesep)
    print("Test Euclidean")
    conf = ConfigFactory.from_dict(conf_dict) # This conf reset may no longer be necessary, but let's keep it in case we have overlooked some modification of conf inside any of the called functions.
    data = create_scene_data(conf, calibrated=True)
    test_data(data, conf)

    print(os.linesep)
    print("Test use_gt GT")
    conf = ConfigFactory.from_dict(conf_dict) # This conf reset may no longer be necessary, but let's keep it in case we have overlooked some modification of conf inside any of the called functions.
    data = create_scene_data(conf, use_gt=True)
    test_data(data, conf)


def test_data(data, conf):
    import loss_functions

    # Test Losses of GT and random on data
    repLoss = loss_functions.ESFMLoss(conf)
    cams_gt = prepare_cameras_for_loss_func(data.y, data)
    cams_rand = prepare_cameras_for_loss_func(torch.rand(data.y.shape), data)

    print("Loss for GT: Reprojection = {}".format(repLoss(cams_gt, data)))
    print("Loss for rand: Reprojection = {}".format(repLoss(cams_rand, data)))


def prepare_cameras_for_loss_func(Ps, data):
    Vs_invT = Ps[:, 0:3, 0:3]
    Vs = torch.inverse(Vs_invT).transpose(1, 2)
    ts = torch.bmm(-Vs.transpose(1, 2), Ps[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()
    pts_3D = torch.from_numpy(geo_utils.n_view_triangulation(Ps.numpy(), data.M.numpy(), data.Ns.numpy())).float()
    return {"Ps": torch.bmm(data.Ns, Ps), "pts3D": pts_3D}


def get_subset(data, subset_size):
    # Get subset indices
    valid_pts = dataset_utils.get_M_valid_points(data.M)
    n_cams = valid_pts.shape[0]

    first_idx = valid_pts.sum(dim=1).argmax().item()
    curr_pts = valid_pts[first_idx].clone()
    valid_pts[first_idx] = False
    indices = [first_idx]

    for i in range(subset_size - 1):
        shared_pts = curr_pts.expand(n_cams, -1) & valid_pts
        next_idx = shared_pts.sum(dim=1).argmax().item()
        curr_pts = curr_pts | valid_pts[next_idx]
        valid_pts[next_idx] = False
        indices.append(next_idx)

    print("Cameras are:")
    print(indices)

    indices = torch.sort(torch.tensor(indices))[0]
    M_indices = torch.sort(torch.cat((2 * indices, 2 * indices + 1)))[0]
    M = M[:, (M > 0).sum(dim=0) > 2]

    depths = data.depths
    if data.store_depth_targets:
        assert depths is not None

    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    if data.store_depth_targets:
        depths = depths[indices, :]

    # Additional point filtering, discarding points that are not visible in at least MIN_N_VIEWS_PER_POINT views:
    # M_valid_pts_mask = dataset_utils.get_M_valid_points(M)
    # points_mask = M_valid_pts_mask.any(dim=0) # M_valid_pts_mask is False for the entire column of such points, so we just need to check for which columns of the mask there are True entries.
    # if M.is_cuda:
    #     M = M[:, points_mask]
    #     if data.store_depth_targets:
    #         depths = depths[:, points_mask]
    # else:
    #     # NOTE: Workaround for bug in nonzero_out_cpu(), internally called by pytorch during advanced indexing operation.
    #     idx = np.nonzero(points_mask.numpy())[0]
    #     assert len(idx.shape) == 1, 'Expected 1D-array, but encountered idx.shape == {}'.format(idx.shape)
    #     M = M[:, torch.from_numpy(idx)]
    #     if data.store_depth_targets:
    #         depths = depths[:, torch.from_numpy(idx)]

    return SceneData(
        M,
        Ns,
        y,
        data.scene_name + "_{}".format(subset_size),
        calibrated = data.calibrated,
        store_depth_targets = data.store_depth_targets,
        depths = depths,
        outliers=data.outlier_indices[indices], 
        dict_info=data.dict_info, 
        nameslist=data.img_list[indices]
    )


if __name__ == "__main__":
    test_dataset()

