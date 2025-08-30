import torch
from utils import geo_utils, dataset_utils, sparse_utils
from datasets import  Euclidean
import os.path
from pyhocon import ConfigFactory
import numpy as np
import warnings


class SceneData:
    def __init__(self, M, Ns, Ps_gt, scan_name, dilute_M=False, outliers=None, dict_info=None, nameslist=None, M_original=None, pairwise_epipoles=None):

        if M_original is None:
            M_original = M.detach().clone()

        # Dilute M
        if dilute_M:
            M = geo_utils.dilutePoint(M)


        n_images = Ps_gt.shape[0]

        # Set attribute
        self.scan_name = scan_name
        self.y = Ps_gt
        self.M = M
        self.M_original = M_original
        self.Ns = Ns
        self.outlier_indices = outliers

        # Calculate and store pairwise epipoles
        self.pairwise_epipoles = geo_utils.compute_pairwise_epipoles(M, Ns) if pairwise_epipoles is None else pairwise_epipoles

        # M to sparse matrix
        self.x = dataset_utils.M2sparse(M, normalize=True, Ns=Ns, M_original=M_original)

        # Get image list
        if nameslist is None:
            self.img_list = torch.arange(n_images)
        else:
            self.img_list = nameslist



        # Prepare Ns inverse transpose
        self.Ns_invT = torch.transpose(torch.inverse(Ns), 1, 2)

        # Get valid points
        self.valid_pts = dataset_utils.get_M_valid_points(M)

        # Normalize M
        self.norm_M = geo_utils.normalize_M(M, Ns, self.valid_pts).transpose(1, 2).reshape(n_images * 2, -1)


        # Stats of the scene
        self.dict_info = dict_info


    def to(self, *args, **kwargs):
        for key in self.__dict__:
            if not key.startswith('__'):
                attr = getattr(self, key)
                if isinstance(attr, sparse_utils.SparseMat) or torch.is_tensor(attr):
                    setattr(self, key, attr.to(*args, **kwargs))

        return self


def create_scene_data(conf, phase=None):
    # Init
    scan = conf.get_string('dataset.scan')
    calibrated = conf.get_bool('dataset.calibrated')
    dilute_M = conf.get_bool('dataset.diluteM', default=False)


    # Get raw data
    if calibrated:
        M, Ns, Ps_gt, outliers, dict_info, namesList, M_original = Euclidean.get_raw_data(conf, scan, phase)
    else:
        raise ValueError("The code doesn't support the uncalibrated case")

    return SceneData(M, Ns, Ps_gt, scan, dilute_M, outliers=outliers, dict_info=dict_info, nameslist=namesList, M_original=M_original)


def sample_data(data, num_samples, adjacent=True):
    """For a given scene, randomly sample num_samples cameras (rows), adjacent or not.
    Note: when the requested num_samples is more than available cameras, all cameras will be returned"""

    # Get indices
    indices = dataset_utils.sample_indices(len(data.y), num_samples, adjacent=adjacent)
    M_indices = np.sort(np.concatenate((2 * indices, 2 * indices + 1)))

    indices = torch.from_numpy(indices).squeeze()
    M_indices = torch.from_numpy(M_indices).squeeze()

    # Get sampled data
    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    outlier_indices = data.outlier_indices[indices]
    outlier_indices = outlier_indices[:, (M > 0).sum(dim=0) > 2]

    M = M[:, (M > 0).sum(dim=0) > 2]

    pairwise_epipoles = data.pairwise_epipoles[indices][:, indices]

    sampled_data = SceneData(M, Ns, y, data.scan_name,outliers=outlier_indices, nameslist=data.img_list[indices], pairwise_epipoles=pairwise_epipoles)
    if (sampled_data.x.pts_per_cam == 0).any():
        warnings.warn('Cameras with no points for dataset '+ data.scan_name)

    return sampled_data


def create_scene_data_from_list(scan_names_list, conf):
    data_list = []
    for scan_name in scan_names_list:
        conf["dataset"]["scan"] = scan_name
        data = create_scene_data(conf)
        data_list.append(data)

    return data_list


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
    y, Ns = data.y[indices], data.Ns[indices]
    M = data.M[M_indices]
    M = M[:, (M > 0).sum(dim=0) > 2]
    return SceneData(M, Ns, y, data.scan_name + "_{}".format(subset_size), outliers=data.outlier_indices[indices], dict_info=data.dict_info, nameslist=data.img_list[indices])

if __name__ == "__main__":
    test_dataset()

