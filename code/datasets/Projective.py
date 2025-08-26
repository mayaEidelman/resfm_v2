import torch  # DO NOT REMOVE
import cv2  # DO NOT REMOVE

from utils import geo_utils, general_utils, dataset_utils, path_utils
import scipy.io as sio
import numpy as np
import os.path
from utils.Phases import Phases
from utils.path_utils import path_to_outliers

def get_raw_data(conf,scene, phase):
    """
    # :param conf:
    :return:
    M - Points Matrix (2mxn)
    Ns - Normalization matrices (mx3x3)
    Ps_gt - Olsson's estimated camera matrices (mx3x4)
    NBs - Normzlize Bifocal Tensor (Normalized Fn) (3mx3m)
    triplets
    """
    # Init
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Projective', '{}.npz')
    remove_outliers_gt = conf.get_bool('dataset.remove_outliers_gt', default=False)
    use_gt = conf.get_bool('dataset.use_gt')
    output_mode = conf.get_int('train.output_mode', default=-1)
    outliers_threshold = conf.get_float("test.outliers_threshold", default=0.6)

    # Get raw data
    dataset = np.load(dataset_path_format.format(scene))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']
    N33 = Ns[:, 2, 2][:, None, None]
    Ns /= N33 # Divide by N33 to ensure last row [0, 0, 1] (although generally the case, a small deviation in scale has been observed for e.g. the PantheonParis scene)
    outliers_np = dataset.get('outliers2', np.zeros((M.shape[0] // 2, M.shape[1])))

    # === Initialize info dictionary ===
    dict_info = {
        'pointsNum': M.shape[1],
        'camsNum': M.shape[0] // 2,
        'outliersPercent': float("%.4f" % dataset.get('outlier_pct', 0.0)),
        'outliers_pred': torch.zeros_like(torch.from_numpy(outliers_np).float())
    }

    if use_gt:
        M = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns))

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()
    outliers = torch.from_numpy(outliers_np).float()
    outliers_mask = outliers.clone()

    # === Fine-tuning: Load predicted outliers ===
    if phase is Phases.FINE_TUNE and output_mode == 3:
        print(f"Fine-tuning phase: loading predicted outliers for scan {scene}")
        print("Loading outliers from:", path_to_outliers(conf, Phases.TEST, epoch=None, scan=scene))
        outliers_mask_np = np.load(path_to_outliers(conf, Phases.TEST, epoch=None, scan=scene) + ".npz")['outliers_pred']
        outliers_mask = torch.from_numpy(outliers_mask_np > outliers_threshold)
        remove_outliers_pred = True

    # === Remove outliers ===
    if remove_outliers_gt or remove_outliers_pred:
        outliers_mask = outliers_mask > 0  # ensure boolean mask

        # Convert shape [2m, n] → [n, m, 2] → [m, n, 2]
        M = M.transpose(0, 1).reshape(-1, M.shape[0] // 2, 2).transpose(0, 1)
        M[outliers_mask] = 0
        # Back to [2m, n]
        M = M.transpose(0, 1).reshape(-1, M.shape[0] * 2).transpose(0, 1)

    if phase is Phases.FINE_TUNE and output_mode == 3:
        _, valid_cam_indices = dataset_utils.check_if_M_connected(M, thr=1, return_largest_component=True)
        double_cam_indices = [j for i in [[idx * 2, idx * 2 + 1] for idx in valid_cam_indices] for j in i]

        Ns = Ns[valid_cam_indices]
        Ps_gt = Ps_gt[valid_cam_indices]
        outliers = outliers[valid_cam_indices]
        names_list = names_list[valid_cam_indices]
        M = M[double_cam_indices]
        M_original = M_original[double_cam_indices]


    return M, Ns, Ps_gt, outliers, dict_info, names_list, M_original


def test_Ps_M(Ps, M, Ns):
    global_rep_err = geo_utils.calc_global_reprojection_error(Ps.numpy(), M.numpy(), Ns.numpy())
    print("Reprojection Error: Mean = {}, Max = {}".format(np.nanmean(global_rep_err), np.nanmax(global_rep_err)))


def test_projective_dataset(scene):
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Projective', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scene))

    # Get bifocal tensors and 2D points
    M = dataset['M']
    Ps_gt = dataset['Ps_gt']
    Ns = dataset['Ns']

    M_gt = torch.from_numpy(dataset_utils.correct_matches_global(M, Ps_gt, Ns)).float()

    M = torch.from_numpy(M).float()
    Ps_gt = torch.from_numpy(Ps_gt).float()
    Ns = torch.from_numpy(Ns).float()

    print("Test Ps and M")
    test_Ps_M(Ps_gt, M, Ns)

    print("Test Ps and M_gt")
    test_Ps_M(Ps_gt, M_gt, Ns)


if __name__ == "__main__":
    scene = "Alcatraz Courtyard"
    test_projective_dataset(scene)


