import csv
import sys
# sys.path.append('/home/projects/ronen/fadi/myProjects/outliersRemoval3D/code/datasets')
# sys.path.append('/home/projects/ronen/fadi/myProjects/outliersRemoval3D/code/utils')
# sys.path.append('/home/projects/ronen/fadi/myProjects/outliersRemoval3D/code')
# print(sys.path)
import cv2  # Do not remove
import torch
import os
import sys

from utils.dataset_utils import get_M_valid_points
from utils.Phases import Phases
from utils.path_utils import path_to_outliers
from utils import geo_utils, general_utils, dataset_utils, path_utils, plot_utils
import scipy.io as sio
import numpy as np
import os.path
import networkx as nx
from tqdm import tqdm
import copy




def get_raw_data(conf, scan, phase):
    """
    Load raw data for SfM training or evaluation.

    Returns:
        M (torch.Tensor): 2D points matrix [2m, n]
        Ns (torch.Tensor): Inverse calibration matrices [m, 3, 3]
        Ps_gt (torch.Tensor): Ground-truth projection matrices [m, 3, 4]
        outliers (torch.Tensor): Ground-truth outlier mask [m, n]
        dict_info (dict): Metadata (e.g., outliers percent)
        names_list (list): List of image names
        M_original (torch.Tensor): Original points matrix (before filtering)
    """

    # === Setup paths and parameters ===
    dataset_name = conf.get_string('dataset.dataset', default="megadepth")
    dataset_path = os.path.join(path_utils.path_to_datasets(dataset_name), f'{scan}.npz')
    output_mode = conf.get_int('train.output_mode', default=-1)
    use_gt = conf.get_bool('dataset.use_gt')
    remove_outliers_gt = conf.get_bool('dataset.remove_outliers_gt', default=False)
    remove_outliers_pred = False
    outliers_threshold = conf.get_float("test.outliers_threshold", default=0.6)
    if scan is None:
        scan = conf.get_string('dataset.scan')

    print(f"Used Dataset: {dataset_name}")
    print(f"Loading from: {dataset_path}")
    dataset = np.load(dataset_path, allow_pickle=True)

    # === Extract raw data ===
    M_np = dataset['M']
    Ps_gt_np = dataset['Ps_gt']
    Ns_np = dataset['Ns']
    names_list = dataset['namesList']
    outliers_np = dataset.get('outliers2', np.zeros((M_np.shape[0] // 2, M_np.shape[1])))

    # === Initialize info dictionary ===
    dict_info = {
        'pointsNum': M_np.shape[1],
        'camsNum': M_np.shape[0] // 2,
        'outliersPercent': float("%.4f" % dataset.get('outlier_pct', 0.0)),
        'outliers_pred': torch.zeros_like(torch.from_numpy(outliers_np).float())
    }

    # === Convert to torch tensors ===
    M = torch.from_numpy(M_np).float()
    M_original = M.clone()
    Ps_gt = torch.from_numpy(Ps_gt_np).float()
    Ns = torch.from_numpy(Ns_np).float()
    outliers = torch.from_numpy(outliers_np).float()
    outliers_mask = outliers.clone()

    # === Fine-tuning: Load predicted outliers ===
    if phase is Phases.FINE_TUNE and output_mode == 3:
        print(f"Fine-tuning phase: loading predicted outliers for scan {scan}")
        print("Loading outliers from:", path_to_outliers(conf, Phases.TEST, epoch=None, scan=scan))
        outliers_mask_np = np.load(path_to_outliers(conf, Phases.TEST, epoch=None, scan=scan) + ".npz")['outliers_pred']
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

    # === Keep only largest connected component if fine-tuning with outputMode == 3 ===
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
    return np.nanmean(global_rep_err), np.nanmax(global_rep_err)




def test_euclidean_dataset(scan):
    dataset_path_format = os.path.join(path_utils.path_to_datasets(), 'Euclidean', '{}.npz')

    # Get raw data
    dataset = np.load(dataset_path_format.format(scan))

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




