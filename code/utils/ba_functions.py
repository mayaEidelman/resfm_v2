import numpy as np
import torch
import time

from utils.ba_advanced import batch_matrix_to_pycolmap, prepare_ba_options, pycolmap_to_batch_matrix, create_ba_results, \
    process_camera_indices_and_bundle_adjustment
from utils import  dataset_utils, ba_advanced, path_utils
from utils import geo_utils
from utils.plot_utils import plot_cameras
import pycolmap





def get_shared_cameras(projection_array, first_list, second_list):
    """
    Return a sorted list of cameras from the second list that share the most 3D points with the first list.

    :param projection_array: np.array (#cams x #3dpoints) - True if the 3D point is projected by the camera
    :param first_list: List[int] - Indices of the cameras in the first list
    :param second_list: List[int] - Indices of the cameras in the second list
    :return: List[int] - Sorted list of camera indices from the second list based on shared 3D points
    """
    # Find all 3D points visible by the cameras in the first list
    points_visible_by_first_list = np.any(projection_array[first_list], axis=0)

    # Count shared 3D points for each camera in the second list
    shared_counts = []
    for camera in second_list:
        # Count the number of shared points
        shared_count = np.sum(projection_array[camera] & points_visible_by_first_list)
        shared_counts.append((camera, shared_count))

    # Sort the cameras in the second list by the number of shared 3D points (descending order)
    sorted_cameras = [camera for camera, _ in sorted(shared_counts, key=lambda x: x[1], reverse=True)]

    return sorted_cameras


def euc_ba(conf, xs, Rs, ts, Ks, Xs_our=None, M=None, Ps=None, Ns=None, repeat=True, triangulation=False, filtering_thr=0.0, M_original=None, images_path=None, img_list=None, curr_epoch=None):
    """
    Computes bundle adjustment with ceres solver
    :param xs: 2d points [m,n,2]
    :param Rs: rotations [m,3,3]
    :param ts: translations [m,3]
    :param Ks: inner parameters, calibration matrices [m,3,3]
    :param Xs_our: initial 3d points [n,3] or None if triangulation needed
    :param Ps: cameras [m,3,4]. Ps[i] = Ks[i] @ Rs[i].T @ [I, -ts[i]]
    :param Ns: normalization matrices. If Ks are known, Ns = inv(Ks)
    :param repeat: run ba twice. default: True
    :param triangulation: For initial point run triangulation. default: False
    :return: results. The new camera parameters, 3d points.
    """

    reconstructions_list = []
    Ks_orig = Ks.copy()
    visible_points = xs[:, :, 0] > 0
    point_indices = np.stack(np.where(visible_points.sum(axis=0) >= 2))[0]
    xs = xs[:, point_indices]
    visible_points = visible_points[:, point_indices]
    M = geo_utils.xs_to_M(xs)

    if Ns is None:
        Ns = np.linalg.inv(Ks)

    if Ps is None:
        Ps = geo_utils.batch_get_camera_matrix_from_rtk(Rs, ts, Ks)

    if triangulation:
        norm_P, norm_x = geo_utils.normalize_points_cams(Ps, xs, Ns)
        Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)
    else:
        Xs = Xs_our

    Xs = Xs[point_indices]


    reconstruction = batch_matrix_to_pycolmap(xs, Rs, ts, Ks, Xs, img_list=img_list)

    ba_options = prepare_ba_options()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    # reconstruction.extract_colors_for_all_images(images_path)
    # reconstruction.write_text(path_utils.path_to_reconstructions(conf, conf.phase, name='00'))
    xs_original = geo_utils.M_to_xs(M_original)
    new_Rs, new_ts, new_Ps, Ks, new_Xs = pycolmap_to_batch_matrix(reconstruction)
    reconstructions_list.append(create_ba_results(M, new_Ps, new_Rs, new_ts, new_Xs, xs, np.arange(len(Rs))))
    if repeat:


        norm_P, norm_x = geo_utils.normalize_points_cams(new_Ps, xs, Ns)
        new_Xs = geo_utils.dlt_triangulation(norm_P, norm_x, visible_points)

        if filtering_thr:
            # Compute global reprojection error per 3D point
            pointwise_reproj_errors = geo_utils.calc_global_reprojection_error_withPoints(new_Ps, M, Ns, new_Xs.T)
            mean_reproj_error_per_point = np.nanmean(pointwise_reproj_errors, axis=0)  # [n]
            # Keep 3D points with finite error below threshold
            valid_point_mask = np.logical_and(~np.isnan(mean_reproj_error_per_point), mean_reproj_error_per_point <= filtering_thr)
            valid_point_indices = np.nonzero(valid_point_mask)[0]

            new_Xs = new_Xs[valid_point_indices]
            newM = M[:, valid_point_indices]

            newM, _, _, _, new_Xs, valid_cam_mask, valid_point_mask_2 = geo_utils.remove_empty_tracks_cams(newM, Ps=new_Ps, Ns=Ns, Xs=new_Xs, pts_per_cam_thresh=10, cam_per_pts_thresh=3)
            # Extract connected camera components
            return_largest = True
            if return_largest:
                _, connected_cam_indices = dataset_utils.check_if_M_connected(
                    torch.tensor(newM), thr=15,
                    return_largest_component=True
                )
            else:
                connected_cam_indices = list(range(len(Ks)))

            connected_components = dataset_utils.check_if_M_connected(
                torch.tensor(newM), thr=15, returnAll=True
            )

            save_path = path_utils.path_to_reconstructions(conf, conf.phase, epoch=curr_epoch)
            for i, comp in enumerate(connected_components):
                if len(comp) < 10 and i != 0:
                    continue

                results_filtered = ba_advanced.process_camera_indices_and_bundle_adjustment(connected_cam_indices, valid_cam_mask, newM, Ks, new_Rs, new_ts,new_Xs, save_path=save_path)
                reconstructions_list.append(results_filtered)
                break  # Only run on the largest component


    ba_results_dict = {"final": reconstructions_list[-1]}

    return ba_results_dict

