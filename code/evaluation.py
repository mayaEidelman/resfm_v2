import torch
import os
import pandas as pd
import numpy as np
from utils import geo_utils, dataset_utils, path_utils, plot_utils
from utils import ba_functions
# from utils.data import export_time

from utils.Phases import Phases
# import Euclidean
from datasets import Euclidean
# import copy


def prepare_predictions(data, pred_cam, conf, bundle_adjustment, phase, curr_epoch=None):
    # Take the inputs from pred cam and turn to ndarray
    outputs = {}
    outputs['scene_name'] = data.scene_name
    calibrated = conf.get_bool('dataset.calibrated')
    images_path = path_utils.path_to_images(conf)
    conf['phase'] = phase

    Ns = data.Ns.cpu().numpy()
    Ns_inv = data.Ns_invT.transpose(1, 2).cpu().numpy()
    M = data.M.cpu().numpy()
    xs = geo_utils.M_to_xs(data.M)
    Ps_norm = pred_cam["Ps_norm"].cpu().numpy()  # Normalized camera!!
    Ps = Ns_inv @ Ps_norm  # unnormalized cameras K @ [R | t]
    pts3D_pred = geo_utils.pflat(pred_cam["pts3D"]).cpu().numpy()
    pts3D_triangulated = geo_utils.n_view_triangulation(Ps, M=M, Ns=Ns)
    xs = xs.cpu().numpy()


    if calibrated:
        Ks = Ns_inv
        Rs_gt, ts_gt = geo_utils.decompose_camera_matrix(data.y.cpu().numpy(), Ks)  # For alignment and R,t errors
        M = geo_utils.xs_to_M(geo_utils.M_to_xs(M))

        outputs['xs'] = xs  # to compute reprojection error later
        outputs['Ps'] = Ps
        outputs['Ps_norm'] = Ps_norm
        outputs['pts3D_pred'] = pts3D_pred  # 4,n
        outputs['pts3D_triangulated'] = pts3D_triangulated  # 4,n
        outputs['namesList'] = data.img_list
        outputs['outlier_indices'] = data.outlier_indices.cpu().numpy()
        outputs['validPoints'] = dataset_utils.get_M_valid_points(M)
        try:
            outputs.update(data.dict_info)
        except:
            pass
        outputs['Ks'] = Ks
        outputs['Rs_gt'] = Rs_gt
        outputs['ts_gt'] = ts_gt

        Rs_pred, ts_pred = geo_utils.decompose_camera_matrix(Ps_norm)
        outputs['Rs'] = Rs_pred
        outputs['ts'] = ts_pred

        Rs_fixed, ts_fixed, similarity_mat = geo_utils.align_cameras(Rs_pred, Rs_gt, ts_pred, ts_gt, return_alignment=True)
        outputs['Rs_fixed'] = Rs_fixed
        outputs['ts_fixed'] = ts_fixed
        outputs['pts3D_pred_fixed'] = (similarity_mat @ pts3D_pred)  # 4,n
        outputs['pts3D_triangulated_fixed'] = (similarity_mat @ pts3D_triangulated)


        if bundle_adjustment:
            repeat = conf.get_bool('ba.repeat')
            triangulation = conf.get_bool('ba.triangulation')
            # Run bundle adjustment
            ba_results_dict = ba_functions.euc_ba(conf, xs, Rs=Rs_pred, ts=ts_pred, Ks=np.linalg.inv(Ns), Xs_our=pts3D_pred.T, M=M, Ps=None, Ns=Ns, repeat=repeat, triangulation=triangulation, filtering_thr=conf.get_float('ba.filter_outliers', default=4.0), M_original=data.M_original.cpu().numpy(), images_path=images_path, img_list=data.img_list, curr_epoch=curr_epoch)

            for name, ba_result in ba_results_dict.items():
                name = "_" + name
                outputs[f'Ps_ba{name}'] = ba_result['Ps']
                outputs[f'Rs_ba{name}'] = ba_result['Rs']
                outputs[f'ts_ba{name}'] = ba_result['ts']
                outputs[f'Xs_ba{name}'] = ba_result['Xs'].T
                outputs[f'xs_ba{name}'] = ba_result['xs']
                outputs[f'xs_ba{name}'] = ba_result['xs']
                outputs[f'validCamIndices_ba{name}'] = ba_result['validCamIndices']
                outputs[f'validPoints{name}'] = dataset_utils.get_M_valid_points(ba_result['M'])

                R_ba_fixed, t_ba_fixed, similarity_mat = geo_utils.align_cameras(ba_result['Rs'], Rs_gt[outputs[f'validCamIndices_ba{name}']], ba_result['ts'], ts_gt[outputs[f'validCamIndices_ba{name}']], return_alignment=True)
                outputs[f'Rs_ba{name}_fixed'] = R_ba_fixed
                outputs[f'ts_ba{name}_fixed'] = t_ba_fixed
                outputs[f'Xs_ba{name}_fixed'] = (similarity_mat @ outputs[f'Xs_ba{name}'])

    return outputs


def prepare_outliers_predictions(data, pred_outliers, conf):
    """
    Prepare and package outlier predictions for a given scene.

    Returns:
        dict: A dictionary containing outlier predictions, camera info, and metadata.
    """
    outputs = {}

    # Basic scene metadata
    outputs['scene_name'] = data.scene_name
    outputs['img_list'] = data.img_list
    outputs['M'] = data.M.cpu().numpy()

    # Ground truth camera parameters
    outputs['Ns'] = data.Ns.cpu().numpy()
    outputs['Ps_gt'] = data.y.cpu().numpy()

    # Outlier GT and prediction
    outputs['outlier_indices'] = data.outlier_indices.cpu().numpy()

    valid_mask = dataset_utils.get_M_valid_points(data.M)
    predicted_outlier_mask = torch.zeros_like(valid_mask, dtype=data.M.dtype)
    predicted_outlier_mask[valid_mask] = pred_outliers.squeeze()

    outputs['outliers_pred__'] = pred_outliers.cpu().numpy()  # raw prediction vector
    outputs['outliers_pred'] = predicted_outlier_mask.cpu().numpy()  # full matrix mask

    return outputs

def compute_errors(outputs, conf, bundle_adjustment):
    model_errors = {}              # Aggregated error metrics (mean, median)
    model_errors_per_cam = {}     # Raw error values per camera (for plots, histograms)

    # === General setup ===
    calibrated = conf.get_bool('dataset.calibrated')
    Ps = outputs['Ps']
    xs = outputs['xs']
    pts3D_pred = outputs['pts3D_pred']
    pts3D_triangulated = outputs['pts3D_triangulated']

    # === Reprojection errors ===
    model_errors["our_repro"] = np.nanmean(
        geo_utils.reprojection_error_with_points(Ps, pts3D_pred.T, xs)
    )
    model_errors["triangulated_repro"] = np.nanmean(
        geo_utils.reprojection_error_with_points(Ps, pts3D_triangulated.T, xs)
    )

    # === Rotation & translation errors (predicted) ===
    if calibrated:
        Rs_fixed = outputs['Rs_fixed']
        ts_fixed = outputs['ts_fixed']
        Rs_gt = outputs['Rs_gt']
        ts_gt = outputs['ts_gt']

        Rs_error, ts_error = geo_utils.tranlsation_rotation_errors(Rs_fixed, ts_fixed, Rs_gt, ts_gt)

        model_errors["ts_mean"] = np.mean(ts_error)
        model_errors["ts_med"] = np.median(ts_error)
        model_errors["Rs_mean"] = np.mean(Rs_error)
        model_errors["Rs_med"] = np.median(Rs_error)

        model_errors_per_cam["Rs_err"] = Rs_error
        model_errors_per_cam["ts_err"] = ts_error

    # === Bundle Adjustment evaluation ===
    if bundle_adjustment:
        for name in ['final']:
            suffix = f"_{name}"

            Xs_ba = outputs[f'Xs_ba{suffix}']
            Ps_ba = outputs[f'Ps_ba{suffix}']
            xs_ba = outputs[f'xs_ba{suffix}']

            error_matrix = geo_utils.reprojection_error_with_points(Ps_ba, Xs_ba.T, xs_ba)
            error_per_point = np.nanmean(error_matrix, axis=0)
            error_per_camera = np.nanmean(error_matrix, axis=1)

            # Filter: inlier 3D points and cameras based on error thresholds
            inlier_pts_mask = error_per_point < 4.0
            inlier_cam_mask = error_per_camera < 5.0
            if(np.sum(inlier_cam_mask) / len(inlier_cam_mask)) > 0.8:
                pass
            else:
                inlier_cam_mask = np.ones_like(inlier_cam_mask, dtype=bool)

            model_errors[f"repro_ba{suffix}"] = np.mean(error_per_point[inlier_pts_mask])
            model_errors[f"#registered_cams{suffix}"] = int(np.sum(inlier_cam_mask))

            outputs[f"#registered_cams{suffix}"] = int(np.sum(inlier_cam_mask))
            outputs["registered_cams_indices"] = inlier_cam_mask

            if calibrated:
                Rs_fixed = outputs[f'Rs_ba{suffix}_fixed']
                ts_fixed = outputs[f'ts_ba{suffix}_fixed']
                valid_cam_ids = outputs[f'validCamIndices_ba{suffix}']
                Rs_gt = outputs['Rs_gt']
                ts_gt = outputs['ts_gt']

                Rs_ba_err, ts_ba_err = geo_utils.tranlsation_rotation_errors(
                    Rs_fixed[inlier_cam_mask],
                    ts_fixed[inlier_cam_mask],
                    Rs_gt[valid_cam_ids][inlier_cam_mask],
                    ts_gt[valid_cam_ids][inlier_cam_mask]
                )

                model_errors[f"ts_ba{suffix}_mean"] = np.mean(ts_ba_err)
                model_errors[f"ts_ba{suffix}_med"] = np.median(ts_ba_err)
                model_errors[f"Rs_ba{suffix}_mean"] = np.mean(Rs_ba_err)
                model_errors[f"Rs_ba{suffix}_med"] = np.median(Rs_ba_err)

                model_errors_per_cam[f"Rs_ba{suffix}_err"] = Rs_ba_err
                model_errors_per_cam[f"ts_ba{suffix}_err"] = ts_ba_err


    projected_mask = geo_utils.get_positive_projected_pts_mask(
        Ps @ pts3D_pred,
        conf.get_float('loss.infinity_pts_margin')
    )
    valid_mask = geo_utils.xs_valid_points(xs)
    unprojectable_mask = np.logical_and(~projected_mask, valid_mask)
    frac_unprojectable = unprojectable_mask.sum() / valid_mask.sum()

    model_errors['unprojected'] = frac_unprojectable

    return model_errors, model_errors_per_cam, outputs


def organize_errors(errors_list):
    """
    Organize a list of per-scene error dictionaries into a summary DataFrame.

    Args:
        errors_list (List[Dict[str, float]]): List of dictionaries, one per scene,
            where each dict includes a 'Scene' key and various error metrics.

    Returns:
        pd.DataFrame: A DataFrame indexed by scene name, with a final "Mean" row
            summarizing the average across all scenes.
    """
    df_errors = pd.DataFrame(errors_list)

    # Compute mean across all numeric columns (excluding 'Scene')
    metric_columns = [col for col in df_errors.columns if col != "Scene"]
    mean_row = df_errors[metric_columns].mean()

    # Append the mean as a new row
    df_errors = pd.concat([df_errors, pd.DataFrame([mean_row])], ignore_index=True)

    # Label the last row as 'Mean'
    df_errors.at[df_errors.index[-1], "Scene"] = "Mean"

    # Set 'Scene' as the index and round all values
    df_errors.set_index("Scene", inplace=True)
    df_errors = df_errors.round(3)

    return df_errors


