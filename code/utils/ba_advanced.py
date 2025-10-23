import os
import numpy as np
import torch
from utils import dataset_utils
from utils import geo_utils
import pyceres
import pycolmap

def get_negative_set(lst, size):
    return list(set(np.arange(size)) - set(lst))


def create_ba_results(M, Ps, Rs, ts, Xs, xs, valid_cam_indices=None, valid_point_indices=None):
    """
    Package bundle adjustment results into a dictionary.

    Args:
        M (np.ndarray): [2m, n] binary visibility matrix.
        Ps (np.ndarray): [m, 3, 4] camera projection matrices.
        Rs (np.ndarray): [m, 3, 3] rotation matrices.
        ts (np.ndarray): [m, 3] translation vectors.
        Xs (np.ndarray): [n, 3] 3D point coordinates.
        xs (np.ndarray): [m, n, 2] 2D image coordinates.
        valid_cam_indices (np.ndarray or list, optional): Indices of valid cameras.
        valid_point_indices (np.ndarray or list, optional): Indices of valid 3D points.

    Returns:
        dict: Dictionary containing the structured bundle adjustment output.
    """
    Xs_h = np.concatenate([Xs, np.ones((Xs.shape[0], 1))], axis=1)  # Convert to homogeneous

    results = {
        'M': M,
        'Rs': Rs,
        'ts': ts,
        'Ps': Ps,
        'Xs': Xs_h,
        'xs': xs,
        'validCamIndices': valid_cam_indices,
        'validPtIndices': valid_point_indices,
    }

    return results


def prepare_ba_options():
    """
    Prepare and return Bundle Adjustment (BA) options for PyCOLMAP.

    Returns:
        pycolmap.BundleAdjustmentOptions: Configured BA options.
    """
    ba_options = pycolmap.BundleAdjustmentOptions()

    # Camera parameter refinement
    ba_options.refine_focal_length = False
    ba_options.refine_principal_point = False

    # Loss function
    ba_options.loss_function_type = pycolmap.LossFunctionType.CAUCHY

    # Solver settings
    # Use a dense QR solver to avoid CHOLMOD SPD requirements that can fail on ill-conditioned problems
    ba_options.solver_options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    ba_options.solver_options.function_tolerance = 1e-4
    ba_options.solver_options.max_num_iterations = 300
    ba_options.solver_options.num_threads = 24

    # Verbosity
    ba_options.print_summary = True

    return ba_options



def pycolmap_to_batch_matrix(reconstruction):
    """
    Convert a PyCOLMAP Reconstruction object back to batched camera matrices and 3D points.

    Args:
        reconstruction (pycolmap.Reconstruction): The COLMAP reconstruction object.

    Returns:
        Rs (np.ndarray): [N, 3, 3] Rotation matrices.
        ts (np.ndarray): [N, 3] Translation vectors.
        Ps (np.ndarray): [N, 3, 4] Projection matrices.
        Ks (np.ndarray): [N, 3, 3] Intrinsic calibration matrices.
        points3D (np.ndarray): [P, 3] 3D point coordinates.
    """

    num_images = len(reconstruction.images)
    max_point_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_point_id, 3))

    # Extract 3D points (COLMAP uses 1-based point3D IDs)
    for point3D_id, point in reconstruction.points3D.items():
        points3D[point3D_id - 1] = point.xyz

    Rs, ts, Ks = [], [], []

    for i in range(num_images):
        image = reconstruction.images[i]

        # Extract extrinsics: cam_from_world
        R = image.cam_from_world.rotation.matrix().T
        t = -R @ image.cam_from_world.translation
        Rs.append(R)
        ts.append(t)

        # Extract intrinsics
        K = reconstruction.cameras[image.camera_id].calibration_matrix()
        Ks.append(K)

    Rs = np.stack(Rs)  # [N, 3, 3]
    ts = np.stack(ts)  # [N, 3]
    Ks = np.stack(Ks)  # [N, 3, 3]
    Ps = geo_utils.batch_get_camera_matrix_from_rtk(Rs, ts, Ks)  # [N, 3, 4]

    return Rs, ts, Ps, Ks, points3D

def batch_matrix_to_pycolmap(xs, Rs, ts, Ks, Xs_our=None, shared_camera=False, img_list=None):
    """
    Convert batched PyTorch-like tensors to a PyCOLMAP Reconstruction object.

    Args:
        xs (np.ndarray): [N, M, 2] 2D keypoints in each image.
        Rs (np.ndarray): [N, 3, 3] Rotation matrices for each camera.
        ts (np.ndarray): [N, 3] Translation vectors for each camera.
        Ks (np.ndarray): [N, 3, 3] Intrinsic calibration matrices.
        Xs_our (np.ndarray): [P, 3] 3D points.
        shared_camera (bool): Whether all images share the same camera intrinsics.
        camera_type (str): Type of COLMAP camera model ("PINHOLE", etc.).
        img_list (list of str): Optional list of image names.

    Returns:
        pycolmap.Reconstruction: The constructed COLMAP-compatible reconstruction.
    """

    N, P, _ = xs.shape
    reconstruction = pycolmap.Reconstruction()

    # Determine visibility
    visible_points = xs[:, :, 0] > 0
    valid_mask = visible_points.sum(axis=0) >= 2  # Tracks visible in at least 2 frames
    valid_idx = np.where(valid_mask)[0]

    # Add 3D points
    points3d = Xs_our[:, :3]
    for pid in valid_idx:
        reconstruction.add_point3D(points3d[pid], pycolmap.Track(), color=np.array([173, 216, 230]))

    # Set camera intrinsics
    camera = None
    for i in range(N):
        if camera is None or not shared_camera:

            intrinsics = np.array([Ks[i][0, 0], Ks[i][1, 1], Ks[i][0, 2], Ks[i][1, 2]])
            camera_type = "PINHOLE"  # Default camera type, can be changed if needed

            width = int(Ks[i][0, 2] * 2)
            height = int(Ks[i][1, 2] * 2)
            camera = pycolmap.Camera(model=camera_type, width=width, height=height,
                                     params=intrinsics, camera_id=i)
            reconstruction.add_camera(camera)

        # Create camera pose
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(Rs[i].T),
            -Rs[i].T @ ts[i]
        )

        # Image name
        image_name = img_list[i].strip() if img_list is not None else f"image_{i:06d}.jpg"

        image = pycolmap.Image(
            id=i,
            name=image_name,
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world
        )

        # Add 2D observations
        points2D_list = []
        point2D_idx = 0

        for j, pid in enumerate(valid_idx, start=1):
            if visible_points[i, pid]:
                point2D_xy = xs[i, pid]
                points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id=j))
                reconstruction.points3D[j].track.add_element(i, point2D_idx)
                point2D_idx += 1

        if points2D_list:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        else:
            image.registered = False
            print(f"Warning: Frame {i} has no valid observations.")

        reconstruction.add_image(image)

    return reconstruction



def process_camera_indices_and_bundle_adjustment(
    connected_cam_indices, valid_cam_mask, visibility_matrix,
    Ks, Rs, ts, Xs, save_path
):
    """
    Process a subset of connected cameras and perform bundle adjustment.

    Args:
        connected_cam_indices (List[int]): Indices (within valid cameras) of a connected subset (length m').
        valid_cam_mask (np.ndarray): Boolean mask over all_cam_indices marking which cameras are valid.
        visibility_matrix (np.ndarray): [2m, n] binary matrix indicating which 3D points are visible in which cameras.
        Ks (np.ndarray): [m, 3, 3] intrinsic matrices.
        Rs (np.ndarray): [m, 3, 3] rotation matrices.
        ts (np.ndarray): [m, 3] translation vectors.
        Xs (np.ndarray): [n, 3] 3D point coordinates.
        save_path (str): Directory where reconstruction will be saved.

    Returns:
        dict: Results dictionary from bundle adjustment.
    """
    # Step 1: Convert connected camera indices to their 2-view format (each cam → 2 rows)
    connected_view_indices = [j for idx in connected_cam_indices for j in (2 * idx, 2 * idx + 1)]
    all_cam_indices = np.arange(len(Ks))

    # Step 2: Filter visibility matrix and camera parameters to match selected cameras
    visibility_matrix = visibility_matrix[connected_view_indices]
    Ks = Ks[valid_cam_mask][connected_cam_indices]
    Rs = Rs[valid_cam_mask][connected_cam_indices]
    ts = ts[valid_cam_mask][connected_cam_indices]

    # Step 3: Remove empty tracks and cameras
    visibility_matrix, _, _, _, Xs, _, valid_point_indices = geo_utils.remove_empty_tracks_cams(
        visibility_matrix, Xs=Xs, pts_per_cam_thresh=0, cam_per_pts_thresh=1
    )

    dataset_utils.check_if_M_connected(torch.tensor(visibility_matrix), thr=5, return_largest_component=False)

    # Step 4: Convert to 2D point array and filter visible ones
    xs = geo_utils.M_to_xs(visibility_matrix)
    visibility_mask = xs[:, :, 0] > 0
    xs_visible = xs[visibility_mask]

    # Step 5: Run bundle adjustment
    reconstruction = batch_matrix_to_pycolmap(xs, Rs, ts, Ks, Xs)
    ba_options = prepare_ba_options()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    os.makedirs(save_path, exist_ok=True)
    reconstruction.write_text(save_path)

    # Step 6: Extract updated parameters from the reconstruction
    Rs, ts, Ps, Ks, Xs = pycolmap_to_batch_matrix(reconstruction)

    # Step 7: Package results
    final_cam_indices = all_cam_indices[valid_cam_mask][connected_cam_indices]
    ba_results = create_ba_results(
        visibility_matrix, Ps, Rs, ts, Xs, xs, final_cam_indices, valid_point_indices
    )

    return ba_results

