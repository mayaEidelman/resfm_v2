import torch
import cv2
import numpy as np
import cvxpy as cp
from numpy.random._mt19937 import MT19937
from numpy.random import Generator
from utils import dataset_utils
import dask.array as da


def compare_rotationsError(R1, R2):
	if isinstance(R1, np.ndarray):
		cos_err = (R1 @ np.transpose(R2, [0, 2, 1]))[:, np.arange(3), np.arange(3)]
		cos_err = (cos_err.sum(axis=-1) - 1) / 2
	else:
		cos_err = (torch.bmm(R1, R2.transpose(1, 2))[:, torch.arange(3), torch.arange(3)].sum(dim=-1) - 1) / 2
	cos_err[cos_err > 1] = 1
	cos_err[cos_err < -1] = -1
	return np.arccos(cos_err) * 180 / np.pi


def project_to_rot(m):
	u, s, v = torch.svd(m)
	vt = torch.transpose(v, 1, 2)
	det = torch.det(torch.matmul(u, vt))
	det = det.view(-1, 1, 1)
	vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
	return torch.matmul(u, vt)


def tranlsation_rotation_errors(R_fixed, t_fixed, gt_Rs, gt_ts):
	R_error = compare_rotationsError(R_fixed, gt_Rs)
	t_error = np.linalg.norm(t_fixed - gt_ts, axis=-1)
	return R_error, t_error


def align_cameras(pred_Rs, gt_Rs, pred_ts, gt_ts, return_alignment=False):
	'''

    :param pred_Rs: torch double - n x 3 x 3 predicted camera rotation
    :param gt_Rs: torch double - n x 3 x 3 camera ground truth rotation
    :param pred_ts: torch double - n x 3 predicted translation
    :param gt_ts: torch double - n x 3 ground truth translation
    :return:
    '''
	# find rotation
	d = 3
	n = pred_Rs.shape[0]

	Q = np.sum(gt_Rs @ np.transpose(pred_Rs, [0, 2, 1]), axis=0)  # sum over the n views of R_gt[i] @ R_pred[i].T
	Uq, _, Vqh = np.linalg.svd(Q)
	sv = np.ones(3)
	sv[-1] = np.linalg.det(Uq @ Vqh)
	R_opt = Uq @ np.diag(sv) @ Vqh

	R_fixed = R_opt.reshape([1, 3, 3]) @ pred_Rs

	# find translation
	pred_ts = pred_ts @ R_opt.T  # Apply the optimal rotation on all the translations
	c_opt = cp.Variable()
	t_opt = cp.Variable((1, d))


	constraints = []
	obj = cp.Minimize(cp.sum(cp.norm(gt_ts - (c_opt * pred_ts + np.ones((n, 1), dtype=np.double) @ t_opt), axis=1)))
	prob = cp.Problem(obj, constraints)
	prob.solve(solver=cp.CLARABEL)

	t_fixed = c_opt.value * pred_ts + t_opt.value.reshape([1, 3])

	if return_alignment:
		similarity_mat = np.eye(4)
		similarity_mat[0:3, 0:3] = c_opt.value * R_opt
		similarity_mat[0:3, 3] = t_opt.value
		return R_fixed, t_fixed, similarity_mat
	else:
		return R_fixed, t_fixed

def decompose_camera_matrix(Ps, Ks=None):
	"""Decomposes a camera matrix into its rotation and translation components.
  Args:
    Ps: A batch of camera matrices.
    Ks: A batch of intrinsic camera matrices. If None, the identity matrix is used.
  Returns:
    Rs: A batch of rotation matrices.
    ts: A batch of translation vectors.
  """
	if isinstance(Ps, np.ndarray):
		Rt = np.linalg.inv(Ks) @ Ps if Ks is not None else Ps
		Rs = np.transpose(Rt[:, 0:3, 0:3], [0, 2, 1])
		ts = (-Rs @ Rt[:, 0:3, 3].reshape([-1, 3, 1])).squeeze()
	else:
		n_cams = Ps.shape[0]
		if Ks is None:
			Ks = torch.eye(3, device=Ps.device).expand((n_cams, 3, 3))

		Rt = torch.bmm(Ks.inverse(), Ps)
		Rs = Rt[:, 0:3, 0:3].transpose(1, 2)
		ts = torch.bmm(-Rs, Rt[:, 0:3, 3].unsqueeze(-1)).squeeze()

	return Rs, ts

def decompose_essential_matrix(E, x1, x2):
	# [R1,t], [R1,−t], [R2,t], [R2,−t]
	if x1.shape[0] == 3:
		x1 = x1 / x1[2]
		x1 = x1[:2]
		x2 = x2 / x2[2]
		x2 = x2[:2]
	R1, R2, t = cv2.decomposeEssentialMat(E=E)
	pose_options = [[R1, t], [R1, -t], [R2, t], [R2, -t]]
	P2_options = [np.concatenate(pose_options[i], axis=1) for i in range(4)]
	P1 = np.concatenate([np.eye(3), np.zeros([3, 1])], axis=1)
	number_of_points_in_front = np.zeros([4, 2])
	for i, P2 in enumerate(P2_options):
		X = pflat(cv2.triangulatePoints(P1, P2, x1, x2))
		proj1 = np.dot(P1, X)
		proj2 = np.dot(P2, X)
		number_of_points_in_front[i, 0] = np.sum(proj1[-1] > 0)
		number_of_points_in_front[i, 1] = np.sum(proj2[-1] > 0)
	best_option = int(np.argmax(np.sum(number_of_points_in_front, axis=1)))
	return pose_options[best_option]


def M_to_xs(M):
	"""
    reshapes the 2d points
    :param M: [2*m, n]
    :return: xs [m,n,2]
    """
	m, n = M.shape
	m = m // 2
	xs = M.reshape([m, 2, n])
	if isinstance(M, np.ndarray):
		xs = np.transpose(xs, [0, 2, 1])
	else:
		xs = xs.transpose(1, 2)
	return xs


def xs_to_M(xs):
	n, m, _ = xs.shape
	if isinstance(xs, np.ndarray):
		xs_tran = np.transpose(xs, [0, 2, 1])
	else:
		xs_tran = xs.transpose(1, 2)
	M = xs_tran.reshape([2 * n, m])
	return M


def get_V_from_RK(Rs, Ks):
	return torch.inverse(Ks).permute(0, 2, 1) @ Rs.permute(0, 2, 1)


def get_cross_product_matrix(t):
	T = torch.zeros((3, 3), device=t.device)
	T[0, 1] = -t[2]
	T[0, 2] = t[1]
	T[1, 0] = t[2]
	T[1, 2] = -t[0]
	T[2, 0] = -t[1]
	T[2, 1] = t[0]
	return T


def batch_get_cross_product_matrix(t):
	batch_size = t.shape[0]
	T = torch.zeros((batch_size, 3, 3), device=t.device)
	T[:, 0, 1] = -t[:, 2]
	T[:, 0, 2] = t[:, 1]
	T[:, 1, 0] = t[:, 2]
	T[:, 1, 2] = -t[:, 0]
	T[:, 2, 0] = -t[:, 1]
	T[:, 2, 1] = t[:, 0]
	return T


def batch_get_bifocal_tensors(Rs, ts):
	n = len(Rs)
	E = torch.zeros([n, n, 3, 3])
	for i in range(n):
		for j in range(n):
			E[i, j] = Rs[i].T @ get_cross_product_matrix(ts[i] - ts[j]) @ Rs[j]
	return E


def batch_fundamental_from_essential(E, Ks):
	n = len(Ks)
	F = torch.zeros_like(E)
	for i in range(n):
		for j in range(n):
			F[i, j] = torch.inverse(Ks[i]).T @ E[i, j] @ torch.inverse(Ks[j])
	return F


def get_fundamental_from_V_t(Vi, Vj, ti, tj):
	Ti = get_cross_product_matrix(ti)
	Tj = get_cross_product_matrix(tj)
	return Vi @ (Ti - Tj) @ Vj.T


def batch_get_fundamental_from_V_t(Vi, Vj, ti, tj):
	Ti = batch_get_cross_product_matrix(ti)
	Tj = batch_get_cross_product_matrix(tj)
	return torch.bmm(Vi, torch.bmm((Ti - Tj), Vj.transpose(1, 2)))


def get_camera_matrix(R, t, K):
	"""
    Get the camera matrix as described in paper
    :param R: Orientation Matrix
    :param t: Camera Position   
    :param K: Intrinsic parameters
    :return: Camera matrix
    """
	if isinstance(R, np.ndarray):
		return K @ R.T @ np.concatenate((np.eye(3), -t.reshape(3, 1)), axis=1)
	else:
		return K @ R.T @ torch.cat((torch.eye(3, device=R.device), -t.view(3, 1)), dim=1)


def batch_get_camera_matrix_from_rtk(Rs, ts, Ks):
	n = len(Rs)
	if isinstance(Rs, np.ndarray):
		Ps = np.zeros([n, 3, 4])
	else:
		Ps = torch.zeros([n, 3, 4], device=Rs.device)
	for i, r, t, k in zip(np.arange(n), Rs, ts, Ks):
		Ps[i] = get_camera_matrix(r, t, k)
	return Ps


def get_camera_matrix_from_Vt(V, t):
	"""
    Get the camera matrix as described in paper
    :param V: inv(K).T @ R.T Orientation Matrix
    :param t: Camera Position
    :return: Camera matrix
    """
	return torch.inverse(V).T @ torch.cat((torch.eye(3), -t), dim=1)


def batch_get_camera_matrix_from_Vt(Vs, ts):
	vT_inv = torch.inverse(Vs).transpose(1, 2)
	return torch.cat((vT_inv, torch.bmm(vT_inv, -ts.unsqueeze(-1))), dim=2)


def pflat(x):
	return x / x[-1, :]


def batch_pflat(x):
	return x / x[:, 2:3, :]


def correct_matches(P1, P2, pts_img1, pts_img2):
	"""
    This function corrects the matches between two images using global triangulation.
    """

	pts3D = cv2.triangulatePoints(P1, P2, pts_img1[0:2, :], pts_img2[0:2, :])
	pts_img1 = torch.from_numpy(pflat(P1 @ pts3D)).float()
	pts_img2 = torch.from_numpy(pflat(P2 @ pts3D)).float()
	return pts_img1, pts_img2


def calc_reprojection_error(P1, P2, pts_img1, pts_img2):
	"""
        This function calculates the reprojection error between two images.
        """

	coorected_pts_img1, coorected_pts_img2 = correct_matches(P1.numpy(), P2.numpy(), pts_img1.numpy(), pts_img2.numpy())
	reproj_err1 = torch.norm(pflat(coorected_pts_img1)[0:2, :] - pflat(pts_img1)[0:2, :], dim=0, p=2)
	reproj_err2 = torch.norm(pflat(coorected_pts_img2)[0:2, :] - pflat(pts_img2)[0:2, :], dim=0, p=2)
	return reproj_err1, reproj_err2




def calc_global_reprojection_error(Ps, M, Ns):
	n = len(Ps)
	valid_pts = dataset_utils.get_M_valid_points(M)
	X = n_view_triangulation(Ps, M, Ns)
	projected_pts = batch_pflat(Ps @ X)[:, 0:2, :]
	image_points = M.reshape(n, 2, M.shape[-1])
	reproj_err = np.linalg.norm(image_points - projected_pts, axis=1)
	return np.where(valid_pts, reproj_err, np.full(reproj_err.shape, np.nan))



def calc_global_reprojection_error_withPoints(Ps, M, Ns, X):
	n = len(Ps)
	valid_pts = dataset_utils.get_M_valid_points(M)
	projected_pts = batch_pflat(Ps @ X)[:, 0:2, :]
	image_points = M.reshape(n, 2, M.shape[-1])
	reproj_err = np.linalg.norm(image_points - projected_pts, axis=1)
	return np.where(valid_pts, reproj_err, np.full(reproj_err.shape, np.nan))


def reprojection_error_with_points(Ps, Xs, xs, visible_points=None):
	"""
    :param Ps: [m,3,4]
    :param Xs: [n,3] or [n,4]
    :param xs: [m,n,2]
    :return: errors [m,n]
    """
	m, n, d = xs.shape
	_, D = Xs.shape
	X4 = np.concatenate([Xs, np.ones([n, 1])], axis=1) if D == 3 else Xs

	if visible_points is None:
		visible_points = xs[:, :, 0] > 0

	projected_points = Ps @ X4.T  # [m,3,4] @ [4,n] -> [m,3,n]
	if isinstance(projected_points, np.ndarray):
		projected_points = np.transpose(projected_points, [0, 2, 1])  # [m,n,3]
	else:
		projected_points = projected_points.transpose(1, 2)
	projected_points = projected_points / projected_points[:, :, -1].reshape([m, n, 1])
	errors = np.linalg.norm(xs[:, :, :2] - projected_points[:, :, :2], axis=2)
	errors[~visible_points] = np.nan
	return errors


def get_points_in_view(M, img_idx, X=None):
	points_indices = np.logical_or(M[2 * img_idx, :] != 0, M[2 * img_idx + 1, :] != 0)
	if X is not None:
		points_indices = np.logical_and(points_indices, ~np.isnan(X[0, :]))

	return np.where(points_indices)[0]


def normalize_points_cams(Ps, xs, Ns):
	"""
    Normalize the points and the cameras using the matrices in N.
    if :
    xs[i,j] ~ P[i] @ X[j]
    than so is:
     N[i] @ xs[i,j] ~ N[i] @ P[i] @ X[j]
    :param Ps:  [m,3,4]
    :param xs:  [m,n,2] or [m,n,3]
    :param Ns:  [m,3,3]
    :return:  norm_P, norm_x
    """
	m, n, d = xs.shape
	xs_3 = np.concatenate([xs, np.ones([m, n, 1])], axis=2) if d == 2 else xs
	norm_P = np.zeros_like(Ps)
	norm_x = np.zeros_like(xs)
	for i in range(m):
		norm_P[i] = Ns[i] @ Ps[i]  # [3,3] @ [3,4]
		norm_points = (Ns[i] @ xs_3[i].T).T  # ([3,3] @ [3,n]) -> [n,3]
		norm_points[norm_points[:, -1] == 0, -1] = 1
		norm_points = norm_points / norm_points[:, -1].reshape([-1, 1])
		if d == 2:
			norm_x[i] = norm_points[:, :2]
	return norm_P, norm_x


def dlt_triangulation(Ps, xs, visible_points):
	"""
    Use  linear triangulation to find the points X[j] such that  xs[i,j] ~ P[i] @ X[j]
    :param Ps:  [m,3,4]
    :param xs: [m,n,2] or [m,n,3]
    :param visible_points: [m,n] a boolean matrix of which cameras see which points
    :return: Xs [n,4] normalized such the X[j,-1] == 1
    """
	m, n, _ = xs.shape
	X = np.zeros([n, 4])
	for i in range(n):
		cameras_showing_ind = np.where(visible_points[:, i])[0]  # The cameras that show this point
		num_cam_show = len(cameras_showing_ind)
		if num_cam_show < 2:
			X[i] = np.nan
			continue
		A = np.zeros([3 * num_cam_show, num_cam_show + 4])
		for j, cam_index in enumerate(cameras_showing_ind):
			xij = xs[cam_index, i, :2]
			Pj = Ps[cam_index]
			A[3 * j:3 * (j + 1), :4] = Pj
			A[3 * j:3 * j + 2, 4 + j] = -xij
			A[3 * j + 2, 4 + j] = -1

		if num_cam_show > 40:
			[U, S, V_H] = da.linalg.svd(
				da.from_array(A))  # in python svd returns V conjugate! so we need the last row and not column
			X[i] = pflat(V_H[-1, :4].compute().reshape([-1, 1])).squeeze()
		else:
			try:
				[U, S, V_H] = np.linalg.svd(A)  # in python svd returns V conjugate! so we need the last row and not column
				X[i] = pflat(V_H[-1, :4].reshape([-1, 1])).squeeze()
			except:
				pass

	return X



def n_view_triangulation(Ps, M, Ns=None):
	# normalizing matrix can be K inverse or a matrix that normalizes the points in M
	xs = M_to_xs(M)
	visible_points = xs[:, :, 0] > 0
	if Ns is not None:
		Ps = Ps.copy()
		Ps, xs = normalize_points_cams(Ps, xs, Ns)
	X = dlt_triangulation(Ps, xs, visible_points)
	return X.T


def xs_valid_points(xs):
	"""

    :param xs: [m,n,2]
    :return: A boolean matrix of the visible 2d points
    """
	if isinstance(xs, np.ndarray):
		return np.logical_or(xs[:, :, 0] > 0, xs[:, :, 1] > 0)
	else:
		return torch.logical_or(xs[:, :, 0] > 0, xs[:, :, 1] > 0)
	#return xs[:, :, 0] > 0


def normalize_M(M, Ns, valid_points=None):
	# from pixel coordinates to camera coordinates using inv calibration or normalization matrix
	if valid_points is None:
		valid_points = dataset_utils.get_M_valid_points(M)
	norm_M = M.clone()
	n_images = norm_M.shape[0] // 2
	norm_M = norm_M.reshape([n_images, 2, -1])  # [m,2,n]
	norm_M = torch.cat((norm_M, torch.ones(n_images, 1, norm_M.shape[-1], device=M.device)), dim=1)  # [m,3,n]

	norm_M = (Ns @ norm_M).permute(0, 2, 1)[:, :, :2]  # [m,3,3]@[m,3,n] -> [m,3,n]->[m,n,3]
	norm_M[~valid_points] = 0
	return norm_M


def normalize_ts(ts, return_parameters=False):
	ts_norm = ts.clone()

	trans_vec = ts_norm.mean(dim=0)
	ts_norm = ts_norm - trans_vec

	scale_factor = ts_norm.norm(p=2, dim=1).mean()
	ts_norm = ts_norm / scale_factor

	if return_parameters:
		return ts_norm, trans_vec, scale_factor
	else:
		return ts_norm


def get_positive_projected_pts_mask(pts2D, infinity_pts_margin):
	return pts2D[:, 2, :] >= infinity_pts_margin


def get_projected_pts_mask(pts2D, infinity_pts_margin):
	return pts2D[:, 2, :].abs() >= infinity_pts_margin


def ones_padding(pts):
	return torch.cat((pts, torch.ones(1, pts.shape[1], device=pts.device)))


def dilutePoint(M):
	# delete point tracks which are viewed from less than 'param' cameras
	if M.shape[1] > 20000:
		param = 4
	else:
		param = 3

	valid_pts = dataset_utils.get_M_valid_points(M)
	rm_pts = valid_pts.sum(axis=0) < param
	keep_pts = ~rm_pts
	newM = M[:, keep_pts]
	return newM


def remove_empty_tracks_cams(M, Ps=None, Ns=None, outliers=None, Xs=None, data=None, pts_per_cam_thresh=1,
                             cam_per_pts_thresh=2):
	"""
    Remove columns/rows that don't pass thresholds by points count
    """

	valid_pts = dataset_utils.get_M_valid_points(M)
	cam_per_pts = valid_pts.sum(axis=0)  # [n_pts]
	pts_per_cam = valid_pts.sum(axis=1)  # [n_cams]
	M = M[pts_per_cam.repeat(2) >= pts_per_cam_thresh]
	M = M[:, cam_per_pts >= cam_per_pts_thresh]

	if Xs is not None:
		Xs = Xs[cam_per_pts >= cam_per_pts_thresh]
	if Ps is not None:
		Ps = Ps[pts_per_cam >= pts_per_cam_thresh]
	if Ns is not None:
		Ns = Ns[pts_per_cam >= pts_per_cam_thresh]
	if outliers is not None:
		outliers = outliers[pts_per_cam >= pts_per_cam_thresh]
		outliers = outliers[:, cam_per_pts >= cam_per_pts_thresh]

	camValidIndices = pts_per_cam >= pts_per_cam_thresh
	pts3DValidIndices = cam_per_pts >= cam_per_pts_thresh
	return M, Ps, Ns, outliers, Xs, camValidIndices, pts3DValidIndices


##################################################################################################

def compute_relative_pose_from_matches(pts1, pts2, K1=None, K2=None, method='8pt'):
    """
    Compute relative pose between two cameras from point matches.
    
    Args:
        pts1: [2, N] or [3, N] points in first image
        pts2: [2, N] or [3, N] points in second image  
        K1: Intrinsic matrix for first camera (optional)
        K2: Intrinsic matrix for second camera (optional)
        method: '8pt' for 8-point algorithm, '5pt' for 5-point algorithm
        
    Returns:
        R: Relative rotation matrix [3, 3]
        t: Relative translation vector [3, 1]
        inliers: Boolean mask of inlier matches
    """
    if isinstance(pts1, torch.Tensor):
        pts1 = pts1.cpu().numpy()
    if isinstance(pts2, torch.Tensor):
        pts2 = pts2.cpu().numpy()
    
    # Ensure points are in correct format
    if pts1.shape[0] == 3:
        pts1 = pts1[:2] / pts1[2]
    if pts2.shape[0] == 3:
        pts2 = pts2[:2] / pts2[2]
    
    pts1 = pts1.T  # [N, 2]
    pts2 = pts2.T  # [N, 2]
    
    if method == '8pt':
        # 8-point algorithm
        if K1 is not None and K2 is not None:
            # Calibrated case - compute essential matrix
            E, inliers = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, 
                                            prob=0.999, threshold=1.0)
        else:
            # Uncalibrated case - compute fundamental matrix
            F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 1.0, 0.999)
            if K1 is not None and K2 is not None:
                E = K2.T @ F @ K1
            else:
                # For uncalibrated case, we can't get unique relative pose
                return None, None, inliers
    elif method == '5pt':
        # 5-point algorithm (calibrated case only)
        if K1 is None or K2 is None:
            raise ValueError("5-point algorithm requires calibrated cameras")
        E, inliers = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, 
                                        prob=0.999, threshold=1.0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if E is None or inliers is None:
        return None, None, None
    
    # Decompose essential matrix to get relative pose
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K1, mask=inliers)
    
    return R, t, inliers


def batch_compute_relative_poses(matches_dict, Ks=None):
    """
    Compute relative poses for all camera pairs in a batch.
    
    Args:
        matches_dict: Dictionary with keys (i, j) containing match data
        Ks: List of intrinsic matrices [m, 3, 3] or None for uncalibrated case
        
    Returns:
        relative_poses: Dictionary with keys (i, j) containing relative poses
    """
    relative_poses = {}
    
    for (i, j), matches in matches_dict.items():
        pts1 = matches['pts1']
        pts2 = matches['pts2']
        
        K1 = Ks[i] if Ks is not None else None
        K2 = Ks[j] if Ks is not None else None
        
        R, t, inliers = compute_relative_pose_from_matches(pts1, pts2, K1, K2)
        
        if R is not None and t is not None:
            relative_poses[(i, j)] = {
                'R': torch.from_numpy(R).float(),
                't': torch.from_numpy(t).float(),
                'inliers': torch.from_numpy(inliers).bool() if inliers is not None else None
            }
    
    return relative_poses


def compute_epipolar_constraint_error(F, pts1, pts2):
    """
    Compute epipolar constraint error using fundamental matrix.
    
    Args:
        F: Fundamental matrix [3, 3]
        pts1: Points in first image [2, N] or [3, N]
        pts2: Points in second image [2, N] or [3, N]
        
    Returns:
        error: Mean epipolar error
    """
    if isinstance(F, torch.Tensor):
        F = F.cpu().numpy()
    if isinstance(pts1, torch.Tensor):
        pts1 = pts1.cpu().numpy()
    if isinstance(pts2, torch.Tensor):
        pts2 = pts2.cpu().numpy()
    
    # Ensure points are in homogeneous coordinates
    if pts1.shape[0] == 2:
        pts1 = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    if pts2.shape[0] == 2:
        pts2 = np.vstack([pts2, np.ones((1, pts2.shape[1]))])
    
    # Compute epipolar lines
    lines1 = F @ pts2  # F * x2
    lines2 = F.T @ pts1  # F^T * x1
    
    # Compute distances
    dist1 = np.abs(np.sum(pts1 * lines1, axis=0)) / np.linalg.norm(lines1[:2, :], axis=0)
    dist2 = np.abs(np.sum(pts2 * lines2, axis=0)) / np.linalg.norm(lines2[:2, :], axis=0)
    
    # Symmetric epipolar distance
    return np.mean(dist1 + dist2)


def validate_relative_pose(R, t, pts1, pts2, K1=None, K2=None, threshold=4.0):
    """
    Validate relative pose by checking if points triangulate in front of both cameras.
    
    Args:
        R: Relative rotation [3, 3]
        t: Relative translation [3, 1]
        pts1: Points in first image [2, N]
        pts2: Points in second image [2, N]
        K1: Intrinsic matrix for first camera
        K2: Intrinsic matrix for second camera
        threshold: Reprojection error threshold
        
    Returns:
        valid: Boolean indicating if pose is valid
        error: Mean reprojection error
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    if isinstance(pts1, torch.Tensor):
        pts1 = pts1.cpu().numpy()
    if isinstance(pts2, torch.Tensor):
        pts2 = pts2.cpu().numpy()
    
    # Create camera matrices
    P1 = np.eye(3, 4)  # First camera at origin
    P2 = np.hstack([R, t])  # Second camera with relative pose
    
    if K1 is not None and K2 is not None:
        P1 = K1 @ P1
        P2 = K2 @ P2
    
    # Triangulate points
    points_3d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_3d / points_3d[3]  # Normalize homogeneous coordinates
    
    # Project back to images
    proj1 = P1 @ points_3d
    proj2 = P2 @ points_3d
    
    # Check if points are in front of cameras
    in_front1 = proj1[2] > 0
    in_front2 = proj2[2] > 0
    in_front = in_front1 & in_front2
    
    if not np.any(in_front):
        return False, float('inf')
    
    # Compute reprojection error
    proj1 = proj1[:2] / proj1[2]
    proj2 = proj2[:2] / proj2[2]
    
    error1 = np.linalg.norm(pts1 - proj1, axis=0)
    error2 = np.linalg.norm(pts2 - proj2, axis=0)
    
    mean_error = np.mean(error1 + error2)
    
    return mean_error < threshold, mean_error
