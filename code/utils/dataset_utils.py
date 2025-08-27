import torch
from utils import geo_utils, general_utils, sparse_utils, plot_utils
from utils.Phases import Phases
import numpy as np
import networkx as nx
import copy



def is_valid_sample(data, min_pts_per_cam=8, phase=Phases.TRAINING):
    if phase is Phases.TRAINING:
        return data.x.pts_per_cam.min().item() >= min_pts_per_cam
    else:
        return True


def divide_indices_to_train_test(N, n_val, n_test=0):
    perm = np.random.permutation(N)
    test_indices = perm[:n_test] if n_test>0 else []
    val_indices = perm[n_test:n_test+n_val]
    train_indices = perm[n_test+n_val:]
    return train_indices, val_indices, test_indices


def sample_indices(N, num_samples, adjacent):
    if num_samples == 1:  # Return all the data
        indices = np.arange(N)
    else:
        if num_samples < 1:  # fraction
            num_samples = int(np.ceil(num_samples * N))
        num_samples = max(2, num_samples)
        if num_samples >= N:
            return np.arange(N)
        if adjacent:
            start_ind = np.random.randint(0,N-num_samples+1)
            end_ind = start_ind+num_samples
            indices = np.arange(start_ind, end_ind)
        else:
            indices = np.random.choice(N,num_samples,replace=False)
    return indices


def save_cameras(outputs, conf, curr_epoch, phase):
    xs = outputs['xs']
    M = geo_utils.xs_to_M(xs)
    general_utils.save_camera_mat(conf, outputs, outputs['scene_name'], phase, curr_epoch)

def save_outliers(outputs, conf, curr_epoch, phase):
    if curr_epoch is None:
        general_utils.save_outliers_mat(conf, outputs, outputs['scene_name'], phase, curr_epoch)

# def save_metrics(outputs, conf, curr_epoch, phase):
#     general_utils.save_outliers_mat(conf, outputs, outputs['scene_name'], phase, curr_epoch)


def get_data_statistics(all_data, outputs=None):
    valid_pts = all_data.valid_pts
    valid_pts_stat = valid_pts.sum(dim=0).float()
    stats = {"Max_2d_pt": all_data.M.max().item(), "Num_2d_pts": valid_pts.sum().item(), "n_pts": all_data.M.shape[-1],
             "pts_per_cam_mean":  valid_pts.sum(dim=1).float().mean().item(), "Cameras_per_pts_mean": valid_pts_stat.mean().item(), "Cameras_per_pts_std": valid_pts_stat.std().item(),
             "Num of cameras": all_data.y.shape[0]}

    if outputs is not None:
        dict_info = all_data.dict_info.copy()
        dict_info.pop("outliers_pred", None)  # safely remove if exists
        stats.update(dict_info)

    return stats


def correct_matches_global(M, Ps, Ns):
    """
    This function corrects the matches using global triangulation.
    Args:
        M (torch.Tensor): A tensor of shape (N, 2), where N is the number of matches. ## correct this line
        Ps (torch.Tensor): A tensor of shape (N, 3, 4), where N is the number of cameras.
        Ns (torch.Tensor): A tensor of shape (N, 3, 3), where N is the number of cameras.
    Returns:
        xs (torch.Tensor): A tensor of shape M
    """

    # First, we get the invalid points in M.

    M_invalid_pts = np.logical_not(get_M_valid_points(M))

    # Next, we perform global triangulation to get the corrected matches.

    Xs = geo_utils.n_view_triangulation(Ps, M, Ns)
    xs = geo_utils.batch_pflat((Ps @ Xs))[:, 0:2, :]

    # Finally, we remove the invalid points from xs.

    xs[np.isnan(xs)] = 0
    # xs[np.stack((M_invalid_pts, M_invalid_pts), axis=1)] = 0
    xs = xs.reshape(M.shape)
    xs = torch.tensor(xs)
    xs = xs.transpose(0, 1).reshape(-1, xs.shape[0] // 2, 2).transpose(0, 1)
    xs[M_invalid_pts] = 0
    xs = xs.transpose(0, 1).reshape(-1, xs.shape[0] * 2).transpose(0, 1)

    return xs.numpy()



def get_M_valid_points(M):
    n_pts = M.shape[-1]

    if type(M) is torch.Tensor:
        M_valid_pts = torch.abs(M.reshape(-1, 2, n_pts)).sum(dim=1) != 0 # zero point
        M_valid_pts[:, M_valid_pts.sum(dim=0) < 2] = False  # mask out point tracks that contain only 1 point (viewed only from one camera-f)
    else:
        M_valid_pts = np.abs(M.reshape(-1, 2, n_pts)).sum(axis=1) != 0
        M_valid_pts[:, M_valid_pts.sum(axis=0) < 2] = False

    return M_valid_pts





def M2sparse(M, normalize=False, Ns=None, M_original=None, features=None):
    n_pts = M.shape[1]
    n_cams = int(M.shape[0] / 2)

    # Get indices
    valid_pts = get_M_valid_points(M)
    cam_per_pts = valid_pts.sum(dim=0).unsqueeze(1)  # [n_pts, 1]
    pts_per_cam = valid_pts.sum(dim=1).unsqueeze(1)  # [n_cams, 1]
    mat_indices = torch.nonzero(valid_pts).T  # [2, the number of points in the scene]
    # Get Values
    # reshaped_M = M.reshape(n_cams, 2, n_pts).transpose(1, 2)  # [2m, n] -> [m, 2, n] -> [m, n, 2]
    if normalize:
        norm_M = geo_utils.normalize_M(M, Ns)
        mat_vals = norm_M[mat_indices[0], mat_indices[1], :]
    else:
        mat_vals = M.reshape(n_cams, 2, n_pts).transpose(1, 2)[mat_indices[0], mat_indices[1], :]

    mat_shape = (n_cams, n_pts, 2)
    
    return sparse_utils.SparseMat(mat_vals, mat_indices, cam_per_pts, pts_per_cam, mat_shape)


def get_M_view_adjacency(M):
    """
    Calculates the view adjacency matrix from  M.

    Args:
        M: A tracks tensor (2 * num_views, num_points).

    Returns:
        view_graph_adj: A torch.Tensor of shape (num_views, num_views) representing the view adjacency matrix,
                       where view_graph_adj[i, j] is the number of shared visible points between views i and j.
    """
    view_graph_adj = torch.zeros([M.shape[0] // 2, M.shape[0] // 2], dtype=torch.int32, device=M.device)
    M_valid_pts = get_M_valid_points(M)
    for i in range(M_valid_pts.shape[0]):
        for j in range(i + 1, M_valid_pts.shape[0]):
            num_shared_points = torch.logical_and(M_valid_pts[i], M_valid_pts[j]).sum()
            view_graph_adj[i, j] = num_shared_points
            view_graph_adj[j, i] = num_shared_points

    return view_graph_adj


def check_if_M_connected(M, thr=1, return_largest_component=False, returnAll=False):
    """
    Check connectivity of the camera-point view graph derived from the visibility matrix M.

    Args:
        M (Tensor): [2m, n] binary visibility matrix.
        thr (int): Minimum number of shared points to consider a connection between views.
        return_largest_component (bool): If True, return the largest connected component.
        returnAll (bool): If True, return all connected components.

    Returns:
        bool or (bool, List[int]) or List[Set[int]] depending on flags:
            - If no flags: returns is_connected (bool)
            - If return_largest_component: returns (is_connected, largest_component)
            - If returnAll: returns list of all components
    """
    import networkx as nx
    import numpy as np

    # Get adjacency matrix of M
    view_graph_adj = get_M_view_adjacency(M)
    view_graph_adj = view_graph_adj.detach().cpu().numpy()

    # Create binary adjacency graph based on threshold
    view_graph = nx.from_numpy_array((view_graph_adj >= thr).astype(int))

    # Check overall connectivity
    connected = nx.is_connected(view_graph)

    # Extract connected and biconnected components
    components = sorted(nx.connected_components(view_graph), key=len, reverse=True)
    #biconnected_components = sorted(nx.biconnected_components(view_graph), key=len, reverse=True)

    # print(f"Component sizes (sorted): {[len(comp) for comp in components]}")
    # print(f"The graph has {len(components)} components")


    if returnAll:
        return components
    if return_largest_component:
        largest_cc = components[0] if components else []
        return connected, list(largest_cc)

    return connected

class AxialAggregationGraphWrapper():
    """
    Wrapper class for defining a graph corresponding to row-wise / column-wise aggregation of sparse matrix elements.
    """
    def __init__(
        self,
        m,
        n,
        agg_dim, # Determines along which dimension aggregation is carried out.
        valid_indices = None, # (2, n_valid_mat_elements) index matrix annotating the positions of valid matrix elements.
        device = None,
    ):
        if device is not None and valid_indices is not None:
            assert valid_indices.device == device
        if valid_indices is not None:
            self.device = valid_indices.device
        elif device is not None:
            self.device = device
        else:
            assert False
        self.m, self.n = m, n
        assert agg_dim in [0, 1]
        self.agg_dim = agg_dim
        self.non_agg_dim = {0: 1, 1: 0}[self.agg_dim] # The remaining dimension after aggregation
        self.n_agg_nodes = {0: m, 1: n}[self.non_agg_dim] # Determine the number of aggregation nodes from the size of the remaining dimension

        self.valid_indices = valid_indices
        if self.valid_indices is None:
            self.dense = True
            # If indices are not given, we assume dense connectivity.
            row_indices = torch.arange(m, dtype=torch.int64, device=self.device)
            col_indices = torch.arange(n, dtype=torch.int64, device=self.device)
            self.valid_indices = torch.cartesian_prod(row_indices, col_indices).T
            # Verify that the order is consistent with a coalesced sparse representation of a dense matrix, since this is what will determine the order during run-time:
            assert torch.all(self.valid_indices == torch.ones((m, n)).to_sparse_coo().indices())
        else:
            self.dense = False
        assert self.valid_indices.device == self.device
        assert self.valid_indices.dtype == torch.int64
        assert len(self.valid_indices.shape) == 2
        n_valid_mat_elements = self.valid_indices.shape[1]
        assert self.valid_indices.shape == (2, n_valid_mat_elements)

        # Define graph for the axial aggregation.
        self.edge_index = self.create_sparse_axial_aggregation_edges()

    # def create_sparse_axial_aggregation_graph(
    def create_sparse_axial_aggregation_edges(
        self,
        # m,
        # n,
    ):
        """
        Given a sparse feature matrix, create a graph aggregating every row or column into one node.
        The nodes consist of one node for every valid matrix element (referred to as a "matrix element nodes"), and one node for every row / column, referred to as "aggregation nodes".
        Every matrix element node is a source node, connected to the aggregation node of its corresponding row / column, consequently making up the target nodes.
        """
        # m, n = self.m, self.n
        n_valid_mat_elements = self.valid_indices.shape[1]

        # Initialize indices for aggregation nodes and matrix element nodes
        agg_node_indices = torch.arange(0, self.n_agg_nodes, dtype=torch.int64, device=self.device) # While the ordering of these indices is arbitrary, it determines the order of the output aggregation node features.
        mat_element_node_indices = torch.arange(0, n_valid_mat_elements, dtype=torch.int64, device=self.device) # While the choice of indices is arbitrary, the ordering is assumed to correspond to the elements in x_mat_elements = M[self.valid_indices[0, :], self.valid_indices[1, :], :] in self.generate_node_features().

        # Adjust aggregation node indices for the new shifted positions, resulting from concatenating source node features with (dummy) target node features in self.generate_node_features().
        agg_node_indices = agg_node_indices + n_valid_mat_elements

        # Define source and target node indices
        source_node_indices = mat_element_node_indices # Every matrix element is a source node.
        target_node_indices = agg_node_indices[self.valid_indices[self.non_agg_dim, :]] # The aggregation nodes to connect to are determined by the corresponding row / column index.

        edge_index = torch.cat([source_node_indices[None, :], target_node_indices[None, :]], dim = 0) # Graph edges, dtype long, shape (2, n_edges), where n_edges = n_valid_mat_elements

        return edge_index

    def generate_node_features(
        self,
        M, # (m, n, n_feat) feature matrix. For a sparse graph, we expect a torch hybrid sparse COO feature matrix, where the last dimension is dense.
        x_agg = None, # Optional features for aggregation nodes (acting as query vectors). Shape (n_agg_nodes, n_feat), where n_agg_nodes is either m or n, depending on along which dimension we are aggregating.
    ):
        """
        Given a sparse feature matrix, generate source nodes from its data, as well as (dummy) target nodes, on which we will apply the graph propagation.
        """
        assert M.device == self.device
        m, n, n_feat = M.shape
        assert (m, n) == (self.m, self.n)
        n_valid_mat_elements = m * n if self.dense else self.valid_indices.shape[1]

        if not self.dense:
            assert M.is_sparse
            assert M.sparse_dim() == 2
            assert M.dense_dim() == 1
            assert M.indices().shape[1] == n_valid_mat_elements
            assert torch.all(self.valid_indices == M.indices())

        # Extract matrix element node features from M
        if self.dense:
            x_mat_elements = M.reshape((m * n, n_feat)) # (n_valid_mat_elements, n_feat)
        else:
            x_mat_elements = M.values() # (n_valid_mat_elements, n_feat)
        assert x_mat_elements.shape == (n_valid_mat_elements, n_feat)

        if x_agg is not None:
            assert not x_agg.is_sparse
            assert x_agg.shape == (self.n_agg_nodes, n_feat)
        else:
            # Dummy zero features for the aggregation nodes
            x_agg = torch.zeros((self.n_agg_nodes, n_feat), dtype=torch.float32, device=self.device)

        # Concatenate source node features with (dummy) target node features
        x = torch.cat((x_mat_elements, x_agg), dim=0) # Projection features, concatenated with dummy aggregation features, shape (n_valid_mat_elements + n_agg_nodes, n_feat)

        return x

    def extract_target_node_features(self, x):
        """
        Given all node features, extract the target features only.
        Input: x, node features, shape (n_valid_mat_elements + n_agg_nodes, n_feat)
        Returns: M_agg_vec, target node features, shape (m, 1, n_feat) or (1, n, n_feat), depending on along which dimension we perform axial aggregation.
        """
        if self.agg_dim == 0:
            assert self.n_agg_nodes == self.n
            M_agg_vec = x[None, -self.n:, :]
        else:
            assert self.n_agg_nodes == self.m
            M_agg_vec = x[-self.m:, None, :]
        return M_agg_vec

    def to(self, device, **kwargs):
        ret = copy.copy(self)
        ret.device = device
        ret.valid_indices = ret.valid_indices.to(device, **kwargs)
        ret.edge_index = ret.edge_index.to(device, **kwargs)
        return ret

