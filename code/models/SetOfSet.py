import torch
from torch import nn
from utils import dataset_utils
from models.baseNet import BaseNet
from models.layers import *
from utils.sparse_utils import SparseMat
from datasets.SceneData import SceneData
from utils import general_utils
from utils.Phases import Phases

class SetOfSetBlock(nn.Module):
    def __init__(self, d_in, d_out, conf):
        super(SetOfSetBlock, self).__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")
        type_name = conf.get_string("model.layer_type", default='SetOfSetLayer')
        self.layer_type = general_utils.get_class("models.layers." + type_name)
        self.layer_kwargs = dict(conf['model'].get('layer_extra_params', {}))

        modules = []
        modules.extend([self.layer_type(d_in, d_out, **self.layer_kwargs), NormalizationLayer()])
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), self.layer_type(d_out, d_out, **self.layer_kwargs), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)

        self.final_act = ActivationLayer()

        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())

    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        if self.use_skip:
            xl = self.skip(x) + xl

        out = self.final_act(xl)
        return out


class SetOfSetOutliersNet(BaseNet):
    def __init__(self, conf, phase=None):
        super(SetOfSetOutliersNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        # self.pairwise_mlp = get_linear_layers([4, 64, num_feats], final_layer=True, batchnorm=False)
        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)
        self.outlier_net = get_linear_layers([num_feats] * 2 + [1], final_layer=True, batchnorm=False)
        if phase is Phases.FINE_TUNE:
            self.mode = 1
        else:
            self.mode = conf.get_int('train.output_mode', default=3)

        if self.mode == 2:
            for param in self.m_net.parameters():
                param.requires_grad = False
            self.m_net.eval()
            for param in self.n_net.parameters():
                param.requires_grad = False
            self.n_net.eval()

        if self.mode == 1:
            for param in self.outlier_net.parameters():
                param.requires_grad = False
            self.outlier_net.eval()


    def forward(self, data: SceneData):

        x: SparseMat = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        if self.mode != 1:
            # outliers predictions
            outliers_out = self.outlier_net(x.values)
            outliers_out = torch.sigmoid(outliers_out)
        else:
            outliers_out = None

        if self.mode != 2:
            # Cameras predictions
            m_input = x.mean(dim=1) # [m,d_out]
            
            # Process pairwise epipoles
            # pairwise_features = self.pairwise_mlp(data.pairwise_epipoles).mean(dim=1)

            # Concatenate with existing camera features
            # m_input = torch.cat([m_input, pairwise_features], dim=1)
            m_out = self.m_net(m_input)  # [m, d_m]

            # Points predictions
            n_input = x.mean(dim=0) # [n,d_out]
            n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

            # predict extrinsic matrix
            pred_cam = self.extract_model_outputs(m_out, n_out, data)

        else:
            pred_cam = None



        return pred_cam, outliers_out


class SetOfSetNet(BaseNet):
    def __init__(self, conf):
        super(SetOfSetNet, self).__init__(conf)
        # n is the number of points and m is the number of cameras
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')

        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2

        self.embed = EmbeddingLayer(multires, d_in)

        self.equivariant_blocks = torch.nn.ModuleList([SetOfSetBlock(self.embed.d_out, num_feats, conf)])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(SetOfSetBlock(num_feats, num_feats, conf))

        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)

    def forward(self, data: SceneData):
        x = data.x  # x is [m,n,d] sparse matrix
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]

        # Cameras predictions
        m_input = x.mean(dim=1) # [m,d_out]
        m_out = self.m_net(m_input)  # [m, d_m]

        # Points predictions
        n_input = x.mean(dim=0) # [n,d_out]
        n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]

        # predict extrinsic matrix
        pred_cam = self.extract_model_outputs(m_out, n_out, data)

        return pred_cam, None


