import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .epipolar_line import get_epipolar_line
from .attention import Transformer


def parse_camera(cameras):
    H = cameras[:, 0]
    W = cameras[:, 1]
    intrinsics = cameras[:, 2:18].reshape((-1, 4, 4))
    c2w = cameras[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def normalize(pixel_locations, h=480, w=640):
    resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, None, :]
    normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
    return normalized_pixel_locations


def drop_diagonal(x):
    n, shape = x.shape[0], x.shape[2:]
    x = x.reshape(n*n, *shape)
    x = x[:-1].reshape(n-1, n+1, *shape)[:, 1:]
    x = x.reshape(n, n-1, *shape)
    return x


class EpipolarLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.pos_encoding = self.posenc(d_hid=32, n_samples=cfg.crossview.n_sample)
        self.attention = Transformer(32, 32, cfg.crossview.epipolar.dropout, cfg.crossview.epipolar.dropout)
    
    def posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda").float().unsqueeze(0)
        return sinusoid_table.unsqueeze(dim=0)

    def forward(self, feats_q, feats_k, mask):
        return self.attention(feats_q, feats_k, self.pos_encoding)


class ViewLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.attention = Transformer(32, 32, cfg.crossview.epipolar.dropout, cfg.crossview.epipolar.dropout)

    def forward(self, feats_q, feats_k, mask):
        out = self.attention(feats_q.squeeze(dim=2), feats_k, 0, mask.unsqueeze(dim=-1))
        return out


class CrossViewLayer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.epipolar_layer = EpipolarLayer(cfg)
        self.view_layer = ViewLayer(cfg)

        self.conv_in = nn.Conv2d(128, 32, 1)
        self.conv_out = nn.Conv2d(32, 128, 1)

    def sample_feat(self, feats, epipolar_line):
        n_view, n_pixel, _, n_point = epipolar_line.shape[:4]
        epipolar_line = epipolar_line.permute(2, 0, 1, 3, 4)
        epipolar_line = epipolar_line.reshape(n_view, n_view*n_pixel, n_point, 2)

        # sample feats
        normalized_pixel_locations = normalize(epipolar_line)
        sample_feat =  F.grid_sample(feats, normalized_pixel_locations, align_corners=True)
        sample_feat = sample_feat.view(n_view, -1, n_view, n_pixel, n_point).permute(2, 0, 3, 4, 1)

        # drop diagonal
        sample_feat = drop_diagonal(sample_feat)

        # [n_view, n_pixel, n_view-1, n_point, dim]
        sample_feat = sample_feat.permute(0, 2, 1, 3, 4)
        return sample_feat

    def aggregate(self, feats_q, feats_k, mask):
        n_view, n_pixel, _, n_point, dim = feats_k.shape
        _, _, h_feats, w_feats = feats_q.shape
        feats_q = feats_q.permute(0, 2, 3, 1).reshape(n_view*n_pixel, 1, 1, dim)
        feats_k = feats_k.reshape(n_view*n_pixel, n_view-1, n_point, dim)
        mask = drop_diagonal(mask.permute(0, 2, 1)).permute(0, 2, 1).reshape(n_view*n_pixel, n_view-1)

        # epipolar aggregate
        feats_aggre = self.epipolar_layer(feats_q, feats_k, mask)
        # view aggregate
        feats_aggre = self.view_layer(feats_q, feats_aggre, mask)

        return feats_aggre.view(n_view, h_feats, w_feats, dim).permute(0, 3, 1, 2)
    
    def forward(self, feats, epipolar_line, mask):
        feats = self.conv_in(feats)
        # sample feature
        sample_feat = self.sample_feat(feats, epipolar_line)
        # aggregate
        aggre_feat = self.aggregate(feats, sample_feat, mask)
        aggre_feat = self.conv_out(aggre_feat)
        return aggre_feat


class CrossView(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        layer = CrossViewLayer(cfg)
        self.layers = _get_clones(layer, self.cfg.crossview.n_layers)

    def forward(self, feats, cameras):
        # [n_view, n_pixel, n_view, n_sample, 2], [n_view, n_pixel, n_view]
        epipolar_line, mask = get_epipolar_line(cameras[0], feats, 
            self.cfg.crossview.n_sample, self.cfg.crossview.deterministic)
        
        for mod in self.layers:
            feats_ = mod(feats, epipolar_line, mask)
            if self.cfg.crossview.skip_connect:
                feats = feats_ + feats
            else:
                feats = feats_
        
        return feats


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
