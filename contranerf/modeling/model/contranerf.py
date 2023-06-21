import torch
import torch.nn as nn

from .mlp_network import build_mlpnet
from .img_encoder import ImgEncoder
from .projection import Projector
from .utils import *

from contranerf.utils import RaySamplerSingleImage, render_single_image


class ContraNeRF(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.net_coarse = build_mlpnet(cfg,
                                       in_feat_ch=cfg.mlpnet.coarse_feat_dim,
                                       n_samples=cfg.data.num_samples)
        self.net_fine = build_mlpnet(cfg,
                                     in_feat_ch=cfg.mlpnet.fine_feat_dim,
                                     n_samples=cfg.data.num_samples + cfg.data.num_importance)
        # create feature extraction network
        self.feature_net = ImgEncoder(cfg, out_dim=cfg.mlpnet.image_feat_dim)
        # create projector
        self.projector = Projector()

    @property
    def device(self):
        return self.net_coarse.base_fc[0].weight.device

    def forward(self, data):
        if not self.training:
            return self.inference(data)
        
        # load training rays
        ray_sampler = RaySamplerSingleImage(data, self.device)
        N_rand = int(
            1.0 * self.cfg.solver.batch_size * self.cfg.data.num_source_views / data['src_rgbs'][0].shape[0])
        batch = ray_sampler.random_sample(N_rand,
                                            sample_mode=self.cfg.data.sample_mode,
                                            center_ratio=self.cfg.data.center_ratio,)
        return self._forward(batch), batch

    def inference(self, data):
        ray_sampler = RaySamplerSingleImage(data, device=self.device)
        ray_batch = ray_sampler.get_all()

        featmaps = self.feature_net(
            ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2), ray_batch['src_cameras'], 
            None, ray_batch['depth_range'],
        )
        ret = render_single_image(self.cfg,
                                  ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=self,
                                  chunk_size=self.cfg.test.chunk_size,
                                  featmaps=featmaps,)
        return ret

    def _forward(self, batch, featmaps=None):
        ret = {'rgb_coarse': None,
               'rgb': None}
        N_samples = self.cfg.data.num_samples
        N_importance = self.cfg.data.num_importance
        inv_uniform = self.cfg.data.inv_uniform
        det = self.cfg.data.deterministic
        if not self.training:
            det = True

        # extract feature map
        if featmaps is None:
            featmaps = self.feature_net(
                batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2), batch['src_cameras'])

        pts, z_vals = sample_along_camera_ray(ray_o=batch['ray_o'],
                                            ray_d=batch['ray_d'],
                                            depth_range=batch['depth_range'],
                                            N_samples=N_samples, inv_uniform=inv_uniform, det=det)
        _, N_samples = pts.shape[:2]

        rgb_feat, ray_diff, mask, projection = self.projector.compute(pts, batch['camera'],
                                                    batch['src_rgbs'],
                                                    batch['src_cameras'],
                                                    featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x]
        pixel_mask = mask[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
        raw_coarse = self.net_coarse(rgb_feat, ray_diff, mask)   # [N_rays, N_samples, 4]

        outputs_coarse = raw2outputs(raw_coarse, pixel_mask,
                                white_bkgd=self.cfg.mlpnet.white_bkgd)
        ret['rgb_coarse'], weights, alpha = outputs_coarse
        ret['weights_coarse'], ret['alpha_coarse'] = weights, alpha
        ret['mask_coarse'], ret['projection_coarse'] = mask, projection

        if N_importance > 0:
            assert self.net_fine is not None
            # detach since we would like to decouple the coarse and fine networks
            weights = weights.clone().detach()            # [N_rays, N_samples]
            if inv_uniform:
                inv_z_vals = 1. / z_vals
                inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
                weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
                inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                        weights=torch.flip(weights, dims=[1]),
                                        N_samples=N_importance, det=det)  # [N_rays, N_importance]
                z_samples = 1. / inv_z_vals
            else:
                # take mid-points of depth samples
                z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
                weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
                z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                                    N_samples=N_importance, det=det)  # [N_rays, N_importance]

            z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

            # samples are sorted with increasing depth
            z_vals, _ = torch.sort(z_vals, dim=-1)
            N_total_samples = N_samples + N_importance

            viewdirs = batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
            ray_o = batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
            pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]

            rgb_feat_sampled, ray_diff, mask, projection = self.projector.compute(pts, batch['camera'],
                                                                batch['src_rgbs'],
                                                                batch['src_cameras'],
                                                                featmaps=featmaps[1])

            pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
            raw_fine = self.net_fine(rgb_feat_sampled, ray_diff, mask)

            outputs_fine = raw2outputs(raw_fine, pixel_mask,
                                white_bkgd=self.cfg.mlpnet.white_bkgd)
            ret['rgb'], ret['weights'], ret['alpha'] = outputs_fine
            ret['mask'], ret['projection'] = mask, projection

        return ret
