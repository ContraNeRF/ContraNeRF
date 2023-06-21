import torch
from collections import OrderedDict

__all__ = ["render_single_image"]


def render_single_image(ray_sampler,
                        ray_batch,
                        model,
                        chunk_size,
                        render_stride=1,
                        featmaps=None):
    all_ret = {
        'rgb': {'rgb': [], 'mask': None},
        'rgb_coarse': {'rgb': [], 'mask': None},
    }

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        if featmaps is None:
            ret = model._forward(chunk)
        else:
            ret = model._forward(chunk, featmaps)

        all_ret['rgb']['rgb'].append(ret['rgb']['rgb'].cpu())
        all_ret['rgb_coarse']['rgb'].append(ret['rgb_coarse']['rgb'].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]

    # merge chunk results and reshape
    for k in ['rgb', 'rgb_coarse']:
        tmp = torch.cat(all_ret[k]['rgb'], dim=0).reshape((rgb_strided.shape[0],
                                                           rgb_strided.shape[1], -1))
        all_ret[k]['rgb'] = tmp.squeeze()

    return all_ret
