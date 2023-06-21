import numpy as np
import torch
import torch.nn as nn

__all__ = ["mse2psnr", "img2mse", "Criterion"]

TINY_NUMBER = 1e-6

mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


def img2mse(x, y, mask=None):
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        mask = mask.float()
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch):
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask']
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)
        return loss


def calculate_mean_variance(x, weight, mask):
    n_ray, n_view = x.shape[0], x.shape[2]

    weight = weight.view([n_ray, -1, 1, 1])
    weight = weight * mask
    weight = weight / (torch.sum(weight, dim=1, keepdim=True) + TINY_NUMBER)

    mean = torch.sum(x * weight, dim=1, keepdim=True)
    var = torch.sum(weight * torch.sum((x - mean)**2, dim=3, keepdim=True), dim=1, keepdim=True)
    return mean.view(n_ray, n_view, 2), var.view(n_ray, n_view)
