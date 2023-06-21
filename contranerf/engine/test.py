import os
import json
import imageio
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from contranerf.data import dataset_dict
from contranerf.utils import *


def test_per_scene(cfg, scene_name, dataset_name, model, logger, out_scene_dir):
    test_dataset = dataset_dict[dataset_name](cfg, 'test', scenes=[scene_name])
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = dict()
    sum_psnr = 0
    running_mean_psnr = 0
    sum_ssim = 0
    running_mean_ssim = 0
    sum_psnr_coarse = 0
    running_mean_psnr_coarse = 0
    sum_ssim_coarse = 0
    running_mean_ssim_coarse = 0

    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        out_image_dir = os.path.join(out_scene_dir, file_id)
        os.makedirs(out_image_dir, exist_ok=True)

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_image_dir, 'average.png'),
                        averaged_img)

        # rendering
        ret = model(data)
        
        gt_rgb = data['rgb'][0]
        pred_rgb = ret['rgb']['rgb'].detach().cpu()
        err_map = torch.sum((pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
        err_map_colored = (colorize_np(err_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(out_image_dir, 'err_map.png'),
                        err_map_colored)
        pred_rgb_np = np.clip(pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
        gt_rgb_np = gt_rgb.numpy()[None, ...]
        psnr, ssim = compute_psnr_ssim(pred_rgb_np, gt_rgb_np)

        # saving outputs ...
        pred_rgb = (255 * np.clip(pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_image_dir, 'pred.png'), pred_rgb)

        gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_image_dir, 'gt_rgb.png'), gt_rgb_np_uint8)

        sum_psnr += psnr
        running_mean_psnr = sum_psnr / (i + 1)
        sum_ssim += ssim
        running_mean_ssim = sum_ssim / (i + 1)

        pred_rgb_coarse = ret['rgb_coarse']['rgb'].detach().cpu()
        pred_rgb_np_coarse = np.clip(pred_rgb_coarse.numpy()[None, ...], a_min=0., a_max=1.)
        psnr_coarse, ssim_coarse = compute_psnr_ssim(pred_rgb_np_coarse, gt_rgb_np)
        # saving outputs ...
        pred_rgb_coarse = (255 * np.clip(pred_rgb_coarse.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_image_dir, 'pred_coarse.png'), pred_rgb_coarse)
        sum_psnr_coarse += psnr_coarse
        running_mean_psnr_coarse = sum_psnr_coarse / (i + 1)
        sum_ssim_coarse += ssim_coarse
        running_mean_ssim_coarse = sum_ssim_coarse / (i + 1)

        logger.info("\n==================\n"
                "{}, curr_id: {} \n"
                "current psnr: {:03f}, current coarse psnr: {:03f} \n"
                "running mean psnr: {:03f}, running mean coarse psnr: {:03f} \n"
                "current ssim: {:03f}, current coarse ssim: {:03f} \n"
                "running mean ssim: {:03f}, running mean coarse ssim: {:03f} \n" 
                "===================\n"
                .format(scene_name, file_id,
                        psnr, psnr_coarse,
                        running_mean_psnr, running_mean_psnr_coarse,
                        ssim, ssim_coarse,
                        running_mean_ssim, running_mean_ssim_coarse,
                        ))

        results_dict[file_id] = {'psnr': float(psnr), 'ssim': float(ssim),
                                 'psnr_coarse': float(psnr_coarse),
                                 'ssim_coarse': float(ssim_coarse),}

    mean_psnr = sum_psnr / total_num
    mean_ssim = sum_ssim / total_num
    mean_psnr_coarse = sum_psnr_coarse / total_num
    mean_ssim_coarse = sum_ssim_coarse / total_num

    logger.info('\n------{}-------\n'
          'final psnr: {}, final coarse psnr: {}\n'
          'final ssim: {}, final coarse ssim: {}\n'
          .format(scene_name, mean_psnr, mean_psnr_coarse, mean_ssim, mean_ssim_coarse))

    results_dict['mean_psnr'] = float(mean_psnr)
    results_dict['mean_ssim'] = float(mean_ssim)
    results_dict['mean_psnr_coarse'] = float(mean_psnr_coarse)
    results_dict['mean_ssim_coarse'] = float(mean_ssim_coarse)

    return results_dict


def test(args, cfg, model, scene_names, dataset_name, logger, expname, step):
    out_folder = os.path.join(cfg.output_base, cfg.output, expname, "results")
    os.makedirs(out_folder, exist_ok=True)
    if args.distributed:
        world_size = dist.get_world_size()
    else:
        world_size = 1
    device = torch.device('cuda:{}'.format(args.local_rank))

    for scene_name in scene_names[args.local_rank::world_size]:
        out_scene_dir = os.path.join(out_folder, '{}_{:06d}'.format(scene_name, step))
        results_dict = test_per_scene(
            cfg, scene_name, dataset_name, model, logger, out_scene_dir, device)
        f = open("{}/metric.txt".format(out_scene_dir), "w")
        f.write(json.dumps(results_dict, indent=2))
        f.close()
    
    synchronize()
    return
