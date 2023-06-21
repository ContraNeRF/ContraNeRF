import os
import json

__all__ = ["merge_results",]


def merge_results(cfg, expname, step):
    out_folder = os.path.join(cfg.output_base, cfg.output, expname, "results")
    start_step = "{:06d}".format(step)
    psnr_all = 0.0
    ssim_all = 0.0
    psnr_all_coarse = 0.0
    ssim_all_coarse = 0.0
    results_dict_all = dict()

    for d in os.listdir(out_folder):
        if start_step not in d or not os.path.exists(os.path.join(out_folder, d, 'metric.txt')):
            continue
        file_path = os.path.join(out_folder, d, 'metric.txt')
        f = open(file_path, "r")
        results_dict = json.loads(f.read())
        
        results_dict_all[d.split('_')[0]] = results_dict
        psnr_all += results_dict['mean_psnr']
        ssim_all += results_dict['mean_ssim']
        psnr_all_coarse += results_dict['mean_psnr_coarse']
        ssim_all_coarse += results_dict['mean_ssim_coarse']

    total_scene_num = len(results_dict_all)
    results_dict_all['psnr_all'] = psnr_all / total_scene_num
    results_dict_all['ssim_all'] = ssim_all / total_scene_num
    results_dict_all['psnr_all_coarse'] = psnr_all_coarse / total_scene_num
    results_dict_all['ssim_all_coarse'] = ssim_all_coarse / total_scene_num
    
    f = open("{}/metric_{:06d}.txt".format(out_folder, int(start_step)), "w")
    f.write(json.dumps(results_dict_all, indent=2))
    f.close()
    return
