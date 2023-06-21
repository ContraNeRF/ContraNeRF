import os
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset


def sort_by_file_name(file_names):
    file_names = list(file_names)
    file_names.sort(key=lambda name: int(name.split('.')[0]))
    return file_names


def rot_matrix_angular_dist(R1, R2):
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + 1e-6, a_max=1 - 1e-6))


def select_support_ids(tar_pose, ref_poses, num_select, dist_weight=0.5, drop_first=True):
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams - 1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    angular_dists = rot_matrix_angular_dist(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    dists = np.linalg.norm(batched_tar_pose[:, :3, 3] - ref_poses[:, :3, 3], axis=1)
    dists = dist_weight * dists + (1 - dist_weight) * angular_dists

    sorted_ids = np.argsort(dists)
    if drop_first:
        selected_ids = sorted_ids[1:num_select+1]
    else:
        selected_ids = sorted_ids[:num_select]
    return selected_ids


class ScanNetTestDataset(Dataset):
    def __init__(self, cfg, mode, scenes) -> None:
        self.mode = mode
        self.num_source_views = cfg.data.num_source_views

        scene_path = os.path.join(cfg.data.rootdir_scannet, scenes[0])
        color_path = os.path.join(scene_path, 'color')
        pose_path = os.path.join(scene_path, 'pose')

        rgb_path_list = [os.path.join(color_path, rgb_name) 
                         for rgb_name in sort_by_file_name(os.listdir(color_path))]
        pose_list = [np.loadtxt(os.path.join(pose_path, pose_name)) 
                     for pose_name in sort_by_file_name(os.listdir(pose_path))]  
        intrinsic = np.loadtxt(os.path.join(scene_path, 'intrinsic.txt'))
        assert len(rgb_path_list) == len(pose_list)

        i_test = np.arange(len(pose_list))[::cfg.data.testskip]
        i_train = np.array([j for j in np.arange(len(pose_list)) if
                            (j not in i_test and j not in i_test)])
        i_render = i_train if mode == 'train' else i_test

        self.intrinsic = intrinsic
        self.train_pose_list = np.array(pose_list)[i_train].tolist()
        self.train_rgb_list = np.array(rgb_path_list)[i_train].tolist()
        self.render_pose_list = np.array(pose_list)[i_render].tolist()
        self.render_rgb_list = np.array(rgb_path_list)[i_render].tolist()

    def __len__(self):
        return len(self.render_pose_list)

    def __getitem__(self, index):
        # read image
        rgb_path = self.render_rgb_list[index]
        rgb = imageio.imread(rgb_path).astype(np.float32) / 255.
        img_size = rgb.shape[:2]

        # read camera
        render_pose = np.array(self.render_pose_list[index])
        camera = np.concatenate((list(img_size), self.intrinsic.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # sample support ids
        if self.mode == 'train':
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
            nearest_pose_ids = select_support_ids(render_pose,
                                                  np.array(self.train_pose_list),
                                                  min(self.num_source_views*subsample_factor, 22))
            nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)
        else:
            nearest_pose_ids = select_support_ids(render_pose,
                                                  np.array(self.train_pose_list),
                                                  min(self.num_source_views, 22),
                                                  drop_first=False)
            nearest_pose_ids = np.random.choice(
                nearest_pose_ids, min(self.num_source_views, len(nearest_pose_ids)), replace=False)  

        # read support images and cameras
        rgb_supports = list()
        camera_supports = list()
        for i in nearest_pose_ids:
            rgb_support = imageio.imread(self.train_rgb_list[i]).astype(np.float32) / 255.
            pose_support = np.array(self.train_pose_list[i])
            camera_support = np.concatenate((list(img_size), self.intrinsic.flatten(),
                                             pose_support.flatten())).astype(np.float32)
            rgb_supports.append(rgb_support)
            camera_supports.append(camera_support)
        
        rgb_supports = np.stack(rgb_supports, axis=0)
        camera_supports = np.stack(camera_supports, axis=0)

        ret = {
            'rgb': torch.from_numpy(rgb[..., :3]),
            'camera': torch.from_numpy(camera),
            'rgb_path': rgb_path,
            'id': torch.tensor(index),
            'src_rgbs': torch.from_numpy(rgb_supports[..., :3]),
            'src_cameras': torch.from_numpy(camera_supports),
            'src_ids': torch.from_numpy(nearest_pose_ids),
            'depth_range': torch.tensor([0.1, 10.0]),
        }
        
        return ret
