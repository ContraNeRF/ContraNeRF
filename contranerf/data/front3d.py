import os
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset

from contranerf.utils import *


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


def fix_pose(pose):
    R_bcam2cv = np.array([[1, 0,  0], [0, -1, 0], [0, 0, -1]])
    location, rotation = pose[:3, 3], pose[:3, :3]
    R_world2bcam = rotation.transpose()
    T_world2bcam = -1 * R_world2bcam @ location

    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    RT_world2cv = np.concatenate(
        [R_world2cv, np.expand_dims(T_world2cv, axis=-1)], axis=-1)
    RT_world2cv = np.concatenate(
        [RT_world2cv, np.array([[0, 0, 0, 1]])], axis=0)
    
    c2w = np.linalg.inv(RT_world2cv)
    return c2w


class Front3DDataset(Dataset):
    def __init__(self, cfg, mode):
        self.folder_path = cfg.data.rootdir_front3d
        self.mode = mode
        self.num_source_views = cfg.data.num_source_views
        self.camera_std = cfg.data.camera_std

        self.scene_infos = dict()
        self.render_rgb_list = list()
        self.render_pose_list = list()
        self.scene_list = list()
        for scene_name in open(os.path.join(self.folder_path, 'train_ids.txt')):
            scene_name = scene_name.strip('\n')
            scene_path = os.path.join(self.folder_path, scene_name)
            color_path = os.path.join(scene_path, 'color')
            pose_path = os.path.join(scene_path, 'pose')

            rgb_path_list = [os.path.join(color_path, rgb_name) 
                             for rgb_name in sort_by_file_name(os.listdir(color_path))]
            pose_list = [fix_pose(np.loadtxt(os.path.join(pose_path, pose_name))) 
                         for pose_name in sort_by_file_name(os.listdir(pose_path))]
            pose_list = self.add_camera_noise(pose_list)
            intrinsic = np.loadtxt(os.path.join(scene_path, 'intrinsic.txt'))
            assert len(rgb_path_list) == len(pose_list)

            scene_infos = dict(
                rgb_paths=np.array(rgb_path_list),
                poses=np.array(pose_list),
                intrinsic=intrinsic,
            )
            self.scene_infos[scene_name] = scene_infos
            self.render_rgb_list.extend(rgb_path_list)
            self.render_pose_list.extend(pose_list)
            self.scene_list.extend([scene_name] * len(rgb_path_list))

    def __len__(self):
        return len(self.render_rgb_list)

    def add_camera_noise(self, pose_list):
        se3_noise = np.random.normal(0, self.camera_std, (len(pose_list), 6))
        pose_noise = se3_to_SE3(torch.tensor(se3_noise))
        pose = torch.tensor(pose_list)
        pose = compose_pair(pose, pose_noise)
        pose_list = [pose[i].numpy() for i in range(len(pose))]
        return pose_list

    def __getitem__(self, index):
        # read image
        rgb_path = self.render_rgb_list[index]
        rgb = imageio.imread(rgb_path).astype(np.float32) / 255.
        img_size = rgb.shape[:2]

        # read camera
        render_pose = self.render_pose_list[index]
        scene_info = self.scene_infos[self.scene_list[index]]
        intrinsic = scene_info['intrinsic']
        camera = np.concatenate((list(img_size), intrinsic.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        # sample support ids
        subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
        num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        nearest_pose_ids = select_support_ids(render_pose,
                                              scene_info['poses'],
                                              min(self.num_source_views*subsample_factor, 22))
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        # read support images and cameras
        rgb_supports_ori = list()
        rgb_supports = list()
        camera_supports = list()
        for i in nearest_pose_ids:
            rgb_support_ori = imageio.imread(scene_info['rgb_paths'][i]).astype(np.float32) / 255.
            rgb_support = rgb_support_ori
            pose_support = scene_info['poses'][i]
            camera_support = np.concatenate((list(img_size), intrinsic.flatten(),
                                             pose_support.flatten())).astype(np.float32)
            rgb_supports_ori.append(rgb_support_ori)
            rgb_supports.append(rgb_support)
            camera_supports.append(camera_support)
        
        rgb_supports_ori = np.stack(rgb_supports_ori, axis=0)
        rgb_supports = np.stack(rgb_supports, axis=0)
        camera_supports = np.stack(camera_supports, axis=0)

        ret = {
            'rgb': torch.from_numpy(rgb[..., :3]),
            'camera': torch.from_numpy(camera),
            'rgb_path': rgb_path,
            'src_rgbs_ori': torch.from_numpy(rgb_supports_ori[..., :3]),
            'src_rgbs': torch.from_numpy(rgb_supports[..., :3]),
            'src_cameras': torch.from_numpy(camera_supports),
            'depth_range': torch.tensor([0.1, 10.0]),
        }
        
        return ret
