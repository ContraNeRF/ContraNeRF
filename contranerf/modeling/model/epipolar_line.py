import numpy as np
import torch


def parse_camera(cameras):
    H = cameras[:, 0]
    W = cameras[:, 1]
    intrinsics = cameras[:, 2:18].reshape((-1, 4, 4))
    c2w = cameras[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def get_rays_single_image(H, W, intrinsics, c2w, render_stride=16):
    batch_size = c2w.shape[0]

    u, v = np.meshgrid(np.arange(W)[::render_stride], np.arange(H)[::render_stride])
    u = u.reshape(-1).astype(dtype=np.float32) + render_stride / 2
    v = v.reshape(-1).astype(dtype=np.float32) + render_stride / 2
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
    pixels = torch.from_numpy(pixels)
    batched_pixels = pixels.unsqueeze(0).repeat(batch_size, 1, 1).to(intrinsics.device)

    rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
    rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[1], 1)  # B x HW x 3
    return rays_o, rays_d


def compute_projections(xyz, cameras):
    original_shape = xyz.shape[:-1]
    xyz = xyz.reshape(-1, 3)
    num_views = len(cameras)
    train_intrinsics = cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
    train_poses = cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
    projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
        .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
    projections = projections.permute(2, 0, 1)  # [n_views, n_points, 4]
    pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
    return pixel_locations.reshape(original_shape + (num_views, ) + (2, )).permute(0, 1, 3, 2, 4)


def calc_intersections(projections, H, W):
    H -= 1
    W -= 1
    K = (projections[..., 0, 0] - projections[..., 1, 0]) / (projections[..., 0, 1] - projections[..., 1, 1] + 1e-9)
    K_inv = 1 / (K + 1e-9)
    A = projections[..., 0, 0] - K * projections[..., 0, 1]
    B = A + K * H
    C = projections[..., 0, 1] - K_inv * projections[..., 0, 0]
    D = C + K_inv * W

    out = torch.zeros_like(projections)
    count = torch.zeros_like(A)

    mask_A = (A >= 0) & (A <= W)
    out[..., 0, :][mask_A] = torch.stack([A, torch.zeros_like(A)], dim=-1)[mask_A]
    count[mask_A] += 1

    mask_B = (B >= 0) & (B <= W)
    out[..., 0, :][mask_B & (count==0)] = torch.stack([B, H*torch.ones_like(B)], dim=-1)[mask_B & (count==0)]
    out[..., 1, :][mask_B & (count==1)] = torch.stack([B, H*torch.ones_like(B)], dim=-1)[mask_B & (count==1)]
    count[mask_B] += 1

    mask_C = (C >= 0) & (C <= H)
    out[..., 0, :][mask_C & (count==0)] = torch.stack([torch.zeros_like(C), C], dim=-1)[mask_C & (count==0)]
    out[..., 1, :][mask_C & (count==1)] = torch.stack([torch.zeros_like(C), C], dim=-1)[mask_C & (count==1)]
    count[mask_C] += 1

    mask_D = (D >= 0) & (D <= H)
    out[..., 0, :][mask_D & (count==0)] = torch.stack([W*torch.ones_like(D), D], dim=-1)[mask_D & (count==0)]
    out[..., 1, :][mask_D & (count==1)] = torch.stack([W*torch.ones_like(D), D], dim=-1)[mask_D & (count==1)]
    count[mask_D] += 1
    
    return out, count


def sampling(intersections, n_sample, deterministic):
    samples = torch.arange(0, n_sample) / (n_sample-1)
    samples = samples.view(1, 1, 1, n_sample, 1).to(intersections.device)
    start, end = intersections[..., :1, :], intersections[..., 1:, :]
    samples = start + (end - start) * samples

    if not deterministic:
        raise NotImplementedError
    return samples


def get_epipolar_line(cameras, feats, n_sample=16, deterministic=True):
    W, H, intrinsics, c2w = parse_camera(cameras)
    render_stride = int(H[0] / feats.shape[2])
    rays_o, rays_d = get_rays_single_image(int(H[0]), int(W[0]), intrinsics, c2w, render_stride)
    # [n_view, n_pixel, n_view, n_point, 2]
    projections = compute_projections(torch.stack([rays_o+rays_d, rays_o+3*rays_d], dim=-2), cameras)

    intersections, count = calc_intersections(projections, int(H[0]), int(W[0]))
    samples = sampling(intersections, n_sample, deterministic)

    return samples, count >= 2
