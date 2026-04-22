# Reference: https://github.com/hiroyasuakada/UnrealEgo

import random
from abc import ABCMeta

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_gaussian_kernel(radius, sigma):
    """Generate a 2D Gaussian kernel."""
    size = radius * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    center = radius
    for i in range(size):
        for j in range(size):
            dist_sq = (i - center) ** 2 + (j - center) ** 2
            kernel[i, j] = np.exp(-dist_sq / (2 * sigma ** 2))
    return kernel


radius = 3
sigma = radius / 3
gaussian_kernel = generate_gaussian_kernel(radius, sigma)
target_heatmap_height = 64
target_heatmap_width = 80
original_height = 256
original_width = 320


def generate_heatmaps(joint_coords):
    """Generate per-joint Gaussian heatmaps from 2D joint coordinates."""
    heatmaps = np.zeros((22, target_heatmap_height, target_heatmap_width), dtype=np.float32)
    for idx, (x, y) in enumerate(joint_coords[:22]):
        heatmap = np.zeros((target_heatmap_height, target_heatmap_width), dtype=np.float32)
        scaled_x = int(x * target_heatmap_width / original_width)
        scaled_y = int(y * target_heatmap_height / original_height)
        if 0 <= scaled_x < target_heatmap_width and 0 <= scaled_y < target_heatmap_height:
            x_min = max(0, scaled_x - radius)
            x_max = min(target_heatmap_width, scaled_x + radius + 1)
            y_min = max(0, scaled_y - radius)
            y_max = min(target_heatmap_height, scaled_y + radius + 1)
            kernel_x_min = max(0, radius - scaled_x)
            kernel_x_max = min(radius * 2 + 1, target_heatmap_width - scaled_x + radius)
            kernel_y_min = max(0, radius - scaled_y)
            kernel_y_max = min(radius * 2 + 1, target_heatmap_height - scaled_y + radius)
            heatmap[y_min:y_max, x_min:x_max] += gaussian_kernel[kernel_y_min:kernel_y_max,
                                                                   kernel_x_min:kernel_x_max]
        heatmaps[idx] = heatmap
    return heatmaps


class EgoPoseVRHeatmapDataset(Dataset, metaclass=ABCMeta):

    def __init__(
        self,
        info_json,
        pre_shuffle=False,
        **kwargs
    ):
        super(EgoPoseVRHeatmapDataset, self).__init__()

        self.info_json = info_json
        self.num_frames = 1
        self.window_size = 0

        self.frame_dataset = self.collect_dataset(self.info_json, pre_shuffle)

    def collect_dataset(self, info_json, pre_shuffle):
        data = []
        with open("." + info_json) as f:
            lines = f.readlines()

        for full_path in lines:
            full_path = full_path.strip()
            if len(full_path) == 0:
                continue
            data.append(full_path)

        if pre_shuffle:
            random.shuffle(data)

        return data

    def load_data(self, idx):
        path = self.frame_dataset[idx]
        frame_data = np.load("../" + path, allow_pickle=True, mmap_mode='r')

        num_frames = frame_data["input_rgbd"].shape[0]
        frame = np.random.randint(num_frames - self.window_size)
        frame_indices = [frame + self.window_size - i * 1 for i in reversed(range(self.num_frames))]

        img = frame_data["input_rgbd"][frame_indices, :3, :, :]
        depthmap = frame_data["input_rgbd"][frame_indices, 3:4, :, :]
        joint_2d_gt = frame_data["gt_joints_relativeCam_2Dpos"][frame + self.window_size:frame + self.window_size + 1, ...].squeeze(0)

        ret_data = dict()
        ret_data["img"] = torch.from_numpy(img).float()
        ret_data["depth_map"] = torch.from_numpy(depthmap).float()
        ret_data["gt_joint_2d"] = torch.from_numpy(joint_2d_gt).float()

        return ret_data

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)
