# Reference: https://github.com/hiroyasuakada/UnrealEgo

import random
from abc import ABCMeta

import numpy as np
import torch
from torch.utils.data import Dataset


def argmax_2d(heatmaps, presence_prob, use_prob_mask=True):
    """
    Extract 2D coordinates from heatmaps via argmax.

    Args:
        heatmaps: [B, J, H, W]
        presence_prob: [B, J] joint presence probabilities.
        use_prob_mask: If True, zero out coordinates for low-confidence joints.

    Returns:
        coords: [B, J, 2] with (x, y) coordinates.
    """
    B, J, H, W = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, J, -1)
    max_vals, max_idx = heatmaps_flat.max(dim=2)

    coords_y = max_idx // W
    coords_x = max_idx % W
    coords = torch.stack((coords_x, coords_y), dim=2).float()

    if use_prob_mask:
        mask = (presence_prob > 0.5).unsqueeze(2)
        coords = coords * mask

    return coords


class EgoPoseVR3DPoseDataset(Dataset, metaclass=ABCMeta):

    def __init__(
        self,
        info_json,
        num_frames,
        depth=False,
        pre_shuffle=False,
        **kwargs
    ):
        super(EgoPoseVR3DPoseDataset, self).__init__()

        self.info_json = info_json
        self.num_frames = num_frames
        self.depth = depth
        self.window_size = 40

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

        input_hmd = frame_data["hmd_position_global_full_gt_list"][frame:frame + self.window_size + 1, ...]
        img = frame_data["input_rgbd"][frame_indices, :3, :, :]
        depthmap = frame_data["input_rgbd"][frame_indices, 3:4, :, :]
        pred_2d = frame_data["pred_2d"][frame:frame + self.window_size + 1, ...]
        pred_3d = frame_data["pred_3d"][frame:frame + self.window_size + 1, ...]
        pelvis_ori_gt = frame_data["body_parms_list"].item()["root_orient"][frame + self.window_size:frame + self.window_size + 1, ...].squeeze(0)
        joint_rot_gt = frame_data["body_parms_list"].item()["pose_body"][frame + self.window_size:frame + self.window_size + 1, ...].squeeze(0)

        # Zero out invalid 3D predictions
        mask = (pred_3d[..., 0] == 0.0) & (pred_3d[..., 1] == 0.0)
        pred_3d[mask] = [0.0, 0.0, 0.0]

        if self.depth:
            joint_2d = pred_3d
        else:
            joint_2d = pred_2d

        ret_data = dict()
        ret_data["L"] = torch.from_numpy(input_hmd).float()
        ret_data["img"] = torch.from_numpy(img).float()
        ret_data["depth_map"] = torch.from_numpy(depthmap).float()
        ret_data["joint_2d"] = torch.from_numpy(joint_2d).float()
        ret_data["gt_global_ori"] = torch.from_numpy(pelvis_ori_gt).float()
        ret_data["gt_joint_rot"] = torch.from_numpy(joint_rot_gt).float()

        return ret_data

    def __len__(self):
        return len(self.frame_dataset)

    def __getitem__(self, idx):
        return self.load_data(idx)
