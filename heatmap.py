"""
Offline heatmap inference script for preprocessing.
Runs RGB and RGBD heatmap models on stored data and saves predicted 2D/3D keypoints.
"""

from pose_estimation.models.estimator import RGBDPoserHeatmap
import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm

window_size = 40
refinement_window_size = 1

# --- RGB model ---
rgb_model_cfg = {
    "num_heatmap": 22,
    "encoder_cfg": {
        "resnet_cfg": {
            "model_name": "resnet18",
            "out_stride": 4,
            "use_imagenet_pretrain": True
        },
        "neck_cfg": {
            "in_channels": [64, 128, 256, 512],
            "out_channels": 128
        },
        "depth": False
    }
}

rgb_model_path = "ckpt/pretrained_heatmap/rgb_epoch=49.ckpt"
rgb_model = RGBDPoserHeatmap(**rgb_model_cfg)
ckpt = torch.load(rgb_model_path, map_location="cpu", weights_only=True)
checkpoint_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
rgb_model.load_state_dict(checkpoint_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rgb_model = rgb_model.to(device)
rgb_model.training = False
rgb_model.eval()

# --- RGBD model ---
rgbd_model_cfg = {
    "num_heatmap": 22,
    "encoder_cfg": {
        "resnet_cfg": {
            "model_name": "resnet18",
            "out_stride": 4,
            "use_imagenet_pretrain": True
        },
        "neck_cfg": {
            "in_channels": [64, 128, 256, 512],
            "out_channels": 128
        },
        "depth": True
    }
}

rgbd_model_path = "ckpt/pretrained_heatmap/rgbd_epoch=49.ckpt"
rgbd_model = RGBDPoserHeatmap(**rgbd_model_cfg)
ckpt = torch.load(rgbd_model_path, map_location="cpu", weights_only=True)
checkpoint_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
rgbd_model.load_state_dict(checkpoint_dict)

rgbd_model = rgbd_model.to(device)
rgbd_model.training = False
rgbd_model.eval()

# --- Process all npz files ---
folder_path = Path('UnityTest/')
npz_files = list(folder_path.rglob('*.npz'))

count = 0
total = len(npz_files)

for file_path in tqdm(sorted(npz_files), desc="Processing frames", unit="file"):
    count += 1
    frame_data = np.load(file_path, allow_pickle=True)
    frame_data_dict = {k: frame_data[k] for k in frame_data}

    img = frame_data["input_rgbd"][:, :3, :, :]
    depthmap = frame_data["input_rgbd"][:, 3:4, :, :]
    joint_2d_gt = frame_data["gt_joints_relativeCam_2Dpos"][:, ...]

    ret_data = dict()
    ret_data["img"] = torch.from_numpy(img).float().unsqueeze(1)
    ret_data["depth_map"] = torch.from_numpy(depthmap).float().unsqueeze(1)
    ret_data["gt_joint_2d"] = torch.from_numpy(joint_2d_gt).float()

    for k in ret_data:
        ret_data[k] = ret_data[k].to(device)

    rgb_output = rgb_model(ret_data["img"], ret_data["depth_map"], ret_data["gt_joint_2d"])
    rgbd_output = rgbd_model(ret_data["img"], ret_data["depth_map"], ret_data["gt_joint_2d"])

    print(
        f"Frame {count}/{total} - "
        f"[RGB] MPJPE: [{rgb_output['mpjpe_2d'].item():.3f}], "
        f"Acc: [{rgb_output['accuracy'].item():.3f}], "
        f"[RGBD] MPJPE: [{rgbd_output['mpjpe_2d'].item():.3f}], "
        f"Acc: [{rgbd_output['accuracy'].item():.3f}]"
    )

    # Compute 3D keypoints from RGBD predictions
    pred_2d = rgbd_output["pred_2d"]

    x_pix = (pred_2d[..., 0] * 4).astype(np.int32)
    y_pix = (pred_2d[..., 1] * 4).astype(np.int32)

    T, J = x_pix.shape
    t_idx = np.arange(T)[:, None]

    depthmap = depthmap.squeeze(1)
    depth_values = depthmap[t_idx, y_pix, x_pix]
    depth_values = np.expand_dims(depth_values, axis=-1)
    pred_3d = np.concatenate([pred_2d, depth_values], axis=-1)

    width = 80
    height = 64
    pred_3d = pred_3d.astype(np.float32)
    pred_3d[..., 0] /= width
    pred_3d[..., 1] /= height

    # Compute 2D keypoints from RGB predictions
    pred_2d = rgb_output["pred_2d"]
    pred_2d = pred_2d.astype(np.float32)
    pred_2d[..., 0] /= width
    pred_2d[..., 1] /= height

    frame_data_dict["pred_2d"] = pred_2d
    frame_data_dict["pred_3d"] = pred_3d

    np.savez_compressed(file_path, **frame_data_dict)
