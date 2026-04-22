"""
RGBDPoser Heatmap: 2D joint heatmap prediction module.

Predicts per-joint heatmaps and joint presence probabilities from
egocentric RGB/RGBD images using a ResNet backbone with FPN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pose_estimation.models.backbones.resnet import ResnetBackbone


def argmax_2d(heatmaps, presence_prob, use_prob_mask=True):
    """
    Extract 2D keypoint coordinates from heatmaps via argmax.

    Args:
        heatmaps: [B, J, H, W] predicted heatmaps.
        presence_prob: [B, J] joint presence probabilities.
        use_prob_mask: zero out coordinates for low-probability joints.

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


def generate_gaussian_kernel_torch(radius, sigma, device="cpu"):
    size = radius * 2 + 1
    ax = torch.arange(-radius, radius + 1, device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel


def generate_heatmaps_batched_torch(joint_coords_batch, target_h, target_w, original_h, original_w, radius=3):
    """
    Generate Gaussian heatmaps from 2D joint coordinates.

    Args:
        joint_coords_batch: [B, J, 2] coordinates in original image size.
        target_h, target_w: output heatmap dimensions.
        original_h, original_w: original image dimensions.
        radius: Gaussian kernel radius.

    Returns:
        heatmaps: [B, J, target_h, target_w]
    """
    device = joint_coords_batch.device
    B, J, _ = joint_coords_batch.shape
    sigma = radius / 3

    kernel = generate_gaussian_kernel_torch(radius, sigma, device=device)
    heatmaps = torch.zeros((B, J, target_h, target_w), dtype=torch.float32, device=device)

    scaled_coords = joint_coords_batch.clone()
    scaled_coords[..., 0] = scaled_coords[..., 0] * target_w / original_w
    scaled_coords[..., 1] = scaled_coords[..., 1] * target_h / original_h

    for b in range(B):
        for j in range(J):
            x, y = scaled_coords[b, j]
            x, y = int(x.item()), int(y.item())

            if x < 0 or y < 0 or x >= target_w or y >= target_h:
                continue

            x_min, x_max = max(0, x - radius), min(target_w, x + radius + 1)
            y_min, y_max = max(0, y - radius), min(target_h, y + radius + 1)
            kx_min = max(0, radius - x)
            kx_max = kx_min + (x_max - x_min)
            ky_min = max(0, radius - y)
            ky_max = ky_min + (y_max - y_min)

            heatmaps[b, j, y_min:y_max, x_min:x_max] += kernel[ky_min:ky_max, kx_min:kx_max]

    return heatmaps


class RGBDPoserHeatmap(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        num_heatmap,
        train_cfg=dict(w_heatmap=1.0),
        **kwargs,
    ):
        super(RGBDPoserHeatmap, self).__init__()

        self.num_heatmap = num_heatmap
        self.train_cfg = train_cfg
        self.depth = encoder_cfg["depth"]

        self.encoder = ResnetBackbone(**encoder_cfg)

        out_channels = self.encoder.get_output_channel()
        if self.depth:
            self.feat_proj = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0)
        else:
            self.feat_proj = nn.Conv2d(out_channels, out_channels, 1, 1, 0)

        self.conv_heatmap = nn.Conv2d(out_channels, num_heatmap, 1)

        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 22),
            nn.Sigmoid(),
        )

        self.criteria = nn.MSELoss(reduction="mean")

    def get_loss(self, pred_heatmap, gt_heatmap):
        loss_heatmap = self.criteria(pred_heatmap, gt_heatmap)
        loss_heatmap *= self.train_cfg.get("w_heatmap", 1.0)
        return loss_heatmap

    def forward_backbone(self, img, depth_map=None):
        return self.encoder(img, depth_map)

    def forward(self, img, depth_map=None, gt_joint_2d=None):
        if gt_joint_2d is not None:
            heatmap_gt = generate_heatmaps_batched_torch(gt_joint_2d, 64, 80, 256, 320).unsqueeze(1)
        else:
            heatmap_gt = None

        if self.training:
            return self.forward_train(img, depth_map, heatmap_gt, gt_joint_2d)
        else:
            return self.forward_test(img, depth_map, heatmap_gt, gt_joint_2d)

    def _extract_features(self, img, depth_map):
        """Extract and project backbone features."""
        B, V, C, img_h, img_w = img.shape

        if self.depth:
            rgb_feats, depth_feats = self.forward_backbone(img, depth_map)
            rgb_feats = rgb_feats.reshape(B * V, *rgb_feats.shape[2:])
            depth_feats = depth_feats.reshape(B * V, *depth_feats.shape[2:])
            combined_feats = torch.cat((rgb_feats, depth_feats), dim=1)
            frame_feats = self.feat_proj(combined_feats)
        else:
            rgb_feats = self.forward_backbone(img)
            rgb_feats = rgb_feats.reshape(B * V, *rgb_feats.shape[2:])
            frame_feats = self.feat_proj(rgb_feats)

        return frame_feats

    def forward_train(self, img, depth_map, heatmap_gt, gt_joint_2d, **kwargs):
        B, V, C, img_h, img_w = img.shape
        frame_feats = self._extract_features(img, depth_map)

        heatmaps = self.conv_heatmap(frame_feats)
        presence_prob = self.presence_head(frame_feats).squeeze(-1).squeeze(-1)

        # Presence targets from 2D joint visibility
        presence_targets = (
            (gt_joint_2d[..., 0] >= 0) & (gt_joint_2d[..., 0] < img_w) &
            (gt_joint_2d[..., 1] >= 0) & (gt_joint_2d[..., 1] < img_h)
        ).float()

        bce_loss = F.binary_cross_entropy(presence_prob, presence_targets)

        # Visibility-masked heatmap loss
        mask = presence_targets.unsqueeze(-1).unsqueeze(-1)
        loss_per_pixel = F.mse_loss(heatmaps, heatmap_gt.squeeze(1), reduction="none")
        masked_loss = (loss_per_pixel * mask).sum() / (mask.sum() + 1e-6)

        total_loss = masked_loss + bce_loss

        return dict(
            heatmap_loss=masked_loss,
            presence_loss=bce_loss,
            total_loss=total_loss,
        )

    def forward_test(self, img, depth_map, heatmap_gt=None, gt_joint_2d=None, **kwargs):
        with torch.no_grad():
            B, V, img_c, img_h, img_w = img.shape
            frame_feats = self._extract_features(img, depth_map)

            heatmaps = self.conv_heatmap(frame_feats)
            presence_prob = self.presence_head(frame_feats).squeeze(-1).squeeze(-1)

            heatmaps = heatmaps.view(B, V, *heatmaps.shape[1:])

            if heatmap_gt is not None:
                presence_targets = (
                    (gt_joint_2d[..., 0] >= 0) & (gt_joint_2d[..., 0] < img_w) &
                    (gt_joint_2d[..., 1] >= 0) & (gt_joint_2d[..., 1] < img_h)
                ).float()

                pred_binary = (presence_prob >= 0.5).float()
                accuracy = (pred_binary == presence_targets).float().mean().detach().cpu().numpy()

                heatmap_gt_list = [heatmap_gt[:, i] for i in range(V)]
                heatmap_list = [heatmaps[:, i] for i in range(V)]

                heatmap_list_flat = [x.reshape(B, -1) for x in heatmap_list]
                heatmap_gt_list_flat = [y.reshape(B, -1) for y in heatmap_gt_list]

                error = sum(
                    torch.abs(x - y)
                    for x, y in zip(heatmap_list_flat, heatmap_gt_list_flat)
                )
                error = error.sum(dim=1).reshape(B,).detach().cpu().numpy()

                pos_inds = [y > 0 for y in heatmap_gt_list_flat]
                pos_error = [
                    sum(
                        torch.abs(x[i][ind[i]] - y[i][ind[i]]).sum().detach().cpu().numpy()
                        for x, y, ind in zip(heatmap_list_flat, heatmap_gt_list_flat, pos_inds)
                    )
                    for i in range(B)
                ]

                pred_2d = argmax_2d(heatmaps.squeeze(1), presence_prob)
                gt_2d = argmax_2d(heatmap_gt.squeeze(1), presence_targets)

                mpjpe_2d = torch.norm(pred_2d - gt_2d, dim=2).mean().detach().cpu().numpy()

                return dict(
                    l1_error=error,
                    pos_l1_error=torch.tensor(pos_error, device=img.device),
                    mpjpe_2d=mpjpe_2d,
                    accuracy=accuracy,
                    heatmap=heatmaps.squeeze(1).detach().cpu().numpy(),
                    pred_2d=pred_2d.detach().cpu().numpy(),
                )
            else:
                pred_2d = argmax_2d(heatmaps.squeeze(1), presence_prob)
                return dict(
                    heatmap=heatmaps.squeeze(1).detach().cpu().numpy(),
                    pred_2d=pred_2d.detach().cpu().numpy(),
                    presence_prob=presence_prob,
                )
