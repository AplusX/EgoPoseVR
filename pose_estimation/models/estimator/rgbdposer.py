"""
RGBDPoser: Multi-modal egocentric 3D pose estimation model.

Combines HMD temporal signals with RGB/RGBD image features through
transformer-based architectures for full-body pose prediction.
"""

import copy
import os
from collections import OrderedDict

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from pose_estimation.models.backbones.resnet import ResnetBackbone
from pose_estimation.models.estimator.hmdposer import HMDPoser
from pose_estimation.models.utils.deform_attn import MSDeformAttn
from pose_estimation.models.utils.transformer import CustomMultiheadAttention, FFN
from pose_estimation.models.utils.camera_models import projection_funcs
from pose_estimation.models.utils.pose_metric import (
    MpjpeLoss,
    MpjreLoss,
    batch_compute_similarity_transform_numpy,
)
from pose_estimation.models.utils import utils_transform
from pose_estimation.models.utils import transform_pelvis

INF = 1e10
EPS = 1e-6


def optimize_upper_body_pose_batch(predicted_pose_batch, gt_positions_batch, known_indices):
    """
    Optimize upper body joint positions using least-squares with bone length
    and structural constraints.

    Args:
        predicted_pose_batch: [B, 22, 3] predicted joint positions.
        gt_positions_batch: [B, 22, 3] ground-truth joint positions.
        known_indices: list of known joint indices (e.g., [0, 15, 18, 21]).

    Returns:
        optimized_pose_batch: [B, 22, 3]
    """
    B, num_joints, dim = predicted_pose_batch.shape

    upper_body_indices = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    lower_body_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    skeleton_connections = [
        (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (12, 13), (12, 14),
        (13, 16), (16, 18), (18, 20), (14, 17), (17, 19), (19, 21),
    ]
    bone_lengths = [
        13.027, 14.061, 5.615, 21.468, 10.273, 13.033, 12.152,
        13.235, 25.684, 26.611, 12.283, 26.239, 26.926,
    ]

    w_align = 100.0
    w_self = 1.0
    w_self_pelvis = 1000.0
    w_length = 1000.0
    w_local_struct = 50.0

    optimized_pose_list = []

    for b in range(B):
        predicted_pose = predicted_pose_batch[b]
        gt_positions = gt_positions_batch[b]

        triplet_rows = []
        triplet_cols = []
        triplet_data = []
        b_vector = []
        eq_counter = 0

        # Alignment constraints for known joints
        for joint_idx in known_indices:
            if joint_idx not in upper_body_indices:
                continue
            for d in range(dim):
                triplet_rows.append(eq_counter)
                triplet_cols.append(joint_idx * dim + d)
                triplet_data.append(w_align)
                b_vector.append(w_align * gt_positions[joint_idx, d].item())
                eq_counter += 1

        # Self-prior for upper-body joints
        for joint_idx in upper_body_indices:
            if joint_idx in known_indices:
                continue
            this_w = w_self_pelvis if joint_idx == 0 else w_self
            for d in range(dim):
                triplet_rows.append(eq_counter)
                triplet_cols.append(joint_idx * dim + d)
                triplet_data.append(this_w)
                b_vector.append(this_w * predicted_pose[joint_idx, d].item())
                eq_counter += 1

        # Self-prior for lower-body joints
        for joint_idx in lower_body_indices:
            for d in range(dim):
                triplet_rows.append(eq_counter)
                triplet_cols.append(joint_idx * dim + d)
                triplet_data.append(w_self_pelvis)
                b_vector.append(w_self_pelvis * predicted_pose[joint_idx, d].item())
                eq_counter += 1

        # Bone length constraints
        current_bonepair = 0
        for parent, child in skeleton_connections:
            if parent not in upper_body_indices or child not in upper_body_indices:
                continue
            rest_vector = predicted_pose[parent] - predicted_pose[child]
            current_length = torch.norm(rest_vector).item()
            if current_length < 1e-6:
                continue
            direction = rest_vector / current_length
            rest_length = bone_lengths[current_bonepair]
            current_bonepair += 1

            for d in range(dim):
                triplet_rows.append(eq_counter)
                triplet_cols.append(parent * dim + d)
                triplet_data.append(w_length * direction[d].item())
                triplet_rows.append(eq_counter)
                triplet_cols.append(child * dim + d)
                triplet_data.append(-w_length * direction[d].item())
            b_vector.append(w_length * rest_length)
            eq_counter += 1

        # Local structure preservation
        for parent, child in skeleton_connections:
            if parent not in upper_body_indices or child not in upper_body_indices:
                continue
            rel_init = predicted_pose[child] - predicted_pose[parent]
            for d in range(dim):
                triplet_rows.append(eq_counter)
                triplet_cols.append(child * dim + d)
                triplet_data.append(w_local_struct)
                triplet_rows.append(eq_counter)
                triplet_cols.append(parent * dim + d)
                triplet_data.append(-w_local_struct)
                b_vector.append(w_local_struct * rel_init[d].item())
                eq_counter += 1

        A = sp.coo_matrix(
            (triplet_data, (triplet_rows, triplet_cols)),
            shape=(eq_counter, num_joints * dim),
        )
        b_np = np.array(b_vector)
        x_opt, _ = spla.lsmr(A, b_np)[:2]

        optimized_pose = predicted_pose.clone().reshape(-1)
        optimized_pose[:len(x_opt)] = torch.tensor(x_opt, dtype=optimized_pose.dtype)
        optimized_pose = optimized_pose.reshape(num_joints, dim)
        optimized_pose_list.append(optimized_pose)

    return torch.stack(optimized_pose_list, dim=0)


class RGBDPoser(nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dims,
        temporal_cfg,
        encoder_cfg,
        transformer_cfg,
        num_former_layers,
        num_pred_mlp_layers,
        image_size,
        camera_model,
        feat_down_stride,
        coor_norm_max,
        coor_norm_min,
        norm_mlp_pred=False,
        num_joints=22,
        num_frames=1,
        to_mm=10.0,
        temporal_pretrained=None,
        encoder_pretrained=None,
        train_cfg=dict(w_mpjpe=1.0),
        depth=False,
        **kwargs,
    ):
        super(RGBDPoser, self).__init__(**kwargs)

        self.kpo = False
        self.depth = depth
        self.rgb_backbone = False
        self.CrossAtten_concat = True
        self.refinement = False
        self.gt_2d_proj = False

        self.invalid_pad = INF
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.embed_dims = embed_dims
        self.to_mm = to_mm

        self.feat_down_stride = feat_down_stride
        self.feat_shape = (
            image_size[0] // feat_down_stride,
            image_size[1] // feat_down_stride,
        )
        self.image_size = image_size

        # Temporal (HMD) branch
        self.temporal = HMDPoser(**temporal_cfg)

        # RGB encoder branch
        self.feat_proj = nn.Conv2d(input_dims, embed_dims, 1, 1, 0)
        self.encoder = ResnetBackbone(**encoder_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # RGB MLP branch
        rgb_input_dim = 66 if self.depth else 44
        self.rgb_linear_embedding = nn.Linear(rgb_input_dim, 256)
        rgb_temporal_encoder_layer = nn.TransformerEncoderLayer(256, nhead=8, batch_first=True)
        self.rgb_temporal_encoder = nn.TransformerEncoder(rgb_temporal_encoder_layer, num_layers=3)
        self.rgb_joint_embed = nn.Parameter(torch.zeros(1, 22, 128))
        rgb_joint_encoder_layer = nn.TransformerEncoderLayer(128, nhead=8, batch_first=True)
        self.rgb_joint_encoder = nn.TransformerEncoder(rgb_joint_encoder_layer, num_layers=3)

        self.rgb_joint_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 22 * 128),
        )
        self.rgb_stabilizer = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        self.rgb_joint_rotation_decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )

        # HMD+RGB cross-attention fusion
        self.cross_attn_fusion = nn.MultiheadAttention(
            embed_dim=embed_dims, num_heads=4, batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dims)
        self.cross_attn_stabilizer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, 6),
        )
        self.cross_attn_joint_rotation_decoder = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, 6),
        )

        # # HMD+RGB concat fusion (deprecated)
        # self.concat_stabilizer = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 6),
        # )
        # self.concat_joint_rotation_decoder = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 6),
        # )


        self.stabilizer = torch.nn.ModuleList()
        for _ in range(num_former_layers):
            stabilizer = []
            for i in range(num_pred_mlp_layers - 1):
                stabilizer.append(nn.Linear(embed_dims, embed_dims))
                stabilizer.append(nn.GELU())
            stabilizer.append(nn.Linear(embed_dims, 6))
            self.stabilizer.append(nn.Sequential(*stabilizer))

        self.joint_rotation_decoder = torch.nn.ModuleList()
        for _ in range(num_former_layers):
            decoder = []
            for i in range(num_pred_mlp_layers - 1):
                decoder.append(nn.Linear(embed_dims, embed_dims))
                decoder.append(nn.GELU())
            decoder.append(nn.Linear(embed_dims, 6))
            self.joint_rotation_decoder.append(nn.Sequential(*decoder))

        self.post_norm = torch.nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(num_former_layers)]
        )

        self.norm_mlp_pred = norm_mlp_pred
        if norm_mlp_pred:
            self.register_buffer("coor_min", torch.tensor(coor_norm_min))
            self.register_buffer("coor_max", torch.tensor(coor_norm_max))

        self.camera_model = camera_model
        self._local_to_image = projection_funcs.get(camera_model)

        self.train_cfg = train_cfg

        self.load_pretrain(temporal_pretrained)
        self.load_pretrain(encoder_pretrained)
        self.pos_criteria = MpjpeLoss()
        self.rot_criteria = MpjreLoss()

    def _forward_cross_attn(self, hmd_joint_feats, rgb_joint_feats):
        """Fuse HMD and RGB features via cross-attention (HMD queries RGB)."""
        fused, _ = self.cross_attn_fusion(hmd_joint_feats, rgb_joint_feats, rgb_joint_feats)
        fused = self.cross_attn_norm(hmd_joint_feats + fused)
        global_orientation = self.cross_attn_stabilizer(fused[:, 0])
        joint_rotation = self.cross_attn_joint_rotation_decoder(fused[:, 1:]).reshape(-1, 126)
        
        return global_orientation, joint_rotation

    # def _forward_concat(self, hmd_joint_feats, rgb_joint_feats):
    #     """Fuse HMD and RGB features via concatenation."""
    #     x = torch.cat((hmd_joint_feats, rgb_joint_feats), dim=-1)
    #     global_orientation = self.concat_stabilizer(x[:, 0])
    #     joint_rotation = self.concat_joint_rotation_decoder(x[:, 1:]).reshape(-1, 126)
    #     return global_orientation, joint_rotation

    def _forward_mlp(self, frame_feats):
        """RGB 2D keypoint MLP branch."""
        B, T = frame_feats.shape[:2]
        x = self.rgb_linear_embedding(frame_feats.reshape(B, T, -1))

        # Temporal encoder
        x = self.rgb_temporal_encoder(x)[:, -1]
        x = self.rgb_joint_mlp(x).reshape(B, self.num_joints, 128)
        x = x + self.rgb_joint_embed
        # Spatial encoder
        x = self.rgb_joint_encoder(x)

        global_orientation = self.rgb_stabilizer(x[:, 0])
        joint_rotation = self.rgb_joint_rotation_decoder(x[:, 1:]).reshape(-1, 126)

        return x, global_orientation, joint_rotation

    def _forward_temporal(self, L):
        """HMD temporal branch."""
        return self.temporal(L)
    

    def _forward(self, L, img, joint_2d=None, gt_global_ori=None, gt_joint_rot=None, gt_trans=None, debug=False):
        """Main forward logic dispatching to different branches."""
        if self.rgb_backbone:
            rgb_joint_feats, init_global_ori, init_joint_rot = self._forward_mlp(joint_2d)
        else:
            joint_feats, init_global_ori, init_joint_rot = self._forward_temporal(L)

        init_joint_3d = self.temporal.fk_smpl(init_global_ori, init_joint_rot, gt_trans)

        preds_3d = [init_joint_3d]
        preds_ori = [init_global_ori]
        preds_rot = [init_joint_rot]

        if self.CrossAtten_concat:
            rgb_joint_feats, global_ori_offset, joint_rot_offset = self._forward_mlp(joint_2d)

            global_orientation, joint_rotation = self._forward_cross_attn(joint_feats, rgb_joint_feats)

            # init_global_ori, init_joint_rot = self._forward_cross_attn(joint_feats, rgb_joint_feats)
            # global_orientation = init_global_ori + global_ori_offset
            # joint_rotation = init_joint_rot + joint_rot_offset
            
            joint_3d = self.temporal.fk_smpl(global_orientation, joint_rotation, gt_trans)
            preds_3d.append(joint_3d)
            preds_ori.append(global_orientation)
            preds_rot.append(joint_rotation)

        return preds_3d, preds_ori, preds_rot

    def load_pretrain(self, pretrain):
        if pretrain is None:
            return
        ckpt = torch.load(pretrain, map_location="cpu", weights_only=True)
        checkpoint_dict = {
            k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()
        }
        self.load_state_dict(checkpoint_dict, strict=False)

    def get_loss(self, pred_pose, gt_pose):
        mpjpe_loss = self.pos_criteria(pred_pose, gt_pose) * self.train_cfg.get("w_mpjpe", 1.0)
        return dict(mpjpe_loss=mpjpe_loss)

    def forward(self, L, img, joint_2d=None, gt_global_ori=None, gt_joint_rot=None, gt_trans=None, debug=False, **kwargs):
        if self.training:
            return self.forward_train(L, img, joint_2d, gt_global_ori, gt_joint_rot, **kwargs)
        else:
            return self.forward_test(L, img, joint_2d, gt_global_ori, gt_joint_rot, gt_trans, debug, **kwargs)

    def forward_train(self, L, img, joint_2d, gt_global_ori, gt_joint_rot, **kwargs):
        B = L.shape[0]

        gt_global_ori = utils_transform.aa2sixd(gt_global_ori.reshape(B, 3))
        gt_joint_rot = utils_transform.aa2sixd(gt_joint_rot.reshape(-1, 3)).reshape(B, -1)
        gt_3d = self.temporal.fk_smpl(gt_global_ori, gt_joint_rot)

        preds_3d, preds_ori, preds_rot = self._forward(L, img, joint_2d, gt_global_ori, gt_joint_rot)

        losses = OrderedDict()
        for i, (pred_ori, pred_rot, pred_3d) in enumerate(zip(preds_ori, preds_rot, preds_3d)):
            losses[f"pelvisori_l1_loss_{i}"] = F.l1_loss(pred_ori, gt_global_ori) * 10
            losses[f"jointrot_l1_loss_{i}"] = F.l1_loss(pred_rot, gt_joint_rot) * 10
            losses_i = self.get_loss(pred_3d, gt_3d)
            for k, v in losses_i.items():
                losses[f"3d_{k}_{i}"] = v

        return losses

    def forward_test(self, L, img, joint_2d=None, gt_global_ori=None, gt_joint_rot=None, gt_trans=None, debug=False, **kwargs):
        B = L.shape[0]
        gt_3d = None

        if gt_global_ori is not None:
            gt_global_ori = utils_transform.aa2sixd(gt_global_ori.reshape(B, 3))
            gt_joint_rot = utils_transform.aa2sixd(gt_joint_rot.reshape(-1, 3)).reshape(B, -1)
            gt_3d = self.temporal.fk_smpl(gt_global_ori, gt_joint_rot, gt_trans)

        start_time = time.time()
        preds_3d, preds_ori, preds_rot = self._forward(L, img, joint_2d, gt_global_ori, gt_joint_rot, gt_trans, debug)

        if self.kpo:
            known_indices = [15, 20, 21]
            optimized_KPOpos = optimize_upper_body_pose_batch(preds_3d[-1], gt_3d, known_indices)
            preds_3d.append(optimized_KPOpos)

        end_time = time.time()
        fps = 1 / (end_time - start_time)

        pose_proposal = preds_3d[0]
        propose_ori = utils_transform.sixd2aa(preds_ori[0].reshape(-1, 6)).reshape(B, 3)
        propose_rot = utils_transform.sixd2aa(preds_rot[0].reshape(-1, 6)).reshape(B, -1)
        propose_rotation = torch.cat((propose_ori, propose_rot), dim=1).reshape(-1, 3)
        propose_rotation = utils_transform.aa2matrot(propose_rotation).reshape(B, 22, 3, 3)

        pred_3d_final = preds_3d[-1]
        pred_ori_final = utils_transform.sixd2aa(preds_ori[-1].reshape(-1, 6)).reshape(B, 3)
        pred_rot_final = utils_transform.sixd2aa(preds_rot[-1].reshape(-1, 6)).reshape(B, -1)
        pred_rotation = torch.cat((pred_ori_final, pred_rot_final), dim=1).reshape(-1, 3)
        pred_rotation = utils_transform.aa2matrot(pred_rotation).reshape(B, 22, 3, 3)

        if gt_global_ori is not None:
            gt_global_ori = utils_transform.sixd2aa(gt_global_ori.reshape(-1, 6)).reshape(B, 3)
            gt_joint_rot = utils_transform.sixd2aa(gt_joint_rot.reshape(-1, 6)).reshape(B, -1)
            gt_rotation = torch.cat((gt_global_ori, gt_joint_rot), dim=1).reshape(-1, 3)
            gt_rotation = utils_transform.aa2matrot(gt_rotation).reshape(B, 22, 3, 3)

        # UE projection for visualization
        pred_global_ori, pred_joint_rot = self.temporal.convert_to_ue(gt_global_ori, gt_joint_rot)
        pred_3d = self.temporal.fk_ue(pred_global_ori, pred_joint_rot)
        origin_3d, origin_rot = transform_pelvis.transform_pelvis_to_camera(pred_joint_rot, pred_3d)
        anchors_2d, anchors_valid = self._local_to_image(pred_3d, origin_3d, origin_rot)

        output_dict = OrderedDict()
        output_dict["pred_pose"] = pred_3d_final.detach().cpu().numpy()
        output_dict["propose_pose"] = pose_proposal.detach().cpu().numpy()
        output_dict["propose_pelvis_ori"] = propose_ori.detach().cpu().numpy()
        output_dict["propose_joint_rot"] = propose_rot.detach().cpu().numpy()
        output_dict["pred_pelvis_ori"] = pred_ori_final.detach().cpu().numpy()
        output_dict["pred_joint_rot"] = pred_rot_final.detach().cpu().numpy()
        output_dict["anchors_2d"] = anchors_2d
        output_dict["anchors_valid"] = anchors_valid
        output_dict["fps"] = fps

        if gt_3d is not None:
            output_dict["gt_pose"] = gt_3d.detach().cpu().numpy()
            metrics_final = self.evaluate(pred_3d_final, gt_3d, pred_rotation, gt_rotation, "final")
            metrics_proposal = self.evaluate(pose_proposal, gt_3d, propose_rotation, gt_rotation, "proposal")
            output_dict.update(metrics_final)
            output_dict.update(metrics_proposal)

        return output_dict

    def evaluate(self, pred_pose, pose_gt, pred_rot, gt_rot, prefix):
        """Compute MPJPE, PA-MPJPE, and MPJRE metrics."""
        B = pred_pose.shape[0]

        S1_hat = batch_compute_similarity_transform_numpy(pred_pose, pose_gt.to(dtype=torch.float))

        error = torch.linalg.norm(pred_pose - pose_gt, dim=-1, ord=2) * 10.0
        pa_error = torch.linalg.norm(S1_hat - pose_gt, dim=-1, ord=2) * 10.0
        rotation_error = geodesic_distance(pred_rot, gt_rot)

        mpjpe = error.mean(dim=1).reshape(B,).detach().cpu().numpy()
        pa_mpjpe = pa_error.mean(dim=1).reshape(B,).detach().cpu().numpy()
        mpjre = rotation_error.mean(dim=1).reshape(B,).detach().cpu().numpy()

        metrics = OrderedDict()
        metrics[f"{prefix}_mpjpe"] = mpjpe
        metrics[f"{prefix}_pa_mpjpe"] = pa_mpjpe
        metrics[f"{prefix}_mpjre"] = mpjre
        return metrics


class EgoformerSpatialMHA(CustomMultiheadAttention):
    """Joint-level spatial self-attention."""

    def forward(self, q, k, v, bias):
        B, J, C = q.shape

        _q = self.q_proj(q).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _k = self.k_proj(k).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _v = self.v_proj(v).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        attn = (_q @ _k.transpose(-2, -1)) * self.scale
        if bias is not None:
            attn = attn + bias
        attn = attn.softmax(dim=-1)

        x = (attn @ _v).permute(0, 2, 1, 3).reshape(B, J, C)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x

def geodesic_distance(m1, m2):
    """Compute geodesic distance between rotation matrices. Returns degrees."""
    m = torch.matmul(m1, m2.transpose(-2, -1))
    trace = m[:, :, 0, 0] + m[:, :, 1, 1] + m[:, :, 2, 2]
    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    return torch.acos(cos_theta) * (180.0 / torch.pi)
