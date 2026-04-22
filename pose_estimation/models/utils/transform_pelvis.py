"""
Compute pelvis-to-camera transform for 2D projection of body joints.
"""

import torch
import numpy as np
from pose_estimation.models.utils import utils_transform
from human_body_prior.tools import tgm_conversion as tgm
from scipy.spatial.transform import Rotation as R


def transform_pelvis_to_camera(JointRotation_6dLocal, AllJoint2PelvisPositions):
    """
    Compute the pelvis-to-camera rotation and translation.

    Args:
        JointRotation_6dLocal: [B, 126] or [B, 21, 6] joint rotations in 6D representation.
        AllJoint2PelvisPositions: [B, 22, 3] joint positions relative to pelvis.

    Returns:
        Pelvis2Camera_Trans: [B, 1, 3] translation from pelvis to camera.
        Pelvis2Head_Rot: [B, 3, 3] rotation from pelvis to head.
    """
    JointRotation_6dLocal = JointRotation_6dLocal.reshape(-1, 21, 6)
    device = JointRotation_6dLocal.device
    B = JointRotation_6dLocal.shape[0]

    HeadtoNeck_6dRot = JointRotation_6dLocal[:, 14]
    NecktoSpine3_6dRot = JointRotation_6dLocal[:, 11]
    Spine3toSpine2_6dRot = JointRotation_6dLocal[:, 8]
    Spine2toSpine1_6dRot = JointRotation_6dLocal[:, 5]
    Spine1toPelvis_6dRot = JointRotation_6dLocal[:, 2]

    R_HeadtoNeck = utils_transform.sixd2matrot(HeadtoNeck_6dRot)
    R_NecktoSpine3 = utils_transform.sixd2matrot(NecktoSpine3_6dRot)
    R_Spine3toSpine2 = utils_transform.sixd2matrot(Spine3toSpine2_6dRot)
    R_Spine2toSpine1 = utils_transform.sixd2matrot(Spine2toSpine1_6dRot)
    R_Spine1toPelvis = utils_transform.sixd2matrot(Spine1toPelvis_6dRot)

    # Create -90 degree rotation matrix around X-axis
    theta = torch.tensor(-torch.pi / 2, device=device)
    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, torch.cos(theta), -torch.sin(theta)],
        [0.0, torch.sin(theta), torch.cos(theta)]
    ], device=device)
    rotate_neg90_matrot = Rx.unsqueeze(0).expand(B, -1, -1)

    # Chain multiplication along the bone chain from Head to Pelvis
    rot_chain = R_Spine1toPelvis @ R_Spine2toSpine1 @ R_Spine3toSpine2 @ R_NecktoSpine3 @ R_HeadtoNeck
    T = rotate_neg90_matrot @ rot_chain @ torch.linalg.inv(rotate_neg90_matrot)
    Pelvis2Head_Rot = torch.linalg.inv(T)

    # Translation from head to pelvis
    Head2Pelvis_Trans = AllJoint2PelvisPositions[:, 15].unsqueeze(-1)
    Pelvis2Head_Trans = -Pelvis2Head_Rot @ Head2Pelvis_Trans

    # Fixed camera offset relative to head
    CameraOffset = torch.tensor([0.0, 20.0, 9.0], device=device).expand(B, -1)
    Pelvis2Camera_Trans = Pelvis2Head_Trans.squeeze(-1) - CameraOffset

    return Pelvis2Camera_Trans.unsqueeze(1), Pelvis2Head_Rot
