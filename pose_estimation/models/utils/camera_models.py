"""
Camera projection models for mapping 3D joint positions to 2D image coordinates.
"""

import torch
from pose_estimation.models.utils import utils_transform


def egoposevr_proj(joint3d, pelvis2cam_Trans, pelvis2cam_rotation):
    """
    Project 3D joints to 2D using egocentric camera intrinsics.

    Args:
        joint3d: [B, J, 3] 3D joint positions.
        pelvis2cam_Trans: [B, 1, 3] pelvis-to-camera translation.
        pelvis2cam_rotation: [B, 3, 3] pelvis-to-camera rotation.

    Returns:
        image_coor_2d: [B, J, 2] normalized 2D coordinates.
        in_fov: [B, J] boolean mask for joints within field of view.
    """
    f_x = 228.503677
    f_y = 228.503693
    c_x = 160.000000
    c_y = 128.000000
    raw_image_size = (256, 320)

    rot = pelvis2cam_rotation
    rotated = rot @ joint3d.transpose(1, 2)
    rotated = rotated.transpose(1, 2)
    cam_3d = rotated + pelvis2cam_Trans
    cam_3d[..., [0, 2]] = -cam_3d[..., [0, 2]]

    x = cam_3d[..., 0]
    y = cam_3d[..., 1]
    z = cam_3d[..., 2]

    u = (f_x * x / z) + c_x
    v = raw_image_size[0] - ((f_y * y / z) + c_y)

    u = u / raw_image_size[1]
    v = v / raw_image_size[0]

    image_coor_2d = torch.stack((u, v), dim=-1)

    in_fov = (
        (image_coor_2d[..., 0] > 0)
        & (image_coor_2d[..., 1] > 0)
        & (image_coor_2d[..., 0] < 1)
        & (image_coor_2d[..., 1] < 1)
    )

    return image_coor_2d, in_fov


projection_funcs = {
    "egoposevr": egoposevr_proj,
}
