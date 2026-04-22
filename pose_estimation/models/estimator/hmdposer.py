
import torch
import torch.nn as nn

from pose_estimation.models.utils import utils_transform
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.body_model import lbs


class HMDPoser(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead):
        super(HMDPoser, self).__init__()

        self.linear_embedding = nn.Linear(input_dim, embed_dim)
        self.joint_embed = nn.Parameter(torch.zeros(1, 22, 128))

        temporal_encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, batch_first=True)
        joint_encoder_layer = nn.TransformerEncoderLayer(128, nhead=nhead, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_encoder_layer, num_layers=num_layer)
        self.joint_encoder = nn.TransformerEncoder(joint_encoder_layer, num_layers=num_layer)

        self.joint_mlp = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 22 * 128),
        )

        self.stabilizer = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        self.joint_rotation_decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )

        subject_gender = "male"
        bm_fname = f"support_data/body_models/smplh/{subject_gender}/model.npz"
        dmpl_fname = f"support_data/body_models/dmpls/{subject_gender}/model.npz"
        self.body_model = BodyModel(bm_fname=bm_fname, num_betas=16, num_dmpls=8, dmpl_fname=dmpl_fname)

    def make_rotation_matrix_from_XZ_batch(self, x_axis, z_axis, eps=1e-6):
        """
        Construct rotation matrices from forward (X) and up (Z) axis vectors.

        Args:
            x_axis: (B, 3) forward axis vectors.
            z_axis: (B, 3) up axis vectors.
            eps: numerical stability threshold.

        Returns:
            (B, 3, 3) rotation matrices.
        """
        x = torch.nn.functional.normalize(x_axis, dim=-1)
        z = torch.nn.functional.normalize(z_axis, dim=-1)

        dot = torch.abs(torch.sum(x * z, dim=-1, keepdim=True))
        parallel = dot > 1.0 - eps

        fallback_z = torch.where(
            torch.abs(x[:, 2:3]) < 1.0 - eps,
            torch.tensor([0.0, 0.0, 1.0], device=x.device).expand_as(z),
            torch.tensor([1.0, 0.0, 0.0], device=x.device).expand_as(z),
        )
        z = torch.where(parallel, fallback_z, z)

        y = torch.cross(z, x, dim=-1)
        y = torch.nn.functional.normalize(y, dim=-1)
        z = torch.cross(x, y, dim=-1)

        return torch.stack([x, y, z], dim=-1)

    def convert_to_ue(self, global_orientation, joint_rotation):
        """Convert SMPL pose to Unreal Engine coordinate system."""
        y = global_orientation[..., 1].clone()
        z = global_orientation[..., 2].clone()
        global_orientation[..., 1] = -z
        global_orientation[..., 2] = y

        global_orientation = utils_transform.aa2matrot(global_orientation.reshape(-1, 3))

        forward = torch.stack([
            global_orientation[:, 0, 0],
            global_orientation[:, 1, 0],
            -global_orientation[:, 2, 0],
        ], dim=1)

        up = torch.stack([
            global_orientation[:, 0, 1],
            global_orientation[:, 1, 1],
            -global_orientation[:, 2, 1],
        ], dim=1)

        rot_matrix = self.make_rotation_matrix_from_XZ_batch(forward, -up)
        global_orientation = utils_transform.matrot2sixd(rot_matrix)

        joint_rotation = joint_rotation.reshape(-1, 21, 3)
        joint_rotation[..., 2] = -joint_rotation[..., 2]
        joint_rotation = joint_rotation.reshape(-1, 3)
        joint_rotation = utils_transform.aa2quat(joint_rotation)
        joint_rotation[..., 1] = -joint_rotation[..., 1]
        joint_rotation = utils_transform.quat2sixd(joint_rotation).reshape(-1, 21 * 6)

        return global_orientation, joint_rotation

    def fk_ue(self, global_orientation, joint_rotation):
        """Forward kinematics in Unreal Engine coordinate system."""
        B = global_orientation.shape[0]

        global_orientation = utils_transform.sixd2aa(global_orientation.reshape(-1, 6)).float()
        joint_rotation = utils_transform.sixd2aa(joint_rotation.reshape(-1, 6)).float()

        global_orientation[..., :] = 0
        joint_rotation[..., 0] = -joint_rotation[..., 0]
        joint_rotation[..., 2] = -joint_rotation[..., 2]
        root_rotmat = utils_transform.aa2matrot(global_orientation).view(B, 1, 3, 3)

        theta = torch.pi / 2
        rotate_pos90_matrot = torch.tensor([[
            [1.0, 0.0, 0.0],
            [0.0, torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta))],
            [0.0, torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta))],
        ]], device=global_orientation.device)

        root_rotmat = rotate_pos90_matrot @ root_rotmat

        joint_rotmat = utils_transform.aa2matrot(joint_rotation).view(B, 21, 3, 3)
        rot_mats = torch.cat([root_rotmat, joint_rotmat], dim=1)

        v_shaped = self.body_model.init_v_template + lbs.blend_shapes(
            self.body_model.init_betas, self.body_model.shapedirs
        )
        joints_all = lbs.vertices2joints(self.body_model.J_regressor, v_shaped)
        joints = joints_all[:, :22]
        parents = self.body_model.kintree_table[0][:22].long()

        Jtr, _ = lbs.batch_rigid_transform(rot_mats, joints.expand(B, -1, -1), parents)
        Jtr[..., 1] = -Jtr[..., 1]
        Jtr = Jtr - Jtr[:, 0:1]

        return Jtr[:, :22] * 100

    def fk_smpl(self, global_orientation, joint_rotation, translation=None):
        """Forward kinematics using the SMPL body model."""
        global_orientation = utils_transform.sixd2aa(
            global_orientation.reshape(-1, 6)
        ).reshape(global_orientation.shape[0], -1).float()
        joint_rotation = utils_transform.sixd2aa(
            joint_rotation.reshape(-1, 6)
        ).reshape(joint_rotation.shape[0], -1).float()

        body_params = {"pose_body": joint_rotation, "root_orient": global_orientation}
        if translation is not None:
            body_params["trans"] = translation.unsqueeze(0)

        body_pose = self.body_model(**body_params)
        joint_position = body_pose.Jtr

        return joint_position[:, :22] * 100

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: [B, T, 54] HMD input features.

        Returns:
            joint_feats: [B, 22, 128] per-joint feature embeddings.
            global_orientation: [B, 6] root orientation in 6D representation.
            joint_rotation: [B, 126] joint rotations (21 joints x 6D).
        """
        x = self.linear_embedding(input_tensor)

        # Temporal encoder
        x = self.temporal_encoder(x)[:, -1]

        # Joint decoder
        x = self.joint_mlp(x).reshape(-1, 22, 128)
        x = x + self.joint_embed

        # Spatial encoder
        x = self.joint_encoder(x)

        global_orientation = self.stabilizer(x[:, 0])
        joint_rotation = self.joint_rotation_decoder(x[:, 1:]).reshape(-1, 126)

        return x, global_orientation, joint_rotation
