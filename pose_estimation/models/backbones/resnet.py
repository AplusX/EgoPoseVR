"""
ResNet backbone with Feature Pyramid Network (FPN) for feature extraction.
Supports RGB and optional depth input streams with separate backbones.

Reference: https://github.com/hiroyasuakada/UnrealEgo
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetTorchvision(nn.Module):
    def __init__(self, model_name, use_imagenet_pretrain, out_stride):
        super().__init__()
        backbone = self._build_resnet(model_name, use_imagenet_pretrain)

        base_layers = list(backbone.children())
        self.layer_s2 = nn.Sequential(*base_layers[:3])
        self.layer_s4 = nn.Sequential(*base_layers[3:5])
        self.layer_s8 = base_layers[5]
        self.layer_s16 = base_layers[6]
        self.layer_s32 = base_layers[7]

        self.out_stride = out_stride

    def _build_resnet(self, model_name, use_pretrain):
        if model_name == "resnet18":
            return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise NotImplementedError(f"Model {model_name} is not supported.")

    def forward(self, x):
        B, V, C, H, W = x.shape
        if C == 1:
            x = x.reshape(B * V, 1, H, W).repeat(1, 3, 1, 1)
        elif C == 3:
            x = x.reshape(B * V, C, H, W)

        out_s2 = self.layer_s2(x)
        out_s4 = self.layer_s4(out_s2)
        out_s8 = self.layer_s8(out_s4)
        out_s16 = self.layer_s16(out_s8)
        out_s32 = self.layer_s32(out_s16)

        for tensor_name in ["out_s2", "out_s4", "out_s8", "out_s16", "out_s32"]:
            tensor = locals()[tensor_name]
            locals()[tensor_name] = tensor.reshape(B, V, *tensor.shape[1:])

        out_s2 = out_s2.reshape(B, V, *out_s2.shape[1:])
        out_s4 = out_s4.reshape(B, V, *out_s4.shape[1:])
        out_s8 = out_s8.reshape(B, V, *out_s8.shape[1:])
        out_s16 = out_s16.reshape(B, V, *out_s16.shape[1:])
        out_s32 = out_s32.reshape(B, V, *out_s32.shape[1:])

        stride_to_output = {
            4: [out_s4, out_s8, out_s16, out_s32],
            8: [out_s8, out_s16, out_s32],
            16: [out_s16, out_s32],
            32: [out_s32],
        }
        if self.out_stride not in stride_to_output:
            raise NotImplementedError(f"out_stride={self.out_stride} not supported.")
        return stride_to_output[self.out_stride]


class EfficientFPN(nn.Module):
    """Feature Pyramid Network with efficient top-down fusion."""

    def __init__(self, in_channels, out_channels, with_relu=True):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_relu = with_relu

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.lateral_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            l_conv = [nn.Conv2d(in_channels[i], out_channels, 1)]
            if self.with_relu:
                l_conv.append(nn.ReLU(inplace=False))
            self.lateral_convs.append(nn.Sequential(*l_conv))

            if i != 0:
                fuse_conv = [nn.Conv2d(out_channels * 2, out_channels, 1, padding=0, stride=1)]
                if self.with_relu:
                    fuse_conv.append(nn.ReLU(inplace=False))
                self.fuse_convs.append(nn.Sequential(*fuse_conv))

                fpn_conv = [nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)]
                if self.with_relu:
                    fpn_conv.append(nn.ReLU(inplace=False))
                self.fpn_convs.append(nn.Sequential(*fpn_conv))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        B, V = inputs[0].shape[:2]

        inputs = [x.flatten(start_dim=0, end_dim=1) for x in inputs]

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.fpn_convs[i - 1](
                self.fuse_convs[i - 1](
                    torch.cat((laterals[i - 1], self.upsample(laterals[i])), dim=1)
                )
            )

        out = laterals[0].reshape(B, V, *laterals[0].shape[1:])
        return out


class ResnetBackbone(nn.Module):
    """Dual-stream ResNet backbone supporting RGB and optional depth inputs."""

    def __init__(self, resnet_cfg, neck_cfg, depth=False):
        super().__init__()
        self.rgb_backbone = ResNetTorchvision(**resnet_cfg)
        self.rgb_neck = EfficientFPN(**neck_cfg)

        self.depth_backbone = ResNetTorchvision(**resnet_cfg)
        self.depth_neck = EfficientFPN(**neck_cfg)

        self.depth = depth

    def get_output_channel(self):
        return self.rgb_neck.out_channels

    def forward(self, image, depth_map=None):
        rgb_feats = self.rgb_backbone(image)
        rgb_out = self.rgb_neck(rgb_feats)

        if self.depth:
            depth_feats = self.depth_backbone(depth_map)
            depth_out = self.depth_neck(depth_feats)
            return rgb_out, depth_out

        return rgb_out
