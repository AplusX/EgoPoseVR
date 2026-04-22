import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy

from torch.optim.lr_scheduler import MultiStepLR

from pose_estimation.models.estimator import RGBDPoser
from pose_estimation.datasets import EgoPoseVR3DPoseDataset

import numpy as np


def get_dataset(dataset_type, root, split, num_frames, depth, **kwargs):
    assert split in ["train", "test", "validation"]
    assert dataset_type in ["egoposevr"]

    if dataset_type == "egoposevr":
        if split == "train":
            return EgoPoseVR3DPoseDataset(
                data_root=os.path.join(root, "egoposevr_impl"),
                info_json=os.path.join(root, "train_npz_paths.txt"),
                num_frames=num_frames,
                depth=depth,
                **kwargs
            )
        elif split == "validation":
            return EgoPoseVR3DPoseDataset(
                data_root=os.path.join(root, "egoposevr_impl"),
                info_json=os.path.join(root, "val_npz_paths.txt"),
                num_frames=num_frames,
                depth=depth,
                **kwargs
            )
        else:
            return EgoPoseVR3DPoseDataset(
                data_root=os.path.join(root, "egoposevr_impl"),
                info_json=os.path.join(root, "test_npz_paths.txt"),
                num_frames=num_frames,
                depth=depth,
                **kwargs
            )
    else:
        raise NotImplementedError


class Pose3DLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        data_root: str,
        lr: float,
        encoder_lr_scale: float,
        weight_decay: float,
        lr_decay_epochs: tuple,
        warmup_iters: int,
        batch_size: int,
        workers: int,
        dataset_kwargs: dict = {}
    ):
        super().__init__()

        assert dataset_type in ["egoposevr"]

        self.dataset_type = dataset_type
        self.dataset_kwargs = dataset_kwargs

        self.model = RGBDPoser(**model_cfg)

        self.num_frames = model_cfg["num_frames"]
        self.depth = model_cfg["depth"]
        self.lr = lr
        self.encoder_lr_scale = encoder_lr_scale
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_iters = warmup_iters

        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

        self.test_fps_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        assert self.model.training

        L = batch["L"]
        image = batch["img"]
        depth_map = batch["depth_map"]
        joint_2d = batch["joint_2d"]
        gt_global_ori = batch["gt_global_ori"]
        gt_joint_rot = batch["gt_joint_rot"]

        loss_dict = self.model(L, image, joint_2d, gt_global_ori, gt_joint_rot)
        loss_total = sum(loss_dict.values())

        for k, v in loss_dict.items():
            self.log(k, v)
        self.log("total_loss", loss_total)
        return loss_total

    def eval_step(self, batch, batch_idx, prefix):
        assert not self.model.training

        L = batch["L"]
        image = batch["img"]
        depth_map = batch["depth_map"]
        joint_2d = batch["joint_2d"]
        gt_global_ori = batch["gt_global_ori"]
        gt_joint_rot = batch["gt_joint_rot"]
        output_dict = self.model(L, image, joint_2d, gt_global_ori, gt_joint_rot)
        for k, v in output_dict.items():
            if "mpjpe" in k or "mpjre" in k:
                self.log(prefix + "_" + k + "_mean", v.mean(), sync_dist=True)
                self.log(prefix + "_" + k + "_std", v.std(), sync_dist=True)

        return None

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, "test")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        if self.trainer.global_step < self.warmup_iters:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / float(self.warmup_iters))
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

    def configure_optimizers(self):
        if self.encoder_lr_scale == 1.0:
            no_decay_params = []
            other_params = []
            for name, param in self.model.named_parameters():
                if 'norm' in name or 'bn' in name or 'ln' in name or 'bias' in name:
                    no_decay_params.append(param)
                else:
                    other_params.append(param)

            optimizer = optim.AdamW(
                [
                    {'params': no_decay_params, 'weight_decay': 0.0},
                    {'params': other_params, 'weight_decay': self.weight_decay}
                ],
                lr=self.lr
            )
        else:
            param_groups = [
                {'params': self.model.encoder.parameters(), 'lr': self.lr * self.encoder_lr_scale}
            ]
            param_groups.append(
                {
                    'params': [
                        param for name, param in self.model.named_parameters() if not name.startswith('encoder')
                    ]
                }
            )
            optimizer = optim.AdamW(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(optimizer, self.lr_decay_epochs, gamma=0.1)

        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        print(f"Batch size: {self.batch_size}")

        if stage == "fit":
            self.train_dataset = get_dataset(self.dataset_type, self.data_root, "train", self.num_frames, self.depth, **self.dataset_kwargs)

        if stage == "test":
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "test", self.num_frames, self.depth, **self.dataset_kwargs)
        else:
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "validation", self.num_frames, self.depth, **self.dataset_kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False
        )

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False
        )
