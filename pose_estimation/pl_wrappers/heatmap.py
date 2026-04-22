import os
from typing import Optional

import torch
from torch.utils.data import Dataset
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from pytorch_lightning import LightningModule
from pytorch_lightning.strategies import ParallelStrategy

from torch.optim.lr_scheduler import MultiStepLR

from pose_estimation.models.estimator import RGBDPoserHeatmap
from pose_estimation.datasets import EgoPoseVRHeatmapDataset


def get_dataset(dataset_type, root, split, **kwargs):
    assert split in ["train", "test", "validation"]
    assert dataset_type in ["egoposevr"]

    if dataset_type == "egoposevr":
        if split == "train":
            return EgoPoseVRHeatmapDataset(
                data_root=os.path.join(root, "egoposevr_impl"),
                info_json=os.path.join(root, "all_npz_paths.txt"),
                **kwargs
            )
        elif split == "validation":
            return EgoPoseVRHeatmapDataset(
                data_root=os.path.join(root, "egoposevr_impl"),
                info_json=os.path.join(root, "val_npz_paths.txt"),
                **kwargs
            )
        else:
            return EgoPoseVRHeatmapDataset(
                data_root=os.path.join(root, "egoposevr_impl"),
                info_json=os.path.join(root, "test_npz_paths.txt"),
                **kwargs
            )
    else:
        raise NotImplementedError


class PoseHeatmapLightningModel(LightningModule):
    def __init__(
        self,
        model_cfg: dict,
        dataset_type: str,
        data_root: str,
        lr: float,
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

        self.model = RGBDPoserHeatmap(**model_cfg)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_iters = warmup_iters

        self.data_root = data_root
        self.batch_size = batch_size
        self.workers = workers

        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        assert self.model.training

        image = batch["img"]
        gt_joint_2d = batch["gt_joint_2d"]
        depth_map = batch["depth_map"]
        loss_dict = self.model(image, depth_map, gt_joint_2d)
        loss_total = loss_dict.get("total_loss")
        self.log("heatmap_loss", loss_dict.get("heatmap_loss").mean())
        self.log("presence_loss", loss_dict.get("presence_loss").mean())
        self.log("train_loss", loss_total)
        return loss_total

    def eval_step(self, batch, batch_idx, prefix):
        assert not self.model.training

        image = batch["img"]
        gt_joint_2d = batch["gt_joint_2d"]
        depth_map = batch["depth_map"]
        output_dict = self.model(image, depth_map, gt_joint_2d)
        self.log(f"{prefix}_l1_error", output_dict.get("l1_error").mean())
        self.log(f"{prefix}_pos_l1_error", output_dict.get("pos_l1_error").mean())
        self.log(f"{prefix}_mpjpe_2d", output_dict.get("mpjpe_2d").mean())
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
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = MultiStepLR(optimizer, self.lr_decay_epochs, gamma=0.1)
        return [optimizer], [scheduler]

    def setup(self, stage: str):
        if isinstance(self.trainer.strategy, ParallelStrategy):
            num_processes = max(1, self.trainer.strategy.num_processes)
            self.batch_size = int(self.batch_size / num_processes)
            self.workers = int(self.workers / num_processes)

        if stage == "fit":
            self.train_dataset = get_dataset(self.dataset_type, self.data_root, "train", **self.dataset_kwargs)

        if stage == "test":
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "test", **self.dataset_kwargs)
        else:
            self.eval_dataset = get_dataset(self.dataset_type, self.data_root, "validation", **self.dataset_kwargs)

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
