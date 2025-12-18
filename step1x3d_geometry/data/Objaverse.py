import math
import os
import json
import re
import cv2
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from step1x3d_geometry import register
from step1x3d_geometry.utils.typing import *
from step1x3d_geometry.utils.config import parse_structured

from streaming import StreamingDataLoader
from .base import BaseDataModuleConfig, BaseDataset


@dataclass
class ObjaverseDataModuleConfig(BaseDataModuleConfig):
    pass


class ObjaverseDataset(BaseDataset):
    pass


@register("Objaverse-datamodule")
class ObjaverseDataModule(pl.LightningDataModule):
    cfg: ObjaverseDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(ObjaverseDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = ObjaverseDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = ObjaverseDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = ObjaverseDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(
        self, dataset, batch_size, collate_fn=None, num_workers=0
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        # [修复] 如果 train_dataset 缺失，手动触发 setup("fit")。
        # 这是必须的，因为 DeepSpeed 策略即使在运行测试模式 (trainer.test()) 时，
        # 也可能尝试访问 train_dataloader 来自动配置 batch size。
        # 如果没有这个检查，它会引发 AttributeError: 'ObjaverseDataModule' object has no attribute 'train_dataset'。
        if not hasattr(self, "train_dataset"):
            self.setup(stage="fit")
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
