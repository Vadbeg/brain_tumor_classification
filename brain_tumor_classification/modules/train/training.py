"""Training module"""

import warnings
from multiprocessing import Lock
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from monai.losses import DiceLoss
from monai.metrics.meandice import compute_meandice
from pytorch_lightning.loggers.base import LoggerCollection
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from torch.utils.data import DataLoader

from brain_tumor_classification.modules.data.dataset import BrainDataset
from brain_tumor_classification.modules.data.utils import (
    create_data_loader,
    get_train_val_paths,
)
from brain_tumor_classification.modules.model.resnet import generate_resnet_model


class BrainClassification3DModel(pl.LightningModule):
    def __init__(
        self,
        train_dataset_path: Union[str, Path],
        labels_path: Union[str, Path],
        train_split_percent: float = 0.7,
        dataset_item_limit: Optional[int] = 1,
        shuffle_dataset: bool = True,
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
        batch_size: int = 2,
        learning_rate: float = 0.001,
        model_depth: int = 10,
        num_input_channels: int = 1,
        num_classes: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_key = 'image'
        self.lbl_key = 'label'

        self.train_paths, self.val_paths = get_train_val_paths(
            train_path=train_dataset_path,
            train_split_percent=train_split_percent,
            item_limit=dataset_item_limit,
            shuffle=shuffle_dataset,
        )
        self.labels_path = labels_path

        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.loss = torch.nn.CrossEntropyLoss()
        self.model = generate_resnet_model(
            model_depth=model_depth,
            n_input_channels=num_input_channels,
            n_classes=num_classes,
        )

    def training_step(
        self, batch: Dict, batch_id: int
    ) -> torch.Tensor:  # pylint: disable=W0613
        image = batch[self.img_key]
        label = batch[self.lbl_key]

        result = self.model(image)
        loss = self.loss(result, label)

        self.log(
            name='train_loss', value=loss, prog_bar=True, logger=True, on_epoch=True
        )

        return loss

    def validation_step(
        self, batch: Dict, batch_id: int
    ) -> torch.Tensor:  # pylint: disable=W0613
        image = batch[self.img_key]
        label = batch[self.lbl_key]

        result = self.model(image)
        loss = self.loss(result, label)

        self.log(name='val_loss', value=loss, prog_bar=True, logger=True, on_epoch=True)

        return loss

    def test_step(
        self,
        batch: Dict,
        batch_id: int,  # pylint: disable=W0613
        dataloader_idx: Optional[int] = None,  # pylint: disable=W0613
    ) -> torch.Tensor:
        self._initialize_interactions_simulator(deterministic=True)

        loss, dice = self._get_interactions_result_loss_metric(batch=batch)
        metric_postfix = self._get_metric_postfix(batch=batch)

        self.log(
            name=f'loss_\"{metric_postfix}\"',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            add_dataloader_idx=True,
        )
        self.log(
            name=f'dice_\"{metric_postfix}\"',
            value=torch.mean(dice),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            add_dataloader_idx=True,
        )

        return loss

    def train_dataloader(self) -> DataLoader:
        train_brain_dataset = BrainDataset(
            list_of_paths=self.train_paths,
            labels_path=self.labels_path,
            spatial_size=self.spatial_size,
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_brain_dataset = BrainDataset(
            list_of_paths=self.train_paths,
            labels_path=self.labels_path,
            spatial_size=self.spatial_size,
        )

        val_brain_dataloader = create_data_loader(
            dataset=val_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        return val_brain_dataloader

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, factor=0.5, patience=3, mode='min'
        )

        configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }

        return configuration

    def _get_model_output(self):
        pass
