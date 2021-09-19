"""Training module"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader

from brain_tumor_classification.modules.data.datasets.train_dataset import BrainDataset
from brain_tumor_classification.modules.data.utils import (
    create_data_loader,
    get_train_val_paths,
)
from brain_tumor_classification.modules.model.effnet.effnet import EfficientNet3D
from brain_tumor_classification.modules.model.resnet.resnet import generate_resnet_model


class BrainClassification3DModel(pl.LightningModule):
    def __init__(
        self,
        train_dataset_path: Union[str, Path],
        labels_path: Union[str, Path],
        train_split_percent: float = 0.7,
        dataset_item_limit: Optional[int] = 1,
        shuffle_dataset: bool = True,
        ct_file_extension: str = '*.nii.gz',
        index_position_in_name: int = 0,
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
        original_clip_min: int = 0,
        original_clip_max: int = 1000,
        batch_size: int = 2,
        num_processes: int = 1,
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
            ct_file_extension=ct_file_extension,
        )
        self.labels_path = labels_path

        self.index_position_in_name = index_position_in_name
        self.original_clip_min = original_clip_min
        self.original_clip_max = original_clip_max
        self.num_classes = num_classes
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.learning_rate = learning_rate

        self.loss = torch.nn.CrossEntropyLoss()
        self.model = generate_resnet_model(
            model_depth=model_depth,
            n_input_channels=num_input_channels,
            n_classes=num_classes,
        )
        self.model = EfficientNet3D.from_name(
            "efficientnet-b5", override_params={'num_classes': 2}, in_channels=1
        )

        self.f1_func = torchmetrics.F1(num_classes=num_classes)
        self.acc_func = torchmetrics.Accuracy(num_classes=num_classes)
        self.roc_auc_func = torchmetrics.AUROC(num_classes=num_classes)

    def training_step(
        self, batch: Dict, batch_id: int
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        image = batch[self.img_key]
        label = batch[self.lbl_key]

        result = self.model(image)
        loss = self.loss(result, label)

        self.log(
            name='train_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='train')

        return {'loss': loss, 'pred': result, 'label': label}

    def validation_step(
        self, batch: Dict, batch_id: int
    ) -> Dict[str, Any]:  # pylint: disable=W0613
        image = batch[self.img_key]
        label = batch[self.lbl_key]

        result = self.model(image)
        loss = self.loss(result, label)

        self.log(
            name='val_loss',
            value=loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self._log_metrics(preds=result, target=label, prefix='val')

        return {'loss': loss, 'pred': result, 'label': label}

    def train_dataloader(self) -> DataLoader:
        train_brain_dataset = BrainDataset(
            list_of_paths=self.train_paths,
            labels_path=self.labels_path,
            spatial_size=self.spatial_size,
            index_position_in_name=self.index_position_in_name,
            original_clip_max=self.original_clip_max,
            original_clip_min=self.original_clip_min,
        )

        train_brain_dataloader = create_data_loader(
            dataset=train_brain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_processes,
        )

        return train_brain_dataloader

    def val_dataloader(self) -> DataLoader:
        val_brain_dataset = BrainDataset(
            list_of_paths=self.train_paths,
            labels_path=self.labels_path,
            spatial_size=self.spatial_size,
            index_position_in_name=self.index_position_in_name,
            original_clip_max=self.original_clip_max,
            original_clip_min=self.original_clip_min,
        )

        val_brain_dataloader = create_data_loader(
            dataset=val_brain_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_processes,
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

    def _log_metrics(
        self, preds: torch.Tensor, target: torch.Tensor, prefix: str
    ) -> None:
        f1_value = self.f1_func(preds, target)
        acc_value = self.acc_func(preds, target)

        self.log(
            name=f'{prefix}_f1',
            value=f1_value,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )
        self.log(
            name=f'{prefix}_acc',
            value=acc_value,
            prog_bar=True,
            logger=True,
            on_epoch=True,
        )

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        all_batch_predictions = []
        all_batch_labels = []

        for curr_output in outputs:
            all_batch_predictions.append(curr_output['pred'])
            all_batch_labels.append(curr_output['label'])

        predictions = torch.cat(tensors=all_batch_predictions)
        labels = torch.cat(tensors=all_batch_labels)

        try:
            predictions = torch.softmax(predictions, dim=1)[:, 1]
            roc_auc_value = self.roc_auc_func(predictions, labels)

            self.log(
                name=f'train_roc_auc',
                value=roc_auc_value,
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
        except ValueError as err:
            warnings.warn(str(err))

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        all_batch_predictions = []
        all_batch_labels = []

        for curr_output in outputs:
            all_batch_predictions.append(curr_output['pred'])
            all_batch_labels.append(curr_output['label'])

        predictions = torch.cat(tensors=all_batch_predictions)
        labels = torch.cat(tensors=all_batch_labels)

        try:
            predictions = torch.softmax(predictions, dim=1)[:, 1]
            roc_auc_value = self.roc_auc_func(predictions, labels)

            self.log(
                name=f'val_roc_auc',
                value=roc_auc_value,
                prog_bar=True,
                logger=True,
                on_epoch=True,
            )
        except ValueError as err:
            warnings.warn(str(err))

    def _get_model_output(self):
        pass
