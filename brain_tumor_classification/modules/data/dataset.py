"""Module with datasets"""
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from torch.utils.data import Dataset

from brain_tumor_classification.modules.data.utils import get_load_transforms


class BrainDataset(Dataset):
    def __init__(
        self, list_of_paths: List[Union[Path, str]], labels_path: Union[Path, str]
    ):
        self.list_of_paths = list_of_paths

        self.img_key = 'image'
        self.lbl_key = 'label'

        self.labels = self._load_labels(labels_path=labels_path)
        self.load_transforms = get_load_transforms(
            img_key=self.img_key,
            original_min=0,
            original_max=1000,
            res_min=0,
            res_max=1,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.list_of_paths[idx]
        image_idx = self._get_image_idx(image_path=image_path)

        image_label = self.labels[image_idx]

        item = {self.img_key: image_path, self.lbl_key: image_label}

        item = self.load_transforms(item)

        return item

    def __len__(self):
        return len(self.list_of_paths)

    @staticmethod
    def _load_labels(labels_path: Union[Path, str]) -> Dict[int, int]:
        labels_dataframe = pd.read_csv(filepath_or_buffer=labels_path)

        image_names = labels_dataframe['BraTS21ID'].tolist()
        label_values = labels_dataframe['MGMT_value'].tolist()

        name_label_mapping = dict(zip(image_names, label_values))

        return name_label_mapping

    @staticmethod
    def _get_image_idx(image_path: Union[str, Path]) -> int:
        image_path = str(image_path)
        image_filename = image_path.split(os.sep)[-1]

        assert image_filename.endswith('.nii.gz'), 'Bad file provided'

        image_index = int(image_filename.split('_')[0])

        return image_index
