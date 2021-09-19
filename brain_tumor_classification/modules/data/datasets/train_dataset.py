"""Module with datasets"""
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

from brain_tumor_classification.modules.data.datasets.base_dataset import BaseDataset
from brain_tumor_classification.modules.data.utils import (
    get_load_transforms,
    get_preprocessing_transforms,
)


class BrainDataset(BaseDataset):
    def __init__(
        self,
        list_of_paths: Union[List[Path], List[str]],
        labels_path: Union[Path, str],
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
        index_position_in_name: int = 0,
        original_clip_min: int = 0,
        original_clip_max: int = 1000,
    ):
        super().__init__(list_of_paths=list_of_paths)

        self.list_of_paths = list_of_paths
        self.index_position_in_name = index_position_in_name

        self.labels = self._load_labels(labels_path=labels_path)
        self.list_of_paths = self._get_paths_with_labels(
            list_of_paths=list_of_paths,
            labels=self.labels,
            index_position_in_name=index_position_in_name,
        )

        self.load_transforms = get_preprocessing_transforms(
            img_key=self.img_key,
            original_min=original_clip_min,
            original_max=original_clip_max,
            res_min=0,
            res_max=1,
            spatial_size=spatial_size,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.list_of_paths[idx]
        image_idx = self._get_image_idx(
            image_path=image_path, index_position_in_name=self.index_position_in_name
        )

        image_label = self.labels[image_idx]
        image = self._load_ct(ct_path=image_path)

        item = {self.img_key: image, self.lbl_key: image_label}

        item = self.load_transforms(item)

        return item

    def __len__(self):
        return len(self.list_of_paths)

    @staticmethod
    def _load_ct(ct_path: Union[str, Path]) -> np.ndarray:
        ct_path = str(ct_path)

        ct_all_info: nib.Nifti1Image = nib.load(filename=ct_path)
        orig_ornt = nib.io_orientation(ct_all_info.affine)
        targ_ornt = nib.orientations.axcodes2ornt(axcodes='LPS')
        transform = nib.orientations.ornt_transform(
            start_ornt=orig_ornt, end_ornt=targ_ornt
        )

        img_ornt = ct_all_info.as_reoriented(ornt=transform)

        return img_ornt.get_fdata(dtype=np.float64)

    @staticmethod
    def _load_labels(
        labels_path: Union[Path, str],
    ) -> Dict[int, int]:
        labels_dataframe: pd.DataFrame = pd.read_csv(filepath_or_buffer=labels_path)

        image_names = labels_dataframe['BraTS21ID'].tolist()
        label_values = labels_dataframe['MGMT_value'].tolist()

        name_label_mapping = dict(zip(image_names, label_values))

        return name_label_mapping

    def _get_paths_with_labels(
        self,
        list_of_paths: Union[List[Path], List[str]],
        labels: Dict[int, int],
        index_position_in_name: int = 0,
    ) -> List[Path]:
        image_names = labels.keys()
        paths_with_labels = []

        for path in list_of_paths:
            path = Path(path)

            image_idx = self._get_image_idx(
                image_path=path, index_position_in_name=index_position_in_name
            )

            if image_idx in image_names:
                paths_with_labels.append(path)

        return paths_with_labels

    def _get_all_image_idxs(
        self,
        all_image_paths: Union[List[Path], List[str]],
        index_position_in_name: int = 0,
    ) -> List[int]:
        all_image_idx: List[int] = []

        for image_path in all_image_paths:
            image_idx = self._get_image_idx(
                image_path=image_path, index_position_in_name=index_position_in_name
            )

            all_image_idx.append(image_idx)

        return all_image_idx

    @staticmethod
    def _get_image_idx(
        image_path: Union[str, Path], index_position_in_name: int = 0
    ) -> int:
        image_path = str(image_path)
        image_filename = image_path.split(os.sep)[-1]

        assert image_filename.endswith('.nii.gz'), 'Bad file provided'

        image_index = int(image_filename.split('_')[index_position_in_name])

        return image_index
