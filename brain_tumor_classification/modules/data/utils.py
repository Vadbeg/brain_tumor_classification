"""Module with utilities for dataset"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
    Transform,
)
from torch.utils.data import DataLoader, Dataset


def get_train_val_paths(
    train_path: Union[str, Path],
    train_split_percent: float = 0.7,
    ct_file_extension: str = '*.nii.gz',
    item_limit: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[Path], List[Path]]:
    train_path = Path(train_path)

    list_of_paths = list(train_path.glob(ct_file_extension))
    if shuffle:
        np.random.shuffle(list_of_paths)

    edge_value = int(train_split_percent * len(list_of_paths))

    train_list_of_paths = list_of_paths[:edge_value]
    val_list_of_paths = list_of_paths[edge_value:]

    if item_limit:
        train_list_of_paths = train_list_of_paths[:item_limit]
        val_list_of_paths = val_list_of_paths[:item_limit]

    return train_list_of_paths, val_list_of_paths


def create_data_loader(
    dataset: Dataset, batch_size: int = 1, shuffle: bool = True, num_workers: int = 2
) -> DataLoader:
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return data_loader


def get_load_transforms(
    img_key: str,
    original_min: float = 0.0,
    original_max: float = 200.0,
    res_min: float = 0.0,
    res_max: float = 1.0,
    spatial_size: Tuple[int, int, int] = (196, 196, 128),
) -> Compose:
    preprocessing_transforms = get_preprocessing_transforms(
        img_key=img_key,
        original_min=original_min,
        original_max=original_max,
        res_min=res_min,
        res_max=res_max,
        spatial_size=spatial_size,
    )

    load_transforms = Compose(
        [LoadImaged(keys=[img_key], dtype=np.float32), *preprocessing_transforms]
    )

    return load_transforms


def get_preprocessing_transforms(
    img_key: str,
    original_min: float = 0.0,
    original_max: float = 200.0,
    res_min: float = 0.0,
    res_max: float = 1.0,
    spatial_size: Tuple[int, int, int] = (196, 196, 128),
) -> List[Transform]:
    preprocessing_transforms = [
        AddChanneld(keys=[img_key]),
        ScaleIntensityRanged(
            keys=[img_key],
            a_min=original_min,
            a_max=original_max,
            b_min=res_min,
            b_max=res_max,
            clip=True,
        ),
        Resized(keys=[img_key], spatial_size=spatial_size),
    ]

    return preprocessing_transforms
