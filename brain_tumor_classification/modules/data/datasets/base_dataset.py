"""Module with datasets"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, list_of_paths: Union[List[Path], List[str]]):
        self.list_of_paths = list_of_paths

        self.img_key = 'image'
        self.lbl_key = 'label'

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError('It is base dataset, use implementation!')

    def __len__(self):
        raise NotImplementedError('It is base dataset, use implementation!')
