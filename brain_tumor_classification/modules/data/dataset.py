"""Module with datasets"""

from typing import List

from torch.utils.data import Dataset


class BrainDataset(Dataset):
    def __init__(self, list_of_paths: List[str]):
        self.list_of_paths = list_of_paths

    def __getitem__(self, idx: int) -> None:
        pass

    def __len__(self):
        pass
