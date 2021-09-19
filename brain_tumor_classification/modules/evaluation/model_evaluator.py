"""Module with class for model evaluation"""

import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from brain_tumor_classification.modules.data.datasets.base_dataset import BaseDataset


class ModelEvaluator:
    _CSV_COLUMN_NAMES = ['BraTS21ID', 'MGMT_value']

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: BaseDataset,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.dataset = dataset
        self.device = device

    def eval(
        self, save_to_csv: bool = False, csv_filepath: str = 'sample_submission.csv'
    ) -> List[Tuple[int, int]]:
        evaluation_result = []

        for idx, dataset_item in enumerate(tqdm(self.dataset, postfix='Evaluation...')):
            image = dataset_item[self.dataset.img_key]
            image_path = self.dataset.list_of_paths[idx]
            image_idx = self._get_image_idx(image_path=image_path)

            label = self._predict_one_item(image=image)

            evaluation_result.append((image_idx, label))

        if save_to_csv:
            self._save_evaluation_result_to_csv(
                evaluation_result=evaluation_result, filename=csv_filepath
            )

        return evaluation_result

    def _predict_one_item(
        self,
        image: np.ndarray,
    ) -> int:
        image_tensor = self._to_tensor(image=image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        result = self.model(image_tensor)
        result = torch.softmax(result, dim=1).cpu()
        label = result[0, 1].detach().numpy()

        return label

    def _save_evaluation_result_to_csv(
        self,
        evaluation_result: List[Tuple[int, int]],
        filename: str = 'submission.csv',
    ) -> None:
        result_df = pd.DataFrame(
            data=evaluation_result,
            columns=self._CSV_COLUMN_NAMES,
        )

        result_df.to_csv(filename, index=False)

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image)

        return tensor

    @staticmethod
    def _get_image_idx(image_path: Union[str, Path]) -> int:
        image_path = str(image_path)
        image_case_name = image_path.split(os.sep)[-2]

        image_idx = int(image_case_name)

        return image_idx
