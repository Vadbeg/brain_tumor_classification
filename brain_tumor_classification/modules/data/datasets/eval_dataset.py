"""Module with evaluation dataset"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import SimpleITK as sitk

from brain_tumor_classification.modules.data.datasets.base_dataset import BaseDataset
from brain_tumor_classification.modules.data.utils import get_preprocessing_transforms


class BrainDicomEvalDataset(BaseDataset):
    def __init__(
        self,
        list_of_paths: Union[List[Path], List[str]],
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
    ):
        super().__init__(list_of_paths=list_of_paths)

        self.list_of_dicom_folder_paths = list_of_paths

        self.preprocessing_transforms = get_preprocessing_transforms(
            img_key=self.img_key,
            original_min=0,
            original_max=1000,
            res_min=0,
            res_max=1,
            spatial_size=spatial_size,
        )

    def __getitem__(self, idx: int) -> Dict:
        dicom_folder_path = self.list_of_dicom_folder_paths[idx]
        image = self.__load_dicom(dicom_folder_path=dicom_folder_path)

        item = {self.img_key: image}

        item = self.preprocessing_transforms(item)

        return item

    def __len__(self) -> int:
        return len(self.list_of_dicom_folder_paths)

    @staticmethod
    def __load_dicom(dicom_folder_path: Union[str, Path]) -> np.ndarray:
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(
            directory=str(dicom_folder_path)
        )
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_folder_path), series_ids[0]
        )
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.LoadPrivateTagsOn()
        image_and_meta: sitk.Image = series_reader.Execute()

        image = sitk.GetArrayFromImage(image=image_and_meta)

        return image
