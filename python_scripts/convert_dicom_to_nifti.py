"""Script for converting DICOM to nifti"""

import argparse
import concurrent.futures
import logging
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Union

import SimpleITK as sitk
from tqdm import tqdm

logger = logging.getLogger(__name__)


def convert_dicom_cli():
    args = get_args()

    dicom_dir = Path(args.dicom_dir)
    nifti_dir = Path(args.nifti_dir)
    data_type = args.data_type
    max_workers = args.num_workers

    pattern = os.path.join('**', data_type)
    folders_to_convert = list(dicom_dir.rglob(pattern=pattern))

    convert_dicom_partial = partial(
        convert_dicom_file_and_fix_align, output_folder=nifti_dir, compression=True
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(convert_dicom_partial, folders_to_convert),
                total=len(folders_to_convert),
                disable=False,
                postfix='Converting DICOM to nifti...',
            )
        )


def get_args():
    parser = argparse.ArgumentParser(
        description='Script for converting DICOM to nifti',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--dicom-dir',
        type=str,
        default=argparse.SUPPRESS,
        required=True,
        help='Path to folder with all DICOM files (Train or Val)',
    )
    parser.add_argument(
        '--nifti-dir',
        type=str,
        default=argparse.SUPPRESS,
        required=True,
        help='Path to folder with resulted nifti files',
    )
    parser.add_argument(
        '--data-type',
        type=str,
        choices=['FLAIR', 'T1w', 'T1wCE', 'T2w'],
        default='FLAIR',
        required=False,
        help='Type of dicom sequence',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        required=False,
        help='Number of workers for concurrency',
    )

    return parser.parse_args()


def convert_dicom_file_and_fix_align(
    dicom_directory: Union[str, Path],
    output_folder: Union[str, Path],
    compression: bool = True,
) -> None:
    dicom_directory = Path(dicom_directory).expanduser()
    output_folder = Path(output_folder).expanduser()

    image = load_dicom(dicom_directory=dicom_directory)

    res_file_path = get_nii_filepath(
        dicom_directory=dicom_directory,
        output_folder=output_folder,
        compression=compression,
    )

    sitk.WriteImage(
        image=image, fileName=str(res_file_path), useCompression=compression
    )


def load_dicom(dicom_directory: Union[str, Path]) -> sitk.Image:
    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory=str(dicom_directory))
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(dicom_directory), series_ids[0]
    )
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.LoadPrivateTagsOn()
    image = series_reader.Execute()

    return image


def get_nii_filepath(
    dicom_directory: Path,
    output_folder: Path,
    compression: bool = True,
) -> Path:
    sequence_type = str(dicom_directory).split(os.sep)[-1]
    sequence_id = str(dicom_directory).split(os.sep)[-2]

    res_file_name = f'{sequence_id}_{sequence_type}.nii'
    if compression:
        res_file_name += '.gz'

    res_file_path = output_folder.joinpath(res_file_name)

    return res_file_path


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    convert_dicom_cli()
