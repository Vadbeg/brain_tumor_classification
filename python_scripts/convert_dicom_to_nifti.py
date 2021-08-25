"""Script for converting DICOM to nifti"""

import argparse
import concurrent.futures
import gc
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, Union

import dicom2nifti.compressed_dicom as compressed_dicom
import dicom2nifti.convert_dicom as convert_dicom
import dicom2nifti.settings
from dicom2nifti.convert_dir import _is_valid_imaging_dicom
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
        convert_dicom_directory,
        output_folder=nifti_dir,
        compression=True,
        reorient=True,
    )

    convert_dicom_partial(folders_to_convert[0])

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


def convert_dicom_directory(
    dicom_directory: Union[str, Path],
    output_folder: Union[str, Path],
    compression: bool = True,
    reorient: bool = True,
) -> None:
    """
    This function will order all dicom files by series and order them one by one

    :param compression: enable or disable gzip compression
    :param reorient: reorient the dicoms according to LAS orientation
    :param output_folder: folder to write the nifti files to
    :param dicom_directory: directory with dicom files
    """
    sequence_type = str(dicom_directory).split(os.sep)[-1]
    sequence_id = str(dicom_directory).split(os.sep)[-2]

    dicom_series = get_dicom_series(dicom_directory=dicom_directory)

    save_dicom_file(
        dicom_series=dicom_series,
        sequence_id=sequence_id,
        sequence_type=sequence_type,
        output_folder=output_folder,
        compression=compression,
        reorient=reorient,
    )


def get_dicom_series(dicom_directory: Union[str, Path]) -> Dict:
    """
    Collects all dicom series info from dicom directory

    :param dicom_directory: path to directory with dicom
    :return: dicom info
    """

    dicom_series: Dict = dict()
    dicom_directory = Path(dicom_directory)

    for dicom_filepath in dicom_directory.glob(pattern='*.dcm'):
        try:
            if compressed_dicom.is_dicom_file(str(dicom_filepath)):
                dicom_headers = compressed_dicom.read_file(
                    dicom_filepath,
                    defer_size="1 KB",
                    stop_before_pixels=False,
                    force=dicom2nifti.settings.pydicom_read_force,
                )

                if not _is_valid_imaging_dicom(dicom_headers):
                    logger.info("Skipping: %s" % dicom_filepath)

                logger.info("Organizing: %s" % dicom_filepath)
                if dicom_headers.SeriesInstanceUID not in dicom_series:
                    dicom_series[dicom_headers.SeriesInstanceUID] = []
                dicom_series[dicom_headers.SeriesInstanceUID].append(dicom_headers)
        except Exception:
            logger.warning("Unable to read: %s" % dicom_filepath)

    return dicom_series


def save_dicom_file(
    dicom_series: Dict,
    sequence_id: str,
    sequence_type: str,
    output_folder: Union[str, Path],
    compression: bool = True,
    reorient: bool = True,
) -> None:
    """
    Saves dicom file from series

    :param dicom_series: dicom info
    :param sequence_id: id of the given sequence
    :param sequence_type: type of the sequence, same as name of the folder (i.e. FLAIR, T1w, T1wCE, T2w)
    :param output_folder: folder to which save resulted nifti
    :param compression: if True returns .gz files
    :param reorient: if True reorients the dicom according to LAS orientation
    """

    for series_id, dicom_input in dicom_series.items():
        name_of_file = sequence_id + '_' + sequence_type

        if compression:
            nifti_file = os.path.join(output_folder, name_of_file + '.nii.gz')
        else:
            nifti_file = os.path.join(output_folder, name_of_file + '.nii')

        convert_dicom.dicom_array_to_nifti(dicom_input, nifti_file, reorient)
        gc.collect()


if __name__ == '__main__':
    convert_dicom_cli()
