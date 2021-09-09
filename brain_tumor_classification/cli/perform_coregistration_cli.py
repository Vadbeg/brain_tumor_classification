"""Module with script for performing coregistration"""


import argparse
from pathlib import Path
from typing import Union

import ants
from tqdm import tqdm


def perform_coregistration_cli() -> None:
    args = get_args()

    perform_files_coregistration(
        nifti_dir=args.nifti_dir, result_dir=args.result_dir, verbose=args.verbose
    )


def get_args():
    parser = argparse.ArgumentParser(
        description='Performs NIFTI files coregistration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--nifti-dir',
        type=str,
        required=True,
        help='Path to raw nifti files',
    )
    parser.add_argument(
        '--result-dir',
        type=str,
        required=True,
        help='Path to coregistrated nifti files',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        default=False,
        action='store_true',
        help='Path to coregistrated nifti files',
    )

    return parser.parse_args()


def perform_files_coregistration(
    nifti_dir: Union[str, Path], result_dir: Union[str, Path], verbose: bool = False
) -> None:
    file_paths = list(Path(nifti_dir).glob(pattern='*.nii.gz'))
    file_paths.sort()

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    assert len(file_paths) >= 1, 'No NIFTI files in dir'

    ct_first = file_paths[0]
    template_ct = ants.image_read(filename=str(ct_first))

    print(ct_first)

    for curr_ct_path in tqdm(
        file_paths[1:], postfix='Coregistration...', disable=not verbose
    ):
        ct_to_transform = ants.image_read(filename=str(curr_ct_path))

        ct_transformed = perform_registration(
            template_ct=template_ct, ct_to_transform=ct_to_transform
        )
        ct_transformed_path = result_dir.joinpath(curr_ct_path.name)

        ants.image_write(image=ct_transformed, filename=str(ct_transformed_path))


def perform_registration(
    template_ct: ants.ANTsImage, ct_to_transform: ants.ANTsImage
) -> ants.ANTsImage:
    ct_transformed = ants.registration(
        fixed=template_ct,
        moving=ct_to_transform,
        type_of_transform='Affine',
        verbose=False,
    )
    ct_transformed = ct_transformed['warpedmovout']

    return ct_transformed
