"""Module with utilities for dataset"""

import numpy as np
from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityRanged


def get_load_transforms(
    img_key: str,
    original_min: float = 0.0,
    original_max: float = 200.0,
    res_min: float = 0.0,
    res_max: float = 1.0,
) -> Compose:
    load_transforms = Compose(
        [
            LoadImaged(keys=[img_key], dtype=np.float32),
            AddChanneld(keys=[img_key]),
            ScaleIntensityRanged(
                keys=[img_key],
                a_min=original_min,
                a_max=original_max,
                b_min=res_min,
                b_max=res_max,
                clip=True,
            ),
        ]
    )

    return load_transforms
