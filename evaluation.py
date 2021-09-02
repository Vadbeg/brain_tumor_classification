"""Module with evaluation"""

import glob
from typing import Dict

import torch

from brain_tumor_classification.modules.data.datasets.eval_dataset import (
    BrainDicomEvalDataset,
)
from brain_tumor_classification.modules.evaluation.model_evaluator import ModelEvaluator
from brain_tumor_classification.modules.model.resnet import generate_resnet_model


def rename_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}

    for layer_name, layer_weights in state_dict.items():
        layer_name = layer_name.replace('model.', '')

        new_state_dict[layer_name] = layer_weights

    return new_state_dict


if __name__ == '__main__':
    pattern = '/home/vadbeg/Data_SSD/Kaggle/rsna-miccai-brain-tumor-radiogenomic-classification/test/**/FLAIR'
    model_path = (
        'logs/brain-classification/ze3flpcm/checkpoints/epoch=28-step=3972.ckpt'
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    list_of_dicom_folder_paths = list(glob.glob(pathname=pattern, recursive=True))

    dataset = BrainDicomEvalDataset(
        list_of_paths=list_of_dicom_folder_paths,
    )

    model_meta = torch.load(model_path, map_location='cpu')

    model_state_dict = model_meta['state_dict']
    model_state_dict = rename_keys(state_dict=model_state_dict)

    model_depth = 10
    num_input_channels = 1
    num_classes = 2

    resnet = generate_resnet_model(
        model_depth=model_depth,
        n_input_channels=num_input_channels,
        n_classes=num_classes,
    )
    resnet.load_state_dict(model_state_dict)
    resnet.to(device=device)

    model_evaluator = ModelEvaluator(model=resnet, dataset=dataset, device=device)

    model_evaluator.eval(save_to_csv=True)
