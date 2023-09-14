""" Conditional Unscented Autoencoder.
Copyright (c) 2024 Robert Bosch GmbH
@author: Faris Janjos
@author: Marcel Hallgarten
@author: Anthony Knittel
@author: Maxim Dolgov

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from lib.data_handling import Dataset, DatasetPathConfig
from lib.utils import class_from_path


@dataclass
class SamplingDataset:
    dataset: torch.utils.data.ConcatDataset
    sample_weights: np.ndarray


@dataclass
class LearningRateSchedulerConfig:
    class StepType(Enum):
        EPOCH = 1
        ITERATION = 2
        PLATEAU = 3

    path_to_class: str
    config: dict
    step_type: StepType = None

    def __post_init__(self):
        if (
            self.path_to_class == "torch.optim.lr_scheduler.ExponentialLR"
            or self.path_to_class == "torch.optim.lr_scheduler.StepLR"
            or self.path_to_class == "torch.optim.lr_scheduler.MultiStepLR"
        ):
            self.step_type = self.StepType.EPOCH
        elif self.path_to_class == "torch.optim.lr_scheduler.CyclicLR":
            self.step_type = self.StepType.ITERATION
        elif self.path_to_class == "torch.optim.lr_scheduler.ReduceLROnPlateau":
            self.step_type = self.StepType.PLATEAU
        else:
            raise ValueError(f"Unknown LR scheduler class: {self.path_to_class}")


def scheduler_from_config(
    config_dict: dict,
    optim: torch.optim.Optimizer,
    train_data_loader: torch.utils.data.DataLoader,
) -> Tuple[torch.optim.lr_scheduler._LRScheduler, str]:
    if config_dict == {}:
        return None, None
    scheduler_config = LearningRateSchedulerConfig(**config_dict)
    if (
        scheduler_config.step_type == LearningRateSchedulerConfig.StepType.EPOCH
        or scheduler_config.step_type == LearningRateSchedulerConfig.StepType.PLATEAU
    ):
        scheduler = class_from_path(scheduler_config.path_to_class)(
            optim, **scheduler_config.config
        )
    elif scheduler_config.step_type == LearningRateSchedulerConfig.StepType.ITERATION:
        scheduler = class_from_path(scheduler_config.path_to_class)(
            optim,
            step_size_up=len(
                train_data_loader
            ),  # 1 cycle in 2 epochs (should be in 2-10)
            cycle_momentum=False,
            **scheduler_config.config,
        )
    return scheduler, scheduler_config.step_type


def make_dataset(
    dataset_paths: List[DatasetPathConfig],
    dataset_name: Optional[str],
) -> SamplingDataset:
    """
    Generates a dataset for each specified dataset path and concatenates them to a single-dataset.
        Inside a dataset path all hdf5 files containing the dataset name are used for the dataset.
    """
    datasets: List[Dataset] = []
    datasets_weights: List[int] = []

    for dataset_path in dataset_paths:
        # aggregate all .h5 files in the folder: "data_{dataset_name}*.h5"
        dataset_names = []

        name = dataset_name if dataset_name is not None else dataset_path.name

        for file_name in sorted(os.listdir(dataset_path.path)):
            if file_name.startswith(f"data_{name}") and file_name.endswith(".h5"):
                dataset_names.append(file_name[len("data_") : -len(".h5")])

        datasets_from_paths = [
            Dataset(
                dataset_path=dataset_path.path,
                dataset_name=name,
            )
            for name in dataset_names
        ]

        datasets.extend(datasets_from_paths)
        datasets_weights.extend(
            [
                [dataset_path.sampling_weight] * len(dataset)
                for dataset in datasets_from_paths
            ]
        )

    concatenated_dataset = torch.utils.data.ConcatDataset(datasets)
    datasets_weights = np.concatenate(
        [np.asarray(sublist) for sublist in datasets_weights]
    )

    dataset_features_labels_names = {
        tuple(dataset.features_labels_names) for dataset in datasets
    }
    assert (
        len(dataset_features_labels_names) == 1
    ), "All provided datasets have to have the same features names."

    # TODO: add a check that that features/labels generators of provided datasets are identical as well

    return SamplingDataset(concatenated_dataset, datasets_weights)


def make_datasets(
    dataset_paths: List[DatasetPathConfig],
) -> Tuple[SamplingDataset, SamplingDataset]:
    train_set = make_dataset(
        dataset_paths=dataset_paths,
        dataset_name="train",
    )
    dev_set = make_dataset(
        dataset_paths=dataset_paths,
        dataset_name="dev",
    )

    return train_set, dev_set


def unpack_batch(
    batch: Dict[str, torch.Tensor],
    use_gpu: bool,
) -> Dict[str, torch.Tensor]:
    """
    Utils function to push a batch onto GPU if possible.
    It returns two dicts for features and labels.
    """
    device = torch.device("cuda" if use_gpu else "cpu")
    batch_size = len(next(iter(batch.values())))

    features_labels = TensorDict(
        batch,
        batch_size=batch_size,
    )

    return features_labels.to(device)
