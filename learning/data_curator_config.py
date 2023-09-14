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
from dataclasses import dataclass, field
from typing import List

from lib.data_handling import DatasetPathConfig
from lib.utils import ensure_init_type


@dataclass
class DataCuratorConfig:
    batch_size: int
    datasets_root: str
    dataset_paths: List[DatasetPathConfig]
    random_train_subset: float  # in (0, 1]
    random_dev_subset: float  # in (0, 1]
    num_dataloader_workers: int
    sample_with_replacement: bool = True
    additional_parameters: dict = field(
        default_factory=dict
    )  # specific to each data curator

    def __post_init__(self):
        self._check_dataset_format_(
            datasets_root=self.datasets_root, dataset_paths=self.dataset_paths
        )
        self._check_parameter_range(
            random_train_subset=self.random_train_subset,
            random_dev_subset=self.random_dev_subset,
        )

    @staticmethod
    def _check_dataset_format_(
        datasets_root: str, dataset_paths: List[DatasetPathConfig]
    ):
        """Raise if invalid, fix format to List[DatasetPathConfig]"""
        if not isinstance(dataset_paths, list):
            raise ValueError(f"Invalid entry dataset_paths = {dataset_paths}")

        for ix, d in enumerate(dataset_paths):
            dataset_paths[ix] = ensure_init_type(d, DatasetPathConfig)

        for path in dataset_paths:
            path.path = os.path.abspath(os.path.join(datasets_root, path.path))

    @staticmethod
    def _check_parameter_range(random_train_subset: float, random_dev_subset: float):
        if not 0 < random_train_subset <= 1:
            raise ValueError(
                f"Invalid entry random_train_subset={random_train_subset} not in (0, 1]"
            )
        if not 0 < random_dev_subset <= 1:
            raise ValueError(
                f"Invalid entry random_dev_subset={random_dev_subset} not in (0, 1]"
            )


@dataclass
class DataCuratorClassConfig:
    path_to_class: str
    config: DataCuratorConfig

    def __post_init__(self):
        self.config = ensure_init_type(self.config, DataCuratorConfig)
