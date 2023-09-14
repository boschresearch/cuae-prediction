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

import torch

from learning.data_curator_config import DataCuratorClassConfig, DataCuratorConfig
from learning.learning_utils import make_datasets
from lib.random import seed_worker
from lib.utils import get_path_to_class


class DataCurator:
    """
    Manages the data during training.
    """

    def __init__(self, config: DataCuratorConfig):
        self.train_set = None
        self.dev_set = None
        self.train_data_loader = None
        self.dev_data_loader = None

        self._config = config
        self._init_data_loaders()

    def _init_data_loaders(
        self,
    ):
        self.train_set, self.dev_set = make_datasets(
            dataset_paths=self._config.dataset_paths,
        )

        if not self._config.sample_with_replacement:
            unique_sampling_weights = set(
                dataset_path.sampling_weight
                for dataset_path in self._config.dataset_paths
            )
            assert (
                len(unique_sampling_weights) == 1
            ), "When sampling without replacement, all sampling weights have to be equal"

        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.train_set.sample_weights,
            num_samples=int(
                len(self.train_set.dataset) * self._config.random_train_subset
            ),
            replacement=self._config.sample_with_replacement,
        )
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_set.dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_dataloader_workers,
            worker_init_fn=seed_worker,
            sampler=train_sampler,
        )

        dev_sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.dev_set.sample_weights,
            num_samples=int(len(self.dev_set.dataset) * self._config.random_dev_subset),
            replacement=self._config.sample_with_replacement,
        )
        self.dev_data_loader = torch.utils.data.DataLoader(
            self.dev_set.dataset,
            batch_size=self._config.batch_size,
            num_workers=self._config.num_dataloader_workers,
            worker_init_fn=seed_worker,
            sampler=dev_sampler,
        )

    @classmethod
    def from_config(cls, config: DataCuratorConfig) -> "DataCurator":
        return cls(config=config)

    def to_config(self) -> DataCuratorClassConfig:
        return DataCuratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=self._config,
        )

    def update_datasets(self, **kwargs) -> None:
        """
        Intended to be called after a training epoch.
        To be implemented by child class.
        """
