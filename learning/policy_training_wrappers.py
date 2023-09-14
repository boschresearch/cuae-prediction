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

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

from features_and_labels.generators_base import MultiOutputGeneratorClassConfig
from learning.data_curator_config import DataCuratorClassConfig, DataCuratorConfig
from learning.learning_utils import unpack_batch
from learning.loss_base import loss_base_from_config
from learning.loss_wrapper import loss_from_config
from learning.policies import PolicyClassConfigBase, policy_from_policy_config
from lib.utils import ensure_init_type, get_path_to_class, instantiate_from_config


@dataclass
class PolicyTrainingWrapperConfig:
    data_curator_config: DataCuratorClassConfig
    policy_config: Union[PolicyClassConfigBase, dict]
    params_from_dataset: dict
    loss_config: list
    additional_dev_loss_config: Optional[list]
    additional_train_loss_config: Optional[list]
    use_gpu: bool

    def __post_init__(self):
        self.data_curator_config = ensure_init_type(
            self.data_curator_config, DataCuratorClassConfig
        )
        # do not touch policy_config, because some initialization is handled within PolicyTrainingWrapper

        # Dicts are a cleaner way to define losses,
        # because the key (name) is not a member of the loss itself.
        # TODO refactor towards dict
        if isinstance(self.loss_config, dict):
            self.loss_config = [
                self.loss_config[key] for key in sorted(self.loss_config.keys())
            ]
        if isinstance(self.additional_dev_loss_config, dict):
            self.additional_dev_loss_config = [
                self.additional_dev_loss_config[key]
                for key in sorted(self.additional_dev_loss_config.keys())
            ]
        if isinstance(self.additional_train_loss_config, dict):
            self.additional_train_loss_config = [
                self.additional_train_loss_config[key]
                for key in sorted(self.additional_train_loss_config.keys())
            ]


@dataclass
class PolicyTrainingWrapperGridDiffSimConfig(PolicyTrainingWrapperConfig):
    sim_stride: int
    num_sim_frames: int
    num_ego_channels: int
    base_grid_cf: dict


def get_matching_config_for_class(
    path_to_class: str,
) -> PolicyTrainingWrapperConfig:
    if path_to_class.endswith("PolicyTrainingWrapper"):
        return PolicyTrainingWrapperConfig


@dataclass
class PolicyTrainingWrapperClassConfig:
    path_to_class: str
    config: Union[PolicyTrainingWrapperConfig, PolicyTrainingWrapperGridDiffSimConfig]

    def __post_init__(self):
        self.config = ensure_init_type(
            self.config, get_matching_config_for_class(self.path_to_class)
        )


class PolicyTrainingWrapper:
    def __init__(
        self,
        data_curator_config: DataCuratorConfig,
        policy_config: PolicyClassConfigBase,
        params_from_dataset: dict,
        loss_config: Union[list, dict],
        use_gpu: bool,
        additional_train_loss_config: Optional[list] = None,
        additional_dev_loss_config: Optional[list] = None,
    ) -> None:
        self._config = PolicyTrainingWrapperConfig(
            data_curator_config=data_curator_config,
            policy_config=policy_config,
            params_from_dataset=params_from_dataset,
            loss_config=loss_config,
            additional_dev_loss_config=additional_dev_loss_config,
            additional_train_loss_config=additional_train_loss_config,
            use_gpu=use_gpu,
        )

        for key, params in params_from_dataset.items():
            if policy_config["config"][key] == "INFER FROM TRAINING WRAPPER":
                policy_config["config"][key] = params
        self.policy = policy_from_policy_config(policy_config)

        self.data_curator = instantiate_from_config(config=data_curator_config)
        self._init_loss(
            loss_config=self._config.loss_config,
            additional_train_loss_config=self._config.additional_train_loss_config,
            additional_dev_loss_config=self._config.additional_dev_loss_config,
        )

    def _init_loss(
        self,
        loss_config: dict,
        additional_train_loss_config: Optional[list],
        additional_dev_loss_config: Optional[list],
    ):
        train_loss_config = (
            loss_config
            if not additional_train_loss_config
            else loss_config + additional_train_loss_config
        )
        self.train_loss = loss_from_config(train_loss_config, self._config.use_gpu)

        dev_loss_config = (
            loss_config
            if not additional_dev_loss_config
            else loss_config + additional_dev_loss_config
        )
        self.dev_loss = loss_from_config(dev_loss_config, self._config.use_gpu)

    def training_step(self, batch):
        features_labels = unpack_batch(
            batch,
            self._config.use_gpu,
        )
        return self.train_loss(self.policy.forward(features_labels), features_labels)

    def validation_step(self, batch):
        features_labels = unpack_batch(batch, self._config.use_gpu)
        return self.dev_loss(  # noqa: F841
            self.policy.forward(features_labels), features_labels
        )

    def on_train_epoch_end(self, epoch: int):
        """
        Is called after each epoch.
        """
        self.policy.eval()
        self.data_curator.update_datasets(
            epoch=epoch, policy=self.policy, use_gpu=self._config.use_gpu
        )

    def get_required_features_labels_names(self) -> List[str]:
        """
        Get required features and labels names.

        Returns:
            - List of required features and labels names
        """
        required_features_labels_names = list(
            set(input_pin.key for input_pin in self.policy_config.config.inputs)
            | set(
                key
                for loss in self._config.loss_config
                for key in loss_base_from_config(loss["loss_function"]).label_names
            )
        )

        return required_features_labels_names

    def get_inference_features_generator_config(
        self, training_features_generator_config: MultiOutputGeneratorClassConfig
    ) -> MultiOutputGeneratorClassConfig:
        """
        Get features generator for the trained policy which will be used during inference.
        """
        inference_configs = deepcopy(training_features_generator_config)
        inference_configs.config.generators = [
            config
            for config in training_features_generator_config.config.generators
            if config.config.output_name in self.policy.input_keys
        ]
        return inference_configs

    @property
    def policy_config(self) -> PolicyClassConfigBase:
        return self.policy.to_config()

    @classmethod
    def from_config(
        cls,
        config: PolicyTrainingWrapperConfig,
    ):
        return cls(
            data_curator_config=config.data_curator_config,
            policy_config=config.policy_config,
            params_from_dataset=config.params_from_dataset,
            loss_config=config.loss_config,
            additional_train_loss_config=config.additional_train_loss_config,
            additional_dev_loss_config=config.additional_dev_loss_config,
            use_gpu=config.use_gpu,
        )

    def to_config(self) -> PolicyTrainingWrapperClassConfig:
        return PolicyTrainingWrapperClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=self._config,
        )


##################################################################################################################################
# CONFIG HANDLING
##################################################################################################################################


def policy_training_wrapper_from_config(
    config: Union[PolicyTrainingWrapperClassConfig, dict],
) -> PolicyTrainingWrapper:
    config = ensure_init_type(config, PolicyTrainingWrapperClassConfig)
    return instantiate_from_config(config)
