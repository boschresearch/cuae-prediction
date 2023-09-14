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

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from learning.loss_base import LossBase, LossBaseConfig, loss_base_from_config
from lib.utils import ensure_init_type


@dataclass
class LossWrapperConfig:
    name: str
    loss_function: LossBaseConfig

    def __post_init__(self):
        self.loss_function = ensure_init_type(self.loss_function, LossBaseConfig)


class LossWrapper:
    """
    Wrapper around loss function that keeps track of average loss.
    """

    def __init__(self, name: str, loss_fn: LossBase):
        self.loss_fn = loss_fn
        self.name = name
        self.acc_batch_loss = 0.0
        self.acc_batch_cnt = 0

    def __call__(
        self,
        policy_output: Dict[str, Tensor],
        labels: Dict[str, Tensor],
        return_sample_loss: bool = False,
    ) -> Tensor:
        sample_loss = self.loss_fn(policy_output=policy_output, labels=labels)
        batch_loss = torch.mean(sample_loss)  # mean over batch_dim
        self.acc_batch_loss += batch_loss.detach().cpu().numpy()
        self.acc_batch_cnt += 1

        if return_sample_loss:
            return sample_loss
        else:
            return batch_loss

    def _reset(self):
        self.acc_batch_loss = 0.0
        self.acc_batch_cnt = 0

    def get_average_loss(self):
        """
        Return averaged accumulated loss, reset loss.
        """
        average_loss = self.acc_batch_loss / self.acc_batch_cnt
        self._reset()
        return average_loss

    @classmethod
    def from_dict(cls, config: dict):
        return cls(
            name=config["name"], loss_fn=loss_base_from_config(config["loss_function"])
        )


class LossWrapperContainer:
    """
    Container for multiple loss wrappers
    """

    def __init__(
        self, loss_wrappers: List[LossWrapper], loss_weights: List[float]
    ) -> None:
        assert len(loss_wrappers) == len(loss_weights)
        self.loss_wrappers = loss_wrappers
        self.loss_weights = loss_weights

    def __call__(self, *args, return_sample_loss: bool = False, **kwargs) -> Tensor:
        loss = 0
        for loss_wrapper, weight in zip(self.loss_wrappers, self.loss_weights):
            loss += weight * loss_wrapper(
                return_sample_loss=return_sample_loss, *args, **kwargs
            )
        return loss

    def get_average_loss(self) -> Tuple[float, Dict]:
        avg_loss = 0
        loss_components = dict()
        for loss_wrapper, weight in zip(self.loss_wrappers, self.loss_weights):
            single_loss = loss_wrapper.get_average_loss()
            avg_loss += weight * single_loss
            loss_components[loss_wrapper.name] = single_loss
        return avg_loss, loss_components

    @classmethod
    def from_dict(cls, config: List[dict], use_gpu: bool = False):
        container = []
        weights = []
        for cfg in config:
            cfg.update({"use_gpu": use_gpu})
            weights.append(cfg.get("weight", 1))
            container.append(LossWrapper.from_dict(cfg))
        return cls(container, weights)

    @property
    def label_names(self) -> List[str]:
        return list(
            set(
                [
                    label_name
                    for loss_wrapper in self.loss_wrappers
                    for label_name in loss_wrapper.loss_fn.label_names
                ]
            )
        )


##################################################################################################################################
#                                                 LOSS WRAPPER CONFIG HANDLING                                                   #
##################################################################################################################################


def loss_from_config(
    loss_config: List[dict], use_gpu: bool = False
) -> LossWrapperContainer:
    return LossWrapperContainer.from_dict(loss_config, use_gpu)
