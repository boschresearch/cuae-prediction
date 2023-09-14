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

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import numpy as np

from features_and_labels.generators_base import (
    GeneratorOutputInfo,
    MultiOutputGenerator,
    MultiOutputGeneratorClassConfig,
    punch_past_and_future_frames,
)
from features_and_labels.mission import mission_from_static_rollout_info
from lib.rollout import Rollout
from lib.utils import ensure_init_type, instantiate_from_config


@dataclass
class FeaturesLabelsGeneratorConfig:
    """
    Stores configuration for FeaturesLabelsGenerator that consists of
    configs for the batched features generators, batched labels generators,
    """

    features_labels_generator_config: MultiOutputGeneratorClassConfig

    def __post_init__(self):
        self.features_labels_generator_config = ensure_init_type(
            self.features_labels_generator_config, MultiOutputGeneratorClassConfig
        )


class FeaturesLabelsGenerator:
    """
    Extracts data from rollout, by calling a list of FeaturesGenerators and a list of LabelsGenerators.
    Important properties are len_tail and len_head. They specify, how many past observations and how many future observations
    relative to the current observation are needed.

                                 past_frames                         future_frames
                 --------------------------------------------   -------------------------
        Frames:  f[-5]   f[-4]   f[-3]   f[-2]   f[-1]   f[0]   f[1]   f[2]   f[3]   f[4]
                 -------------------------------------          -------------------------
                              len_tail                                   len_head
    """

    def __init__(
        self,
        features_labels_generator_config: MultiOutputGeneratorClassConfig,
    ) -> None:
        self.features_labels_generator: MultiOutputGenerator = instantiate_from_config(
            features_labels_generator_config
        )

    @property
    def len_tail(self):
        return self.features_labels_generator.len_tail

    @property
    def len_head(self):
        return self.features_labels_generator.len_head

    @property
    def features_labels_infos(self) -> Dict[str, GeneratorOutputInfo]:
        return self.features_labels_generator.outputs_infos

    def __call__(
        self, rollout: Rollout, rollout_stride: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Generates batched features and labels from a rollout.
        The rollout is traversed with provided stride.

        Args:
            rollout (Rollout): rollout from which the features and labels are generated
            stride (int): stride with which the rollout is traversed

        Returns:
            dict: Features for batching
            dict: Labels for batching
        """
        obs_data = rollout.frames

        mission = mission_from_static_rollout_info(rollout.static_info)
        self.features_labels_generator.reset(
            mission=mission, ego_id=rollout.static_info.ego_id
        )

        features_labels = defaultdict(list)
        for idx_current_obs in range(
            self.len_tail,
            len(obs_data) - self.len_head,
            rollout_stride,
        ):

            # punch frames for features generation
            (past_frames, future_frames,) = punch_past_and_future_frames(
                obs_data,
                idx_current_obs,
                self.features_labels_generator.len_tail,
                self.features_labels_generator.len_head,
            )

            # TODO: implement samples perturbation

            # generate features using each features generator and write to features dict
            curr_features_labels = self.features_labels_generator(
                past_frames=past_frames,
                future_frames=future_frames,
            )
            for k, v in curr_features_labels.items():
                features_labels[k].append(v)

        # stack features due to additional batch dimension (batch size 1)
        return {
            key: np.concatenate(val, axis=0) for key, val in features_labels.items()
        }

    @classmethod
    def from_config(cls, config: FeaturesLabelsGeneratorConfig):
        return cls(
            config.features_labels_generator_config,
        )

    def to_config(self) -> FeaturesLabelsGeneratorConfig:
        return FeaturesLabelsGeneratorConfig(
            features_labels_generator_config=self.features_labels_generator.to_config(),
        )

    def trim_tail_and_head(self, rollout: Rollout) -> Rollout:
        """
        Slice rollout such that its steps correspond to those returned by __call__.
        """
        return rollout[
            slice(
                self.len_tail,
                -self.len_head if self.len_head > 0 else None,
            )
        ]
