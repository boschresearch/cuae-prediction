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

import numpy as np
import torch

from learning.policies import Policy, PolicyPin
from lib.base_classes import FULL_EGO_ATTRIBUTES, STATE_EGO_ATTRIBUTES


@dataclass
class KinematicBicycleModelConfig:
    inputs: List[PolicyPin]
    outputs: List[PolicyPin]
    attributes: List[str]
    sample_time: float = 0.1

    def __post_init__(self):
        self.inputs = [
            inpt if isinstance(inpt, PolicyPin) else PolicyPin(**inpt)
            for inpt in self.inputs
        ]
        self.outputs = [
            outpt if isinstance(outpt, PolicyPin) else PolicyPin(**outpt)
            for outpt in self.outputs
        ]


class KinematicBicycleModel(Policy):
    def __init__(
        self,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        attributes: List[str],
        sample_time: float = 0.1,
    ):
        """
        Policy wrapper for a forward kinematic bicycle model.
        """
        assert (
            len(inputs) == 2
        ), "Only actions and ego_state_features inputs are supported"
        assert len(outputs) == 1, "Only one output is supported"

        self.actions_input: PolicyPin = None
        self.ego_state_features_input: PolicyPin = None
        self.ego_states_output: PolicyPin = None

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            sample_time=sample_time,
            attributes=attributes,
        )

        if not self.actions_input.info.shape[-1] == 2:
            raise ValueError(
                f"input shape = {self.actions_input.info.shape} must be of the form [*, 2]"
            )
        if not self.ego_states_output.info.shape[1] == 4:
            raise ValueError(
                f"output_shape = {self.ego_states_output.info.shape} must be of the form [*, 4]"
            )

        self.dt = sample_time
        self.attributes = attributes

    def _check_attributes(self) -> None:
        # check attributes to ensure correct indexing
        assert (
            self.attributes == FULL_EGO_ATTRIBUTES
            or self.attributes == STATE_EGO_ATTRIBUTES
        ), f"Attributes are not ordered right. Only the orderings {FULL_EGO_ATTRIBUTES} and {STATE_EGO_ATTRIBUTES} are supported"

    def _check_ego_features_shape(self, ego_features: torch.Tensor) -> None:
        # check ego-feature input to ensure correct indexing
        assert ego_features.shape[1] == len(
            self.attributes
        ), f"Ego-features num_attributes: {ego_features.shape[1]} has to match with selected attributes {self.attributes}"

    def _get_ego_state(self, ego_features: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :returns
            ego_state=[x, y, yaw, v]: shape: [batch_size, 4]
        """

        self._check_attributes()
        self._check_ego_features_shape(ego_features)

        if self.attributes == FULL_EGO_ATTRIBUTES:
            ego_state = torch.stack(
                (
                    ego_features[:, 0],  # x
                    ego_features[:, 1],  # y
                    ego_features[:, 2],  # yaw
                    torch.sqrt(
                        ego_features[:, 3] ** 2
                        + ego_features[:, 4] ** 2  # v = sqrt(v_x**2+ v_y**2)
                    ),
                ),
                dim=1,
            )
        elif self.attributes == STATE_EGO_ATTRIBUTES:
            ego_state = ego_features[:, :4]

        return ego_state

    def _get_ego_params(self, ego_features: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :returns
            ego_params=[length, width]: shape: [batch_size, 2]
        """

        self._check_attributes()
        self._check_ego_features_shape(ego_features)

        ego_params = ego_features[:, -2:]  # length, width

        return ego_params

    @staticmethod
    def get_state_ego_attributes(full_ego_attributes: torch.Tensor) -> torch.Tensor:
        """
        :param:
            full_ego_attributes=[x, y, yaw, v_x, v_y, a_x, a_y, length, width]: shape: [batch_size, 9]
        :returns
            state_ego_attributes=[x, y, yaw, v, length, width]: shape: [batch_size, 6]
        """

        assert full_ego_attributes.shape[1] == len(
            FULL_EGO_ATTRIBUTES
        ), f"full_ego_attributes must have {len(FULL_EGO_ATTRIBUTES)} attributes: {FULL_EGO_ATTRIBUTES}"

        state_ego_attributes = torch.stack(
            (
                full_ego_attributes[:, 0],  # x
                full_ego_attributes[:, 1],  # y
                full_ego_attributes[:, 2],  # yaw
                torch.sqrt(
                    full_ego_attributes[:, 3] ** 2
                    + full_ego_attributes[:, 4] ** 2  # v = sqrt(v_x**2+ v_y**2)
                ),
                full_ego_attributes[:, 7],  # length
                full_ego_attributes[:, 8],  # width
            ),
            dim=1,
        )

        return state_ego_attributes

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Iterate over each time-step and apply the forward kinematic bicycle model w.r.t. a state and an action.

        :param x: dict with:
                chronologically ordered actions over a number of timesteps, shape [batch_size, num_timesteps, 2],
                and ego_state_features at the most recent timestep, shape: [batch_size, num_ego_attributes]
        :return chronologically ordered ego_states: [x, y, yaw, v], shape: [batch_size, num_timesteps, 4]
        """
        actions = x[self.actions_input.key]
        future_state = self._get_ego_state(x[self.ego_state_features_input.key])

        params = self._get_ego_params(x[self.ego_state_features_input.key])

        # build tensor to store output
        batch_size = actions.shape[0]
        num_timesteps = actions.shape[1]
        states = torch.empty(
            (batch_size, num_timesteps, 4), device=actions.device
        )  # [x, y, yaw, v]

        # iteratively apply forward bicycle model
        for k in range(num_timesteps):
            future_state = self.bicycle_1st_order_forward_update(
                future_state, actions[:, k, :], params
            )
            states[:, k, :] = future_state

        return {self.ego_states_output.key: states}

    def bicycle_1st_order_forward_update(
        self, state_k: torch.Tensor, action_k: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a first order update on the state variable [x, y, yaw, v] at timestep k, w.r.t the actions [acc, steer] at timestep k.
        The state at timestep k+1 is computed via:
            state_k+1 = state_k + dot{state}_k * dt,
        where dot{state}_k incorporates the actions.

        :param state_k: state [x, y, yaw, v] at timestep k, shape [batch_size, 4]
        :param action_k: action variable at timestep k, shape [batch_size, 2]
        :return state_k+1: state [x, y, yaw, v] at timestep k+1, shape [batch_size, 4]
        """

        yaw = state_k[:, 2]
        v = state_k[:, 3]

        length = params[:, 0]

        a = action_k[:, 0]  # acceleration
        steer = action_k[:, 1]  # steering angle

        return state_k + self.dt * self.bicycle_state_dot(yaw, v, length, a, steer)

    def bicycle_state_dot(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        length: torch.Tensor,
        a: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the dot{state} according to the bicycle model while applying the actions. Source: Eq. (4) in https://arxiv.org/pdf/2109.10024.pdf

        :param theta,v,length,a,delta: states/param/actions, shape: [batch_size]
        :return state_dot: [x_dot, y_dot, theta_dot, v_dot], shape [batch_size, 4]
        """
        lf = length / 2.0  # distance to front axle
        lr = length / 2.0  # distance to rear axle

        beta = torch.atan2(
            torch.tan(delta) * lr, (lr + lf)
        )  # angle between v and and theta

        x_dot = v * torch.cos(theta + beta)
        y_dot = v * torch.sin(theta + beta)
        theta_dot = torch.sin(beta) * v / lr
        v_dot = a

        return torch.stack((x_dot, y_dot, theta_dot, v_dot), dim=1)

    @staticmethod
    def compute_actions(
        state_k, state_kplus1, params, sample_time: float
    ) -> np.ndarray:
        """
        Computes the actions [a, delta] that induced the change in the state from state_k to state_kplus1, given parameters and sample time. Source: Eq. (3,4) in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8814080
        :param state_k, state_kplus1: states [x,y,yaw,v] at consecutive time-steps, shape: [4]
        :param params: parameters [length, width], shape: [2]
        :param sample_time: sample time between two time-steps
        :return actions: acceleration and steering angle
        """
        yaw_k = state_k[2]
        yaw_kplus1 = state_kplus1[2]
        v_k = state_k[3]
        v_kplus1 = state_kplus1[3]

        lr = params[0] / 2  # distance to rear axle
        lf = params[0] / 2  # distance to front axle

        # compute acceleration a
        a_k = (v_kplus1 - v_k) / sample_time

        # compute steering angle delta
        v_avg = (v_kplus1 + v_k) / 2
        yaw_dot = (yaw_kplus1 - yaw_k) / sample_time
        diff_sq = (v_avg / (yaw_dot + 1e-6)) ** 2 - lr**2
        delta_k = 0.0  # standstill steering angle
        if diff_sq > 0 and np.abs(v_avg) > 1e-2:  # no standstill if avg. vel. > 1e-2
            delta_k = np.sign(yaw_dot / v_avg) * np.arctan2((lr + lf), np.sqrt(diff_sq))

        return np.array([a_k, delta_k])

    @classmethod
    def from_config(cls, cf: dict):
        config = KinematicBicycleModelConfig(**cf)
        return cls(
            config.inputs,
            config.outputs,
            config.attributes,
            config.sample_time,
        )
