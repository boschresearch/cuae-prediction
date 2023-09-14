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

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

AGGREGATION_FN_TYPE = Callable[
    [torch.Tensor, torch.Tensor, Union[int, float]], torch.Tensor
]


class ExponentialMovingAverage:
    """
    Exponential moving average.
    The effective number of steps of the moving average is 1/(1-discount).
    """

    def __init__(self, discount: float = 0.8):
        self.discount = discount
        self.value = 0.0
        self.norm = 0.0

    def update(self, x: float) -> float:
        self.value = self.discount * self.value + (1.0 - self.discount) * x
        self.norm = self.discount * self.norm + (1.0 - self.discount)
        return self.value / self.norm


@torch.jit.script
def prepare_data_for_minDE(
    policy_output: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    policy_output_waypoints_key: str,
    label_waypoints_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    From raw input to any minDE loss function,
    extract candidate trajectories + ground truth trajectory.

    Returns tuple
    - candidate trajectories is a tensor of shape [batch_size, num_trajectories, num_timesteps, 2=(xy)]
    - gt trajectory is a tensor of shape [batch_size, 1, num_timesteps, 2=(xy)]
    """
    policy_waypoints = policy_output[policy_output_waypoints_key]

    if len(policy_waypoints.shape) == 3:
        candidate_trajectories = policy_waypoints[:, np.newaxis, :, :]
    elif len(policy_waypoints.shape) == 4:
        candidate_trajectories = policy_waypoints
    else:
        raise ValueError(
            f"Policy waypoints do not have expected shape: {policy_waypoints.shape}. "
            + "Expected (batch_size, num_timesteps, 2) or (batch_size, num_trajectories, num_timesteps, 2)"
        )

    gt_trajectory = labels[label_waypoints_key][:, np.newaxis, :, :]

    assert (
        candidate_trajectories.shape[0] == gt_trajectory.shape[0]
    ), "batch size for predicted trajectories and ground truth trajectories must match"
    assert (
        candidate_trajectories.shape[2:] == gt_trajectory.shape[2:]
    ), "number of waypoints for predicted trajectories and ground truth trajectories must match"

    return candidate_trajectories, gt_trajectory


@torch.jit.script
def prepare_data_for_DE(
    policy_output: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    policy_output_waypoints_key: str,
    label_waypoints_key: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Used as input to any xDE loss function **without** min operation. Extracts (multi-modal) candidate trajectory and (multi-modal) label trajectory. Both have to contain the same number of modes. If none, mode dimension is added.

    Args:
        policy_output[0]: trajectory tensor of shape [batch_size, num_trajectories, num_timesteps, dim] or [batch_size, num_timesteps, dim]
        labels[0] labels tensor of shape [batch_size, num_trajectories, num_timesteps, dim] or [batch_size, num_timesteps, dim]

    Returns:
        tuple of the two trajectories used in DE computation
    """
    candidate_trajectory = policy_output[policy_output_waypoints_key]
    assert (
        len(candidate_trajectory.shape) == 3 or len(candidate_trajectory.shape) == 4
    ), f"Policy waypoints trajectory is of shape: {candidate_trajectory.shape}. Expected (batch_size, num_timesteps, 2) or (batch_size, num_trajectories, num_timesteps, 2)"

    label_trajectory = labels[label_waypoints_key]
    assert (
        label_trajectory.shape == candidate_trajectory.shape
    ), f"Label trajectory is of shape: {label_trajectory.shape}. Expected: f{candidate_trajectory.shape}"

    if len(candidate_trajectory.shape) == 3:
        candidate_trajectory = candidate_trajectory.clone()[:, np.newaxis, :, :]
        label_trajectory = label_trajectory.clone()[:, np.newaxis, :, :]

    return candidate_trajectory, label_trajectory


@torch.jit.script
def calculate_ADE(
    candidate_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    order: Union[int, float],
) -> torch.Tensor:
    """
    Return ADE (average displacement error).

    Inputs:
    - candidate_trajectories: shape == (batch_size, num_trajectories, num_timesteps, 2=(xy))
    - gt_trajectory: shape == (batch_size, 1, num_timesteps, 2=(xy))
    - order: the order of the vector norm (2 for Euclidean)
    """
    DE = torch.linalg.vector_norm(
        candidate_trajectories - gt_trajectory, ord=order, dim=-1
    )
    ADE = torch.mean(DE, dim=2)
    return ADE


@torch.jit.script
def calculate_minADE(
    candidate_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    order: Union[int, float],
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return minADE (average displacement error of the best mode) and best mode indices.

    Inputs:
    - candidate_trajectories: shape == (batch_size, num_trajectories, num_timesteps, 2=(xy))
    - gt_trajectory: shape == (batch_size, 1, num_timesteps, 2=(xy))
    - order: the order of the vector norm (2 for Euclidean)
    - mask: shape == (batch_size, num_trajectories).
      True means that trajectory and probability are valid.
    """
    ADE = calculate_ADE(
        candidate_trajectories=candidate_trajectories,
        gt_trajectory=gt_trajectory,
        order=order,
    )
    if mask is not None:
        ADE[~mask] = torch.inf
    minADE, minADE_ixs = torch.min(ADE, dim=1)
    return minADE, minADE_ixs


@torch.jit.script
def calculate_FDE(
    candidate_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    order: Union[int, float],
) -> torch.Tensor:
    """
    Return FDE (final displacement error).

    Inputs:
    - candidate_trajectories: shape == (batch_size, num_trajectories, num_timesteps, 2=(xy))
    - gt_trajectory: shape == (batch_size, 1, num_timesteps, 2=(xy))
    - order: the order of the vector norm (2 for Euclidean)
    """
    FDE = torch.linalg.vector_norm(
        candidate_trajectories[:, :, -1, :] - gt_trajectory[:, :, -1, :],
        ord=order,
        dim=-1,
    )
    return FDE


@torch.jit.script
def calculate_minFDE(
    candidate_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    order: Union[int, float],
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return minFDE (final displacement error of the best mode) and best mode indices.

    Inputs:
    - candidate_trajectories: shape == (batch_size, num_trajectories, num_timesteps, 2=(xy))
    - gt_trajectory: shape == (batch_size, 1, num_timesteps, 2=(xy))
    - order: the order of the vector norm (2 for Euclidean)
    - mask: shape == (batch_size, num_trajectories).
      True means that trajectory and probability are valid.
    """
    FDE = calculate_FDE(
        candidate_trajectories=candidate_trajectories,
        gt_trajectory=gt_trajectory,
        order=order,
    )
    if mask is not None:
        FDE[~mask] = torch.inf
    minFDE, minFDE_ixs = torch.min(FDE, dim=1)
    return minFDE, minFDE_ixs


def calculate_minDE(
    candidate_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
    order: Union[str, int, float],
    aggregation_fn: AGGREGATION_FN_TYPE,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return minADE (average displacement error of the best mode) or
    minFDE (final displacement error of the best mode),
    as well as best mode indices.

    Inputs:
    - candidate_trajectories: shape == (batch_size, num_trajectories, num_timesteps, 2=(xy))
    - gt_trajectory: shape == (batch_size, 1, num_timesteps, 2=(xy))
    - order: the order of the vector norm (2 for Euclidean)
    - aggregation_fn: ADE or FDE
    - mask: shape == (batch_size, num_trajectories).
      True means that trajectory and probability are valid.
    """
    xDE = aggregation_fn(
        candidate_trajectories=candidate_trajectories,
        gt_trajectory=gt_trajectory,
        order=order,
    )
    if mask is not None:
        xDE[~mask] = torch.inf
    minxDE, minxDE_ixs = torch.min(xDE, dim=1)
    return minxDE, minxDE_ixs
