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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from learning.clustering_utils import clustering_function
from learning.loss_utils import (
    AGGREGATION_FN_TYPE,
    calculate_ADE,
    calculate_FDE,
    calculate_minDE,
    prepare_data_for_DE,
    prepare_data_for_minDE,
)
from lib.utils import class_from_path, ensure_init_type

###
# general loss functions
###


class AttributeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


@dataclass
class LossBaseConfig:
    path_to_class: str
    config: dict


class LossBase:
    """
    Base class for losses and metrics
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: contains at least keys for referencing policy outputs and labels
        """
        self._config = AttributeDict(config)

    def __call__(
        self, policy_output: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        """
        Computes loss from values in <policy_output> and <labels> referenced with keys
        that are specified in <self._config>
        """
        return self._loss_fn(policy_output=policy_output, labels=labels, **self._config)

    @property
    def label_names(self) -> List[str]:
        """
        List of keys that reference labels
        """
        raise NotImplementedError

    @property
    def output_names(self) -> List[str]:
        """
        List of keys that reference outputs
        """
        raise NotImplementedError

    @property
    def _loss_fn(self) -> Callable:
        """
        Wrapped base loss function, e.g. l1_loss, cvar_l2_loss, ...
        """
        raise NotImplementedError


###
# multi-modal waypoints / trajectory losses
###


def _minDE_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_key: str,
    output_label_key: str,
    aggregation_fn: AGGREGATION_FN_TYPE,
    order: Union[int, float] = 2,
):
    """
    minADE / minFDE (specified via aggregation_fn).
    If only one candidate trajectory is given, the metric for this trajectory is calculated.
    order = order of the vector norm (2 for Euclidean)
    """
    candidate_trajectories, gt_trajectory = prepare_data_for_minDE(
        policy_output=policy_output,
        labels=labels,
        policy_output_waypoints_key=output_key,
        label_waypoints_key=output_label_key,
    )

    minxDE, _ = calculate_minDE(
        candidate_trajectories,
        gt_trajectory,
        order=order,
        aggregation_fn=aggregation_fn,
    )
    return minxDE


def minADE_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_key: str,
    output_label_key: str,
    order: Union[int, float] = 2,
):
    """
    Calculates the minimum average displacement error. This is the temporal average displacement between the ground truth trajectory and the best candidate trajectory (->min).
    If only one candidate trajectory is given, the average displacement for this trajectory is calculated.
    order = order of the vector norm (2 for Euclidean)
    """
    return _minDE_loss(
        policy_output=policy_output,
        labels=labels,
        output_key=output_key,
        output_label_key=output_label_key,
        aggregation_fn=calculate_ADE,
        order=order,
    )


def minFDE_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_key: str,
    output_label_key: str,
    order: Union[str, int, float] = 2,
):
    """
    Calculates the minimum final displacement error. This is the displacement between the endpoints of ground truth trajectory and the best candidate trajectory (->min).
    If only one candidate trajectory is given, the average displacement for this trajectory is calculated.
    order = order of the vector norm (2 for Euclidean)
    """
    return _minDE_loss(
        policy_output=policy_output,
        labels=labels,
        output_key=output_key,
        output_label_key=output_label_key,
        aggregation_fn=calculate_FDE,
        order=order,
    )


@torch.jit.script
def ADE_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_key: str,
    output_label_key: str,
    order: Union[int, float] = 2,
) -> Tensor:
    """
    Wrapper for the Average Displacement Error (ADE) calculation. This is the average displacement between two (multi-modal) trajectories (with the same number of modes).

    Args:
        policy_output[0]: waypoints, shape [batch_size, num_timesteps, 2] or [batch_size, num_modes, num_timesteps, dim]
        labels[0]: labels, shape [batch_size, num_timesteps, 2] or [batch_size, num_modes, num_timesteps, dim]
        order: order of the vector norm (2 for Euclidean)
    """
    candidate_trajectory, label_trajectory = prepare_data_for_DE(
        policy_output=policy_output,
        labels=labels,
        policy_output_waypoints_key=output_key,
        label_waypoints_key=output_label_key,
    )

    return calculate_ADE(
        candidate_trajectories=candidate_trajectory,
        gt_trajectory=label_trajectory,
        order=order,
    )


@torch.jit.script
def FDE_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_key: str,
    output_label_key: str,
    order: Union[int, float] = 2,
):
    """
    Wrapper for the Final Displacement Error (FDE) calculation. This is the displacement between the endpoints of two (multi-modal) trajectories (with the same number of modes).

    Args:
        policy_output[0]: waypoints, shape [batch_size, num_timesteps, 2] or [batch_size, num_modes, num_timesteps, dim]
        labels[0]: labels, shape [batch_size, num_timesteps, 2] or [batch_size, num_modes, num_timesteps, dim]
        order: order of the vector norm (2 for Euclidean)
    """
    candidate_trajectory, label_trajectory = prepare_data_for_DE(
        policy_output=policy_output,
        labels=labels,
        policy_output_waypoints_key=output_key,
        label_waypoints_key=output_label_key,
    )

    return calculate_FDE(
        candidate_trajectories=candidate_trajectory,
        gt_trajectory=label_trajectory,
        order=order,
    )


def vae_reconstruction_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_names: List[str],
    label_names: List[str],
    reconstruction_loss_func_config: dict,
    reduce_sample_multimodality: str,
    reduce_decoder_multimodality: str,
) -> Tensor:
    """
    Calculates the reconstruction loss for a multimodal vae / cvae / uae
    : param policy_output[output_names[0]]: multimodal_waypoints mean: shape [batch_size, num_latent_samples, num_modes_per_sample, num_timesteps, 2]
    : param policy_output[output_names[1]]: multimodal_waypoints std: shape [batch_size, num_latent_samples, num_modes_per_sample, num_timesteps, 2]
    : param labels[output_names[0]]: waypoints: shape [batch_size, num_timesteps, 2]
    : returns: loss: shape [batch_size]
    """

    def _reduce_dim(input: Tensor, strategy: str, dim: int):
        if strategy == "mean":
            return input.mean(dim=dim, keepdim=False)
        elif strategy == "wta":
            return input.min(dim=dim, keepdim=False)[0]
        else:
            raise RuntimeError(
                f"reduction strategy has to be either 'mean' or 'wta' but {strategy} was passed"
            )

    reconstruction_loss_fctn = loss_base_from_config(reconstruction_loss_func_config)

    num_samples, num_modes_per_sample = policy_output[output_names[0]].shape[1:3]
    # repeat label to match dimensions of policy output
    label = (
        labels[label_names[0]]
        .unsqueeze(1)
        .unsqueeze(1)
        .repeat(1, num_samples, num_modes_per_sample, 1, 1)
    )

    reconstruction_loss: Tensor = reconstruction_loss_fctn(
        {
            output_names[0]: policy_output[output_names[0]],
            output_names[1]: policy_output[output_names[1]],
        },
        {label_names[0]: label},
    )

    reconstruction_loss = _reduce_dim(
        reconstruction_loss, strategy=reduce_decoder_multimodality, dim=-1
    )
    reconstruction_loss = _reduce_dim(
        reconstruction_loss, strategy=reduce_sample_multimodality, dim=-1
    )

    return reconstruction_loss


def kld_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    posterior_mu_key: str,
    posterior_cov_key: str,
    prior_mu_key: str,
    prior_cov_key: str,
) -> Tensor:
    """
    calculates the KL divergence between a prior distribution (given as label) and a posterior (given as policy_output)
    both distributions are multivariate gaussians given by mu and lower triangular matrix L with LL^T=Sigma
    Args:
        policy_output[output_names[0]]: mu of posterior: shape [batch_size, D] mean of posterior
        policy_output[output_names[1]]: L of posterior: shape [batch_size, D, D] lower triangular of covariance matrix
        labels[output_names[0]]: mu of prior: shape [batch_size, D] mean of piror
        labels[output_names[1]]: L of prior: shape [batch_size, D, D] lower triangular of covariance matrix

    Returns:
        loss: shape [batch_size]
    """

    posterior = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=policy_output[posterior_mu_key],
        scale_tril=policy_output[posterior_cov_key],
    )
    prior = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=labels[prior_mu_key],
        scale_tril=labels[prior_cov_key],
    )

    return torch.distributions.kl.kl_divergence(p=posterior, q=prior)


def vae_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_names: List[str],
    label_names: List[str],
    reconstruction_loss_func_config: str,
    posterior_prior_loss_func_config: str,
    reduce_sample_multimodality: str,
    reduce_decoder_multimodality: str,
    beta: float,
) -> Tensor:
    """
    Vanilla VAE loss

    Args:
        policy_output[output_names[0]]: multimodal_waypoints mean: shape [batch_size, num_latent_samples, num_modes_per_sample, num_timesteps, 2]
        policy_output[output_names[1]]: multimodal_waypoints std: shape [batch_size, num_latent_samples, num_modes_per_sample, num_timesteps, 2, 2]
        policy_output[output_names[2]]: mu of posterior: shape [batch_size, D] mean of posterior
        policy_output[output_names[3]]: L of posterior: shape [batch_size, D, D] lower triangular covariance matrix
        policy_output[output_names[4]]: mu of prior: shape [batch_size, D] mean of prior
        policy_output[output_names[5]]: L of prior: shape [batch_size, D, D] lower triangular covariance matrix
        labels[output_names[0]]: waypoints: shape [batch_size, num_timesteps, 2]

    Returns:
        loss: shape [batch_size]
    """
    reconstruction_loss_val = vae_reconstruction_loss(
        {
            output_names[0]: policy_output[output_names[0]],
            output_names[1]: policy_output[output_names[1]],
        },
        {label_names[0]: labels[label_names[0]]},
        [output_names[0], output_names[1]],
        [label_names[0]],
        reconstruction_loss_func_config,
        reduce_sample_multimodality,
        reduce_decoder_multimodality,
    )

    posterior_prior_loss_function = loss_base_from_config(
        posterior_prior_loss_func_config
    )
    assert (
        "posterior" in output_names[2] and "posterior" in output_names[3]
    ), "Invalid prior/posterior ordering in VAE loss config .yaml, output_names[2] and output_names[3] should contain posterior distribution."
    assert (
        "prior" in output_names[4] and "prior" in output_names[5]
    ), "Invalid prior/posterior ordering in VAE loss config .yaml, output_names[4] and output_names[5] should contain prior distribution."
    # policy output is considered as posterior, labels as prior
    posterior_loss = posterior_prior_loss_function(
        policy_output={k: policy_output[k] for k in output_names[2:4]},
        labels={k: policy_output[k] for k in output_names[4:6]},
    )

    return reconstruction_loss_val + beta * posterior_loss


def _multimodal_gmm_loss(
    predicted_means: Tensor,
    predicted_covars: Tensor,
    predicted_probabilities: Tensor,
    target_waypoints: Tensor,
    eps: float = 1e-6,
):
    """
    Define loss of multimodal GMM model. Given predictions as means, covariance distribution and
    component probabilities of predictions, the NLL of the target waypoints under the
    predicted distribution is evaluated.  Losses are defined that train position and
    covariance predictions to minimise NLL, and losses to minimise the predicted mode component
    weights based on the given predictions and GT.

    Args:
        predicted_means: Predicted trajectory means with shape (batch, num_modes, samples=1, timesteps, values)
        predicted_covars: Covariance matrices, shape (batch, modes, timesteps, 2, 2)
        predicted_probabilities: Predicted mode probabilities, shape (batch, modes)
        target_waypoints: Ground-truth trajectories with shape (batch, timesteps, values=2)
        eps: Numerical stability parameter

    Returns:
        nll_covar_position_loss: Position and covariance training loss
        training_mode_weights: Weighting of training of each component
        mode_loss: Mode training loss

    """
    # find NLL of targets under predicted distribution
    position_deltas = _find_position_deltas(predicted_means, target_waypoints)
    neg_log_gaussian = _get_gaussian_error(predicted_covars, position_deltas)

    ground_truth_mask = torch.ones_like(target_waypoints[:, :, 0])
    (
        closest_modes,
        closest_modes_onehot,
        trajectory_differences,
        mode_posteriors,
    ) = _find_closest_modes(
        neg_log_gaussian,
        position_deltas,
        ground_truth_mask,
    )

    training_mode_weights = torch.detach(closest_modes_onehot)

    # predicted component weights are trained based on two factors, firstly under the NLL loss (where the distribution
    # parameters are kept constant), and secondly using a soft weighting with the distance to the GT for each mode.
    # define soft mode loss here. in previous DiPA methods there are separate mode estimates produced
    # as output by the network, here the mode prediction is fed into two different losses
    nll_covar_position_loss, nll_mode_loss = _nll_covar_loss(
        neg_log_gaussian,
        predicted_probabilities,
        ground_truth_mask,
        training_mode_weights,
    )

    # train predicted mode weights directly against the one-hot distribution
    training_mode_weights_screened = _screen_distribution(
        training_mode_weights, eps
    )
    predicted_probabilities_screened = _screen_distribution(
        predicted_probabilities, eps
    )
    log_predicted_probabilities = torch.log(predicted_probabilities_screened)
    mode_loss = torch.mean(
        torch.nn.functional.kl_div(
            log_predicted_probabilities,
            training_mode_weights_screened,
            reduction="none",
        ),
        dim=1,
    )

    return (
        nll_covar_position_loss,
        training_mode_weights,
        mode_loss,
    )


def vae_covar_loss(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_names: List[str],
    label_names: List[str],
    beta: float,
    posterior_prior_loss_func_config: Optional[Dict] = None,
    reconst_loss_weight: float = 1.0,
) -> Tensor:
    """
    Implementation of VAE loss using covariance representation of outputs, and training using either winner-takes-all
    or DiPA-style losses

    Args:
        policy_output[output_names[0]]: posterior multimodal_waypoints mean: shape [batch_size, num_latent_samples, num_modes_per_sample, num_timesteps, 2]
        policy_output[output_names[1]]: posterior multimodal_waypoints covar: shape [batch_size, num_latent_samples, num_modes_per_sample, num_timesteps, 2, 2]
        policy_output[output_names[2]]: mu of posterior: shape [batch_size, D] mean of piror
        policy_output[output_names[3]]: L of posterior: shape [batch_size, D, D] lower triangular covariance matrix
        policy_output[output_names[4]]: mu of prior: shape [batch_size, D] mean of piror
        policy_output[output_names[5]]: L of prior: shape [batch_size, D, D] lower triangular covariance matrix
        policy_output[output_names[6]]: prior predicted component weights: shape [batch_size, num_components]
        policy_output[output_names[7]]: posterior predicted component weights: shape [batch_size, num_components]
        policy_output[output_names[8]]: prior multimodal_waypoints mean: shape [batch_size, num_components, num_timesteps, 2]
        policy_output[output_names[9]]: prior multimodal_waypoints covar: shape [batch_size, num_components, num_timesteps, 2, 2]
        labels[output_names[0]]: waypoints: shape [batch_size, num_timesteps, 2]

    Returns:
        loss: shape [batch_size]
    """

    posterior_means = policy_output[output_names[0]][
        :, :, 0
    ]  # shape [batch, modes=1, timesteps, dim=2]
    posterior_covars = policy_output[output_names[1]][
        :, :, 0
    ]  # shape [batch, modes=1, timesteps, dim=2, dim=2]
    target_waypoints = labels[label_names[0]]  # shape [batch, timesteps, dim=2]
    prior_component_weights = policy_output["prior_components_weights"]
    posterior_component_weights = policy_output["posterior_components_weights"]

    losses = []
    (
        nll_covar_position_loss_posterior,
        training_mode_weights_posterior,
        mode_loss_posterior,
    ) = _multimodal_gmm_loss(
        posterior_means,
        posterior_covars,
        posterior_component_weights,
        target_waypoints,
    )
    training_mode_weights = training_mode_weights_posterior
    losses.extend(
        [
            reconst_loss_weight * nll_covar_position_loss_posterior,
            mode_loss_posterior,
        ]
    )

    # train prior distribution against posterior, using training weight distribution
    # record training distribution with outputs
    policy_output["training_mode_distribution"] = training_mode_weights
    assert (
        "posterior" in output_names[2] and "posterior" in output_names[3]
    ), "Invalid prior/posterior ordering in VAE loss config .yaml, output_names[2] and output_names[3] should contain posterior distribution."
    assert (
        "prior" in output_names[4] and "prior" in output_names[5]
    ), "Invalid prior/posterior ordering in VAE loss config .yaml, output_names[4] and output_names[5] should contain prior distribution."
    posterior_loss = kld_loss_gmm_prior_multimodal(
        policy_output=policy_output,
        labels=policy_output,  # labels is the prior distribution
        posterior_mu_key=output_names[2],
        posterior_cov_key=output_names[3],
        prior_mu_key=output_names[4],
        prior_cov_key=output_names[5],
        training_weights=training_mode_weights,
    )

    # train prior mode weights to approximate posterior
    prior_mode_dist = torch.distributions.categorical.Categorical(
        probs=prior_component_weights
    )
    posterior_mode_dist = torch.distributions.categorical.Categorical(
        probs=torch.detach(posterior_component_weights)
    )
    mode_kld_loss = torch.distributions.kl.kl_divergence(
        p=posterior_mode_dist, q=prior_mode_dist
    )
    losses.extend([beta * posterior_loss, mode_kld_loss])

    return torch.sum(torch.stack(losses, dim=1), dim=1)


def _find_position_deltas(
    trajectory_prediction: Tensor, ground_truth: Tensor
) -> Tensor:
    """
    Find position differences between predicted trajectories and ground-truth. Predicted and ground-truth values
    are for future timesteps not including the last observed.

    Args:
        trajectory_prediction: Predicted trajectories with shape (batch, num_modes, samples=1, timesteps, values)
        ground_truth: Ground-truth trajectories with shape (batch, timesteps, values=2)

    Returns:
        position_deltas: Differences between position values, shape (agents, num_modes, timesteps, values=2)
    """
    predicted_positions = trajectory_prediction[:, :, :, :2]
    ground_truth_positions = torch.detach(ground_truth[:, :, :2])
    # position_deltas is shape (agents, num_modes, timesteps, values=2)
    position_deltas = predicted_positions - ground_truth_positions[:, None, :, :]

    return position_deltas


def _get_gaussian_error(
    covar: Tensor,
    position_deltas: Tensor,
    neg_log_gaussian_limit: Optional[float] = 100.0,
) -> Tensor:
    """
    Find NLL value of ground truth based on predicted covariance

    Args:
        covar: Covariance matrices, shape (batch, modes, timesteps, 2, 2)
        position_deltas: Position differences, shape (batch, modes, timesteps, 2)

    Returns:
        neg_log_gaussian: Negative-log probability density values, array of shape (batch, modes, timesteps)
    """
    # training instability can happen if the determinant of the covar matrix gets too small, as the inverse function
    # is based on 1/det = 1/(ad-bc) .  this is prevented from getting too small by restricting the min covar std.
    # the range of values that the covar std can cover is controlled using the non-linearity of the covar std output
    # produced by the network model.
    covar_inv = torch.linalg.inv(covar)

    covar_multiply_right = torch.einsum("nmtab,nmtb->nmta", covar_inv, position_deltas)
    covar_multiply_left = torch.einsum(
        "nmta,nmta->nmt", position_deltas, covar_multiply_right
    )
    neg_log_exp_term = 0.5 * covar_multiply_left

    # covar_det is shape (agents, modes, timesteps)
    covar_det = torch.linalg.det(covar)
    denom = 2 * torch.pi * torch.sqrt(covar_det)
    neg_log_denom_term = torch.log(denom)
    # neg_log_gaussian is shape (agents, modes, timesteps)
    neg_log_gaussian = neg_log_exp_term + neg_log_denom_term

    # applying limit is disabled as it depended on a function to clip while preserving gradient.
    # a consequence is that it may lead to numerical instability without this clamping
    if neg_log_gaussian_limit is not None:
        neg_log_gaussian = clip_by_value_preserve_gradient.apply(
            neg_log_gaussian, -torch.inf, neg_log_gaussian_limit
        )
    return neg_log_gaussian


class clip_by_value_preserve_gradient(torch.autograd.Function):
    """
    Torch function that clips values but uses a linear gradient, to allow training of
    instances occurring in the clipped regions.  This is comparable to the function
    clip_by_value_preserve_gradient in the tensorflow probability library.
    """

    @staticmethod
    def forward(ctx, x, clip_min, clip_max):
        """
        Apply clamp in forward pass
        """
        return torch.clamp(x, clip_min, clip_max)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass a linear partial derivative is used (which passes grad_output as-is)
        """
        return grad_output, None, None


def _find_closest_modes(
    neg_log_gaussian: Tensor,
    position_deltas: Tensor,
    ground_truth_mask: Tensor,
    eps: float = 1e-9,
    position_distances_threshold: Optional[float] = 20.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Find which modes are closest to GT and define trajectory differences and posteriors of observed positions under
    the predicted spatial distribution.

    Args:
        neg_log_gaussian: Negative-log probability density values of observed positions in the
        predicted spatial distribution. Array of shape (batch, modes, timesteps)
        position_deltas: Position differences between predicted and ground-truth positions, array of
        shape (batch, modes, timesteps, values=2)
        ground_truth_mask: Mask representing if each ground truth position/timestep is defined,
        boolean array of shape (batch, timesteps), where timesteps include the last observed timestep.
        eps: Numerical stability parameter
        position_distances_threshold: Ceiling threshold value to apply on position distances, to prevent large
        losses when predictions are far from the ground truth

    Returns:
        Tuple of closest_modes, closest_modes_onehot, trajectory_differences, mode_posteriors.
        closest_modes is an integer array of shape (batch,) representing which mode is the closest
        closest_modes_onehot is a one-hot representation of closest_modes, with shape (agents, modes)
        trajectory_differences is an array of shape (batch, modes) representing the difference between the
        predicted and ground-truth trajectory for each mode.  the method of calculating difference can vary
        mode_posteriors is an array of shape (batch, modes) representing the posterior of observed positions under
        the predicted spatial distribution.
    """
    position_distances = torch.linalg.norm(position_deltas, axis=3)
    if position_distances_threshold is not None:
        position_distances = clip_by_value_preserve_gradient.apply(
            position_distances, 0.0, position_distances_threshold
        )

    eval_position_distances = position_distances

    # reduce distances over timesteps using mask
    ground_truth_mask_future = ground_truth_mask[:, None, :]
    ground_truth_agent_weight = torch.sum(ground_truth_mask_future, dim=2)
    ground_truth_mask_sum = torch.clamp_min(ground_truth_agent_weight, eps)
    reduced_position_distances = (
        torch.sum(eval_position_distances * ground_truth_mask_future, dim=2)
        / ground_truth_mask_sum
    )

    # trajectory_differences is shape (batch, modes)
    trajectory_differences = reduced_position_distances

    closest_modes = torch.argmin(trajectory_differences, dim=1)
    closest_modes_onehot = torch.detach(
        torch.nn.functional.one_hot(closest_modes, position_deltas.shape[1])
    )

    # find posterior weights of each mode of ground-truth positions from predicted GMM distribution
    timestep_probabilities = torch.exp(-neg_log_gaussian)
    # normalise probabilities for each timestep
    # timestep_posteriors is shape (agents, modes, timesteps)
    timestep_posteriors = timestep_probabilities / torch.clamp_min(
        torch.sum(timestep_probabilities, dim=1, keepdim=True), eps
    )
    # find posterior distribution over modes by reducing over timesteps using mask
    # posterior_distribution is shape (agents, modes)
    posterior_distribution = (
        torch.sum(timestep_posteriors * ground_truth_mask_future, dim=2)
        / ground_truth_mask_sum
    )

    # re-normalise, in case some timesteps were thresholded and all-zero, which would lead to a sum < 1
    posterior_distribution = posterior_distribution / torch.clamp_min(
        torch.sum(posterior_distribution, dim=1, keepdim=True), eps
    )

    # mode_posteriors is shape (agents, modes)
    mode_posteriors = torch.detach(posterior_distribution)

    return (
        closest_modes,
        closest_modes_onehot,
        trajectory_differences,
        mode_posteriors,
    )


def _nll_covar_loss(
    neg_log_gaussian: Tensor,
    mode_prediction: Tensor,
    ground_truth_mask: Tensor,
    training_mode_weights: Tensor,
    eps: float = 1e-6,
    modes_training_weight: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """
    Define loss for training of component weight estimates using spatial error distribution

    Args:
        neg_log_gaussian: Array of NLL values, shape (batch, modes, timesteps)
        mode_prediction: Predicted mode weight distribution for spatial distribution,
        shape (batch, modes)
        ground_truth_mask: Mask representing if each ground truth position/timestep is defined,
        boolean array of shape (batch, timesteps), where timesteps include the last observed timestep.
        training_mode_weights: Mode weight distribution used for training of spatial
        distribution
        eps: Numerical stability parameter
        modes_training_weight: Ratio for balancing training between posterior and predicted mode losses

    Returns:
        Tuple of spatial loss (batch,) and predicted log likelihood values with shape (batch, timesteps)
    """
    # ensure distribution values are non-zero
    mode_prediction_screened = _screen_distribution(mode_prediction, eps)
    training_mode_weights_screened = _screen_distribution(training_mode_weights, eps)

    ground_truth_mask_future = ground_truth_mask

    # train mode weights to minimise NLL loss based on current covar/mean positions.
    # disable gradient through covar/position terms as this would train them against predicted modes.
    predicted_mode_nll_loss, predicted_log_likelihoods_timesteps = _multimodal_nll(
        mode_prediction_screened,
        torch.detach(neg_log_gaussian),
        ground_truth_mask_future,
    )

    # find loss between modes and posteriors
    # train NLL mode values to match posterior values
    log_mode_prediction = torch.log(mode_prediction_screened)
    posterior_nll_mode_loss = torch.nn.functional.kl_div(
        log_mode_prediction, training_mode_weights_screened
    )

    nll_mode_loss = (
        modes_training_weight * posterior_nll_mode_loss
        + (1.0 - modes_training_weight) * predicted_mode_nll_loss
    )

    nll_covar_position_loss, _ = _multimodal_nll(
        training_mode_weights_screened, neg_log_gaussian, ground_truth_mask_future
    )

    return nll_covar_position_loss, nll_mode_loss


def _multimodal_nll(
    mode_weights: Tensor,
    mode_timestep_nll: Tensor,
    ground_truth_mask_future: Tensor,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """
    Find overall NLL score given mode weights and NLL values per mode and timestep

    Args:
        mode_weights: Predicted mode weight distribution, shape (batch, modes)
        mode_timestep_nll: Array of NLL values, shape (batch, modes, timesteps)
        ground_truth_mask_future: Mask representing if each ground truth position/timestep is defined,
        boolean array of shape (batch, timesteps).
        eps: Numerical stability parameter

    Returns:
        Tuple of NLL error (batch,) and predicted log likelihood values with shape (batch, timesteps)
    """
    # log_weighted_likelihoods is shape (agents, modes, timesteps)
    log_weighted_likelihoods = torch.log(mode_weights[:, :, None]) + (
        -mode_timestep_nll
    )

    # reduce over modes in a numerically stable way (preventing overflow from exp)
    max_val, _ = torch.max(log_weighted_likelihoods, dim=1)
    log_likelihoods_timesteps = max_val + torch.log(
        torch.sum(torch.exp(log_weighted_likelihoods - max_val[:, None, :]), dim=1)
        + eps
    )
    # find NLL loss from reducing over timesteps, using mask
    masked_nll = -torch.sum(
        log_likelihoods_timesteps * ground_truth_mask_future, dim=1
    ) / torch.clamp_min(torch.sum(ground_truth_mask_future, dim=1), eps)

    return masked_nll, log_likelihoods_timesteps


def _screen_distribution(distribution: Tensor, weighting: Optional[float]):
    """
    Ensure each value of the distribution is non-zero, by combining with a flat distribution with a given weighting.
    This is an alternative to adding a small epsilon value, and ensures that the resulting output is a normalised
    distribution.

    Args:
        distribution: Distribution to (optionally) modify, where the distribution is represented over
        the last dimension (and is assumed to be normalised).
        weighting: Weighting given to the flat distribution, typically small.

    Returns:
        Resulting distribution after combining the input distribution with a flat distribution with the given
        weighting
    """
    if weighting is None:
        return distribution
    flat_dist = torch.ones_like(distribution) / distribution.shape[-1]
    return distribution + weighting * (flat_dist - distribution)


def mixture_covar_nll_eval(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_names: List[str],
    label_names: List[str],
    timestamp_ratio: Optional[float],
) -> Tensor:
    """
    Method for evaluating NLL error of ground truth positions under GMM predictions.

    Args:
        policy_output[output_names[0]]: multimodal_vae_prior_future_waypoints: shape [batch_size, num_modes, samples=1, num_timesteps, 2],
        policy_output[output_names[1]]: multimodal_vae_prior_future_waypoints_covar: shape [batch_size, num_modes, samples=1, num_timesteps, 2, 2],
        policy_output[output_names[2]]: prior_components_weights: shape [batch_size, num_modes],
        labels[output_names[0]]: waypoints: shape [batch_size, num_timesteps, 2]
        timestamp_ratio: ratio of timestamp length for specific timestamp to calculate

    Returns:
        loss, shape [batch_size]
    """
    prior_future_waypoints = policy_output[output_names[0]][
        :, :, 0
    ]  # shape [batch, modes, timesteps, dim=2]
    prior_future_waypoints_covar = policy_output[output_names[1]][
        :, :, 0
    ]  # shape [batch, modes, timesteps, dim=2, dim=2]
    prior_components_weights = policy_output[output_names[2]]  # shape [batch, modes]
    target_waypoints = labels[label_names[0]]  # shape [batch, timesteps, dim=2]

    if timestamp_ratio is not None:
        end_ts = round(prior_future_waypoints.shape[2] * timestamp_ratio)
        prior_future_waypoints = prior_future_waypoints[:, :, end_ts - 1 : end_ts]
        prior_future_waypoints_covar = prior_future_waypoints_covar[
            :, :, end_ts - 1 : end_ts
        ]
        target_waypoints = target_waypoints[:, end_ts - 1 : end_ts]

    # find NLL of targets under predicted distribution
    position_deltas = _find_position_deltas(prior_future_waypoints, target_waypoints)
    # disable use of NLL limit when used for evaluation
    neg_log_gaussian = _get_gaussian_error(
        prior_future_waypoints_covar, position_deltas, neg_log_gaussian_limit=None
    )

    ground_truth_mask = torch.ones_like(target_waypoints[:, :, 0])

    masked_nll, _ = _multimodal_nll(
        prior_components_weights, neg_log_gaussian, ground_truth_mask
    )

    return masked_nll


def dual_multimodality_adapter(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    *,
    output_names: List[str],
    label_names: List[str],
    loss_func_config: dict,
    wta: bool = False,
) -> Tensor:
    """
    Flattens two multimodality dimensions into one so that multimodal losses (minxDE) can be called. If 'wta' is set to true, non-multimodal losses (xDE) can be called as well via the multimodal_wta_loss wrapper.
    Example: [batch_size, num_modes_1, num_modes_2, num_timesteps, dim] -> [batch_size, num_modes_1*num_modes_2, num_timesteps, dim]
    """
    assert len(output_names) == len(label_names) == 1

    policy_output = {
        output_names[0]: policy_output[output_names[0]].flatten(start_dim=1, end_dim=2)
    }
    label = {label_names[0]: labels[label_names[0]]}

    if wta:
        wta_loss_func = MultimodalWTALoss(config={"loss_func_config": loss_func_config})
        loss_value = wta_loss_func(policy_output, labels=label)
    else:
        loss_function = loss_base_from_config(loss_func_config)
        loss_value = loss_function(policy_output, labels=label)
    return loss_value


def multimodal_trajectory_spread(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_names: List[str],
    label_names: List[str],
    loss_func_config: dict,
) -> Tensor:
    """
    Computes the spread between the multi-modal trajectories given a loss function. The loss function is applied to all mode combinations as inputs. Useful to gauge "multi-modality" of a predictor.

    Args:
        policy_output: multi-modal trajectories, shape: [batch_size, num_modes, num_points, dim]
        labels: ground-truth trajectory -- unused
        path_to_loss_func: Loss function to compute inter-mode loss.
    Returns:
        The average inter-mode loss value, shape: [batch_size]
    """
    assert len(output_names) == len(label_names) == 1  # single output-label pair
    assert (
        len(policy_output[output_names[0]].shape) == 4
    ), "Only multi-modal output supported"
    # index loss function
    loss_function = loss_base_from_config(loss_func_config)
    policy_tensor = policy_output[output_names[0]]

    # compute multi-modal loss
    batch_size, num_modes, _, _ = policy_tensor.shape

    inter_mode_loss = torch.zeros((batch_size, num_modes), device=policy_tensor.device)
    for i in range(num_modes):
        # roll modes by i+1 to use as ground truth
        rolled_policy_output = {output_names[0]: policy_tensor.clone()}
        rolled_policy_output[label_names[0]] = rolled_policy_output[
            output_names[0]
        ].roll(shifts=i + 1, dims=1)

        inter_mode_loss += loss_function(
            policy_output=policy_output,
            labels=rolled_policy_output,
        )

    # average inter-mode loss by num_modes
    inter_mode_loss = torch.mean(inter_mode_loss, dim=1)
    return inter_mode_loss


def nll_mixture_attimestep(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_mean_key: str,
    output_std_key: str,
    output_weights_key: str,
    label_key: str,
    ratio: float,
    normal_dist_std: Union[str, float] = 1.0,
    clamp_std: bool = True,
) -> Tensor:
    """
    Wrapper for computing the negative log prob. of the difference between label trajectory and a mixture of predicted trajectories, computed at a single timestep

    The `normal_dist_std` argument tunes the std (and thus variance) of the normal distribution. It can either be the regressed value from the model or fixed to a float value.

    Args:
        policy_output[output_mean_key]: predicted trajectory, shape [B, C, N, T, D], C is mixture components, N is decoder modes
        policy_output[output_std_key]: std of predicted trajectory, shape [B, C, N, T, D], C is mixture components, N is decoder modes
        policy_output[output_weights_key]: mixture_weights, shape [B, C]
        labels: ground-truth trajectory, shape [B, T, D]
        ratio: ratio of the trajectory used to compute the NLL at a specific time step
        normal_dist_std: selects the std method, either the predicted trajectory std or a fixed value
        clamp_std: clamp stds in training
    Returns:
        log_prob: negative log prob. of entire ground-truth trajectory and prediction mismatch at a timestep, shape: [B]
    """
    assert (
        policy_output[output_mean_key].shape == policy_output[output_std_key].shape
    ), f"Mixture means and stds must be of same dimension, received: {policy_output[output_mean_key].shape} and {policy_output[output_std_key].shape}"
    assert (
        policy_output[output_mean_key].shape[2] == 1
    ), "> 1 decoder modes not supported"

    mixture_means = policy_output[output_mean_key].squeeze(2)
    mixture_stds = policy_output[output_std_key].squeeze(2)
    mixture_weights = policy_output[output_weights_key]

    mixture_weights_sum = torch.sum(mixture_weights, dim=1)
    assert torch.allclose(
        mixture_weights_sum, torch.ones_like(mixture_weights_sum)
    ), "Invalid mixture weights; do not sum to one"

    # repeat label to match dimensions of policy output
    num_components = policy_output[output_mean_key].shape[1]
    label = labels[label_key].unsqueeze(1).repeat(1, num_components, 1, 1)

    nll_stepwise_value = _nll_mixture(
        mixture_means=mixture_means,
        mixture_stds=mixture_stds,
        mixture_weights=mixture_weights,
        label=label,
        normal_dist_std=normal_dist_std,
        clamp_std=clamp_std,
    )  # [B, T]

    assert (
        len(nll_stepwise_value.shape) == 2
    ), f" Shape of step-wise NLL must be [B, T], received: {len(nll_stepwise_value.shape)}"

    timestep = round(ratio * nll_stepwise_value.shape[-1]) - 1

    return nll_stepwise_value[..., timestep]


def _nll_mixture(
    mixture_means: Tensor,
    mixture_stds: Tensor,
    mixture_weights: Tensor,
    label: Tensor,
    normal_dist_std: Union[str, float] = 1.0,
    clamp_std: bool = True,
) -> Tensor:
    """
    Computes the negative log prob. of the difference between label trajectory and a mixture of predicted trajectories, each evaluated under a normal distribution with mean 0 and tunable variance. Computed individually over each component and mixture-weighted-summed together.
    The `normal_dist_std` argument tunes the std (and thus variance) of the normal distribution. It can either be the regressed value from the model or fixed to a float value.

    Args:
        mixture_means: predicted trajectory, shape [B, C, T, D], C is mixture components
        mixture_stds: std of predicted trajectory, shape [B, C, T, D], C is mixture components
        mixture_weights: mixture_weights, shape [B, C]
        label: ground-truth trajectory, shape [B, C, T, D]
        normal_dist_std: selects the std method, either the predicted trajectory std or a fixed value
        clamp_std: clamp stds in training
    Returns:
        log_prob: negative log prob. of entire ground-truth trajectory and prediction mismatch, shape: [B, T]
    """
    nll_stepwise_value = _nll_stepwise(
        output_trajectory_mean=mixture_means,
        output_trajectory_std=mixture_stds,
        label_trajectory=label,
        normal_dist_std=normal_dist_std,
        clamp_std=clamp_std,
    )  # [B, C, T]

    nll_stepwise_value = nll_stepwise_value * mixture_weights.unsqueeze(-1)  # [B, C, T]
    nll_stepwise_value = nll_stepwise_value.sum(dim=1)  # [B, T]

    return nll_stepwise_value


def nll_trajectorywise(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_mean_key: str,
    output_std_key: str,
    label_key: str,
    normal_dist_std: Union[str, float] = 1.0,
    clamp_std: bool = True,
) -> Tensor:
    """
    Wrapper for computing the negative log prob. of the difference between label trajectory and predicted trajectory, evaluated under a normal distribution with mean 0 and tunable variance. Computed over entire trajectory with timesteps summed together.
    The `normal_dist_std` argument tunes the std (and thus variance) of the normal distribution. It can either be the regressed value from the model or fixed to a float value.

    Args:
        policy_output[0]: predicted trajectory, shape [B, ...,  T, D],  '...' indicates any number of multi-modality dimensions
        policy_output[1]: std of predicted trajectory, shape [B, ...,  T, D],  '...' indicates any number of multi-modality dimensions
        labels: ground-truth trajectory, shape [B, ...,  T, D]
        normal_dist_std: selects the std method, either the predicted trajectory std or a fixed value
        clamp_std: clamp stds in training
    Returns:
        log_prob: negative log prob. of entire ground-truth trajectory and prediction mismatch, shape: [B, ...]
    """
    assert isinstance(normal_dist_std, float) or (
        isinstance(normal_dist_std, str) and normal_dist_std == "regressed_std"
    ), "std of the predicted trajectory in NLL computation must be `regressed_std` or a set float value"

    # compute step-wise NLL, [B, ..., T]
    nll_stepwise_value = _nll_stepwise(
        output_trajectory_mean=policy_output[output_mean_key],
        output_trajectory_std=policy_output[output_std_key],
        label_trajectory=labels[label_key],
        normal_dist_std=normal_dist_std,
        clamp_std=clamp_std,
    )
    assert (
        len(nll_stepwise_value.shape) >= 2
    ), f" Shape of step-wise NLL must be [B, ..., T], received: {len(nll_stepwise_value.shape)}"

    return nll_stepwise_value.sum(dim=-1)


def nll_attimestep(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    output_mean_key: str,
    output_std_key: str,
    label_key: str,
    ratio: float,
    normal_dist_std: Union[str, float] = 1.0,
    clamp_std: bool = True,
) -> Tensor:
    """
    Wrapper for computing the negative log prob. of the difference between label trajectory and predicted trajectory, evaluated under a normal distribution with mean 0 and tunable variance. Computed at a time step given by ratio along the trajectory.
    The `normal_dist_std` argument tunes the std (and thus variance) of the normal distribution. It can either be the regressed value from the model or fixed to a float value.

    Args:
        policy_output[0]: predicted trajectory, shape [B, ...,  T, D],  '...' indicates any number of multi-modality dimensions
        policy_output[1]: std of predicted trajectory, shape [B, ...,  T, D],  '...' indicates any number of multi-modality dimensions
        labels: ground-truth trajectory, shape [B, ...,  T, D]
        ratio: ratio of the trajectory used to compute the NLL at a specific time step
        normal_dist_std: selects the std method, either the predicted trajectory std or a fixed value
        clamp_std: clamp stds in training
    Returns:
        log_prob: negative log prob. of ground-truth trajectory at a timestep, shape: [B, ...]
    """
    assert isinstance(normal_dist_std, float) or (
        isinstance(normal_dist_std, str) and normal_dist_std == "regressed_std"
    ), "std of the predicted trajectory in NLL computation must be `regressed_std` or a set float value"
    assert ratio >= 0.0 and ratio <= 1.0, f"ratio must be [0.0, 1.0], received: {ratio}"

    # compute step-wise NLL, [B, ..., T]
    nll_stepwise_value = _nll_stepwise(
        output_trajectory_mean=policy_output[output_mean_key],
        output_trajectory_std=policy_output[output_std_key],
        label_trajectory=labels[label_key],
        normal_dist_std=normal_dist_std,
        clamp_std=clamp_std,
    )
    assert (
        len(nll_stepwise_value.shape) >= 2
    ), f" Shape of step-wise NLL must be [B, ..., T], received: {len(nll_stepwise_value.shape)}"

    timestep = round(ratio * nll_stepwise_value.shape[-1]) - 1

    return nll_stepwise_value[..., timestep]


def _nll_stepwise(
    output_trajectory_mean: Tensor,
    output_trajectory_std: Tensor,
    label_trajectory: Tensor,
    normal_dist_std: Union[str, float] = 1.0,
    clamp_std: bool = True,
) -> Tensor:
    """
    Computes the negative log prob. of the ground-truth trajectory and predicted trajectory mismatch given the std method.

    Args:
        output_trajectory_mean: predicted trajectory mean, shape [B, ...,  T, D],  '...' indicates any number of multi-modality dimensions
        output_trajectory_std: predicted trajectory std, shape [B, ...,  T, D],  '...' indicates any number of multi-modality dimensions
        label_trajectory: ground-truth trajectory, shape [B, ...,  T, D]
        normal_dist_std: selects the std method, either the predicted trajectory std or a fixed value
        clamp_std: clamp stds in training
    Returns:
        log_prob: negative log prob. of ground-truth trajectory and prediction mismatch at each time step, shape: [B, ..., T]
    """

    def compute_log_prob(mean: torch.Tensor, std: torch.Tensor, val: torch.Tensor):
        if clamp_std:
            std = torch.clamp(std, min=0.1, max=10.0)
        var = std**2
        return (
            -((val - mean) ** 2) / (2 * var)
            - torch.log(std)
            - torch.log(torch.sqrt(torch.tensor(2 * torch.pi)))
        )

    assert (
        output_trajectory_mean.shape[-1]
        == output_trajectory_std.shape[-1]
        == label_trajectory.shape[-1]
        == 2
    ), "Only x, y, output and labels are allowed"

    assert (
        output_trajectory_mean.shape
        == output_trajectory_std.shape
        == label_trajectory.shape
    ), f"Predictions mean, predictions std, and label trajectory must match shape-wise, received: {output_trajectory_mean.shape}, {output_trajectory_std.shape}, {label_trajectory.shape}"

    # compute diff to gt, [B, ..., T, D]
    diff_to_gt = output_trajectory_mean - label_trajectory

    # compute log probability under assumed normal
    if normal_dist_std == "regressed_std":
        # under regressed std
        log_prob = compute_log_prob(
            mean=torch.zeros_like(output_trajectory_std),
            std=output_trajectory_std,
            val=diff_to_gt,
        )
    else:
        # under set std
        gt_dist = torch.distributions.Normal(loc=0, scale=normal_dist_std)
        log_prob = gt_dist.log_prob(diff_to_gt)

    # sum over dim, [B, ..., T, D] -> [B, ..., T]
    log_prob = log_prob.sum(dim=-1)

    return -log_prob


def kld_loss_gmm_prior_multimodal(
    policy_output: Dict[str, Tensor],
    labels: Dict[str, Tensor],
    posterior_mu_key: str,
    posterior_cov_key: str,
    prior_mu_key: str,
    prior_cov_key: str,
    training_weights: Tensor,
    detach_posterior: bool = False,
) -> Tensor:
    """
    Calculates the KL divergence between a prior distribution (given as label) and a posterior (given as policy_output)
    both distributions are multivariate gaussians given by mu and lower triangular matrix L with LL^T=Sigma

    Args:
        policy_output[posterior_mu_key]: mu of posterior: shape [batch_size, D] mean of posterior
        policy_output[posterior_cov_key]: L of posterior: shape [batch_size, D, D] lower triangular of covariance matrix
        labels[prior_mu_key]: mu of prior: shape [batch_size, K, D] mean of prior
        labels[prior_cov_key]: L of prior: shape [batch_size, K, D, D] lower triangular of covariance matrix
        labels[component_weights_key]: components weights of prior: shape [batch_size, K] lower triangular of covariance matrix

    Returns:
        loss: shape [batch_size]
    """

    posterior_loc = policy_output[posterior_mu_key]
    posterior_tril = policy_output[posterior_cov_key]

    if detach_posterior:
        posterior_loc_ = torch.detach(posterior_loc)
        posterior_tril_ = torch.detach(posterior_tril)
    else:
        posterior_loc_ = posterior_loc
        posterior_tril_ = posterior_tril

    prior_mu = labels[prior_mu_key]
    prior_tril = labels[prior_cov_key]
    n_components = labels[prior_mu_key].shape[1]
    d_kl_per_component = []
    for n in range(n_components):
        posterior = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=posterior_loc_[:, n, :],
            scale_tril=posterior_tril_[:, n, :],
        )

        prior = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=prior_mu[:, n, :],
            scale_tril=prior_tril[:, n, :, :],
        )
        d_kl_per_component.append(
            torch.distributions.kl.kl_divergence(p=posterior, q=prior)
        )

    stacked_d_kl = torch.stack(d_kl_per_component, dim=1)
    return torch.sum(stacked_d_kl * training_weights, dim=1)


def loss_base_from_config(config: dict) -> LossBase:
    """
    Initializes a loss function

    Args:
        config: has values at keys <path_to_class> and <config>; value at <config> is the config for the loss
    """
    cfg = ensure_init_type(config, LossBaseConfig)
    loss_fn = class_from_path(cfg.path_to_class)
    return loss_fn(config=cfg.config)


class FlatteningLossWrapper(LossBase):
    """
    A wrapper that takes policy output and labels of
    shape = (batch_size, dim_1, dim_2, dim_3),
    flattens them to
    shape = (batch_size, dim_1 * dim_2, dim_3),
    and applies wrapped_loss_function to the flattened tensors.

    It is primarily used/introduced for differentiable simulation with dim_1=sim_steps, dim_2=num_waypoints, dim_3=dim_waypoints

    Args:
        wrapped_loss_config: loss function that will be called upon flattened elements in <policy_output> and <labels>
        ignore_until_index: values until this index are ignored for loss computation in dim_1

    Returns:
        computed loss value
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self._ignore_until_index = self._config.get("ignore_until_index", 0)
        self._wrapped_loss = loss_base_from_config(self._config.wrapped_loss_config)

    def __call__(
        self, policy_output: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        assert all(
            self._ignore_until_index < value.shape[1]
            for value in policy_output.values()
        )

        policy_output_flatted = {
            key: value[:, self._ignore_until_index :, ...].flatten(0, 1)
            for key, value in policy_output.items()
        }
        labels_flattend = {
            key: value[:, self._ignore_until_index :, ...].flatten(0, 1)
            for key, value in labels.items()
        }
        return self._wrapped_loss(policy_output_flatted, labels_flattend)

    @property
    def label_names(self) -> List[str]:
        return self._wrapped_loss.label_names

    @property
    def output_names(self) -> List[str]:
        return self._wrapped_loss.output_names


class MultimodalWTALoss(LossBase):
    """
    Implements winner-takes-all multi-modal loss using the loss function in function argument.

    Config is
    {
        "loss_func_config": ...,
    }
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)
        self._wrapped_loss = loss_base_from_config(self._config.loss_func_config)

    def __call__(
        self, policy_output: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> Union[np.ndarray, Tensor]:
        """
        Args:
            policy_output: multi-modal trajectories, shape: [batch_size, num_modes, num_points, dim]
            labels: uni-modal ground truth, shape: [batch_size, num_points, dim]
        """

        num_modes = policy_output[self.output_names[0]].shape[1]

        # make labels multimodal
        multimodal_labels = {
            label_key: labels[label_key].unsqueeze(1).repeat(1, num_modes, 1, 1)
            for label_key in self.label_names
        }
        mode_losses = self._wrapped_loss(policy_output, multimodal_labels)

        # find lowest loss mode-wise
        mode_wise_min_loss, _ = torch.min(mode_losses, dim=1)
        return mode_wise_min_loss

    @property
    def label_names(self) -> List[str]:
        return self._wrapped_loss.label_names

    @property
    def output_names(self) -> List[str]:
        return self._wrapped_loss.output_names


class ClusteredOutputLossFunctionAdapter(LossBase):
    """
    Adapter function for computing a loss between a label and a clustered model output,
    for example a multi-modal trajectory whose modes are clustered into fewer modes.
    Receives a loss function and a clustering function. Calls clustering function
    on the unclustered model output and then the loss function between clustered output and label

    Config is:
        num_clusters: Number of desired clusters
        path_to_clustering_func: Clustering function
        clustering_func_kwargs: Additional arguments for the clustering function
        use_cluster_weights_in_loss_func: Pass weights/probabilities of each cluster to the loss function
        loss_func_config: Loss function
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.clustering_function = lambda x: clustering_function(
            x,
            path_to_clustering_func=self._config.path_to_clustering_func,
            num_clusters=self._config.num_clusters,
            **self._config.get("clustering_func_kwargs", {}),
        )
        self._wrapped_loss = loss_base_from_config(
            self._config.get("loss_func_config", {})
        )

    def __call__(
        self, policy_output: Dict[str, Tensor], labels: Dict[str, Tensor]
    ) -> np.ndarray | Tensor:
        cluster_centroids, cluster_std, cluster_weights = self.clustering_function(
            x=policy_output[self._wrapped_loss.output_names[0]]
        )
        # construct policy output
        clustered_output = {
            self._wrapped_loss.output_names[0]: cluster_centroids,
        }
        output_names = [self._wrapped_loss.output_names[0]]
        if self._config.use_cluster_weights_in_loss_func:
            cluster_weights_output_name = "probabilities"
            output_names.append(cluster_weights_output_name)
            clustered_output.update(
                {
                    cluster_weights_output_name: cluster_weights,
                }
            )
        return self._wrapped_loss(policy_output=clustered_output, labels=labels)

    @property
    def label_names(self) -> List[str]:
        return [self._config.output_key]

    @property
    def output_names(self) -> List[str]:
        return [self._config.output_label_key]

    @property
    def _loss_fn(self) -> Callable:
        self._wrapped_loss
