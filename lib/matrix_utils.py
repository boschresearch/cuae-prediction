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

import numpy as np
import torch


def build_tril_matrix(logvar: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
    """
    Builds a tril matrix using logvar == log sigma^2 and corr == symmetric correlation factors between each dimension.

    Example 3D:
    sigma_1                    0                          0
    r_21 * sigma_2 * sigma_1   sigma_2                    0
    r_31 * sigma_3 * sigma_1   r_32 * sigma_3 * sigma_2   sigma_3

    :param logvar: Tensor of variances [batch_size, dim]
    :param corr: Tensor of covariances [batch_size, (dim*(dim-1))/2]
    :returns lower_triangle_covariance_matrix [batch_size, dim, dim]
    """
    assert logvar.shape[0] == corr.shape[0]

    std = torch.exp(0.5 * logvar)
    batch_size = std.shape[0]
    dim = std.shape[-1]
    corr_dim = corr.shape[-1]

    assert corr_dim == dim * (dim - 1) // 2
    assert torch.all(torch.ge(corr, -1 * torch.ones_like(corr))) and torch.all(
        torch.le(corr, torch.ones_like(corr))
    )

    # build symmetric matrix with \sigma_i * \sigma_i on the diagonal and \sigma_i * \sigma_j off-diagonal
    var_matrix = std.unsqueeze(-1).repeat(1, 1, dim)
    var_matrix = var_matrix * var_matrix.transpose(2, 1)  # el-wise product

    # build matrix with zeros on diagonal and correlations under
    corr_matrix = torch.zeros((batch_size, dim, dim), device=std.device)
    tril_indices = torch.tril_indices(row=dim, col=dim, offset=-1)
    corr_matrix[:, tril_indices[0], tril_indices[1]] = corr

    # build lower triangular covariance
    tril = var_matrix * corr_matrix  # multiply correlations and std's
    tril = tril + torch.diag_embed(std)  # add std's on diagonal

    return tril


def compute_sigma_points(
    mu: torch.Tensor, tril: torch.Tensor, lmd: float = 1e-3
) -> torch.Tensor:
    """
    :param mu: (Tensor) Mean of the Gaussian distribution [B x D]
    :param tril: (Tensor) lower triangular covariance matrix of the Gaussian distribution [B x D x D]
    :return: (Tensor) [B x 2*D+1 x D]
    The returned sigmas are in the following order [mu, s_11, s_21, s_31, ..,  s_D1, s_12, s_22, s_32, ..., s_D2]
        where s_i1 and s_i2 are the sigma points along along the i-th dimension of the gaussian
    """
    batch_size, feat_dim = mu.shape
    max_num_sigmas = 2 * feat_dim + 1

    sigmas = torch.empty(
        (batch_size, max_num_sigmas, feat_dim), device=mu.device
    )  # [B x 2*D+1 x D]

    sigmas[:, 0, :] = mu
    rep_mu = mu.unsqueeze(1).repeat(1, feat_dim, 1)  # repeat mean along rows

    scaled_tril = torch.sqrt(torch.tensor(lmd + feat_dim)) * tril

    sigmas[:, 1 : (feat_dim + 1), :] = rep_mu + scaled_tril
    sigmas[:, (feat_dim + 1) : 2 * feat_dim + 1, :] = rep_mu - scaled_tril

    return sigmas


def sample_with_unscented_transform(
    mu: torch.Tensor, tril: torch.Tensor, num_samples: int, heuristic: str = "random"
) -> torch.Tensor:
    """
    :param mu: (Tensor) Mean of the Gaussian distribution [B x D]
    :param tril: (Tensor) lower triangular covariance matrix of the Gaussian distribution [B x D x D]
    :param num_samples: Number of sigma points to select
    :param heuristic: Sampling heuristic, supported are: random sigma points, mean + random axis-pairs, mean + largest eigval axis-pairs
    :return: (Tensor) [B x S x D], S is the number of samples
    """
    sigmas = compute_sigma_points(mu, tril)

    batch_size, max_num_sigmas, feat_dim = sigmas.shape

    assert (
        num_samples <= max_num_sigmas
    ), f"num_samples has to be smaller than the maximum number of calculated sigma points={max_num_sigmas} but {num_samples} was passed"
    if heuristic == "mean_random_pairs" or heuristic == "mean_top_eigval_pairs":
        assert (
            num_samples % 2 == 1
        ), f"num_samples has to be odd for heuristic={heuristic}"
    if heuristic == "random_pairs" or heuristic == "top_eigval_pairs":
        assert (
            num_samples % 2 == 0
        ), f"num_samples has to be even for heuristic={heuristic}"

    if heuristic == "random":
        """
        Pick random sigma points
        """
        indices = [
            np.random.choice([*range(max_num_sigmas)], size=num_samples, replace=False)
            for _ in range(batch_size)
        ]
        indices = np.stack(indices)
        indices = torch.tensor(indices, device=mu.device, dtype=torch.long)

    elif heuristic == "random_pairs":
        """
        Pick (num_samples - 1)//2 random pairs of sigma points.
        """
        indices = [
            np.random.choice(
                [*range(0, feat_dim)], size=num_samples // 2, replace=False
            )
            for _ in range(batch_size)
        ]
        indices = np.stack(indices)
        indices = torch.tensor(indices, device=mu.device, dtype=torch.long)

        indices = torch.cat([indices + 1, indices + feat_dim + 1], dim=1)

    elif heuristic == "top_eigval_pairs":
        """
        Pick (num_samples - 1)//2 pairs of sigma points with largest displacement along a given axis, determined by the corresponding eigenvalues. Eigenvalues are approximated by the diagonal entries of the tril matrix.
        """
        # select sigma points with highest eigenvalue
        num_relevant_dimensions = num_samples // 2
        eig_values = torch.diagonal(tril, dim1=1, dim2=2)
        indices = torch.topk(eig_values, dim=1, k=num_relevant_dimensions)[1]

        indices = torch.cat([indices + 1, indices + feat_dim + 1], dim=1)

    elif heuristic == "mean_random_pairs":
        """
        Pick mean and (num_samples - 1)//2 random pairs of sigma points.
        """
        indices = [
            np.random.choice(
                [*range(0, feat_dim)], size=(num_samples - 1) // 2, replace=False
            )
            for _ in range(batch_size)
        ]
        indices = np.stack(indices)
        indices = torch.tensor(indices, device=mu.device, dtype=torch.long)

        mu_idx = torch.zeros([batch_size, 1], device=mu.device, dtype=torch.long)
        indices = torch.cat([mu_idx, indices + 1, indices + feat_dim + 1], dim=1)

    elif heuristic == "mean_top_eigval_pairs":
        """
        Pick mean and (num_samples - 1)//2 pairs of sigma points with largest displacement along a given axis, determined by the corresponding eigenvalues. Eigenvalues are approximated by the diagonal entries of the tril matrix.
        """
        # select sigma points with highest eigenvalue
        num_relevant_dimensions = (num_samples - 1) // 2
        eig_values = torch.diagonal(tril, dim1=1, dim2=2)
        indices = torch.topk(eig_values, dim=1, k=num_relevant_dimensions)[1]

        mu_idx = torch.zeros([batch_size, 1], device=mu.device, dtype=torch.long)
        indices = torch.cat([mu_idx, indices + 1, indices + feat_dim + 1], dim=1)

    else:
        raise ValueError(f"Sigma point sampling heuristic {heuristic} not supported.")

    sigmas = torch.take_along_dim(sigmas, indices[:, :, None], dim=1)

    return sigmas
