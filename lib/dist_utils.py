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

from typing import Tuple

import torch
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianMixtureModel:
    def __init__(
        self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor
    ):
        """
        Wrapper class for a batched Gaussian Mixture Model of N components.

        Args:
          means: shape (batch_size, num_components, event_dim)
          covariances: shape (batch_size, num_components, event_dim, event_dim)
          weights: (batch_size, num_components)
        """
        assert (
            len(weights.shape) == 2
        ), f"Weights shape {weights.shape}. Expected: (batch_size, num_components)"
        assert (
            len(means.shape) == 3
        ), f"Means shape: {means.shape}. Expected: (batch_size, num_components, event_dim)"
        assert (
            len(covariances.shape) == 4
        ), f"Covariances shape {covariances.shape}. Expected: (batch_size, num_components, event_dim, event_dim)"

        self.batch_size, self.num_components, self.event_dim = means.shape
        assert weights.shape == (
            self.batch_size,
            self.num_components,
        ), f"Shapes of means and weights are not compatible: {means.shape}, {weights.shape}. Expected: (batch_size, num_components, event_dim) and (batch_size, num_components). "
        assert covariances.shape == (
            self.batch_size,
            self.num_components,
            self.event_dim,
            self.event_dim,
        ), f"Shapes of means and covariances are not compatible: {means.shape}, {covariances.shape}. Expected: (batch_size, num_components, event_dim) and (batch_size, num_components, event_dim, event_dim)."

        self.means = means
        self.covariances = covariances
        self.weights = weights

        # construct mixture model using mixture dist and component dist
        self._mixture_dist = Categorical(probs=self.weights, validate_args=False)
        # self.covariances = torch.diag_embed(
        #     torch.ones_like(self.means)
        # )  # uncomment for the integration test to pass
        self._component_dist = MultivariateNormal(
            loc=self.means, covariance_matrix=self.covariances, validate_args=False
        )
        self._mixture_model = MixtureSameFamily(
            mixture_distribution=self._mixture_dist,
            component_distribution=self._component_dist,
            validate_args=False,
        )

    @property
    def uncond_mixture_dist(self):
        return self._mixture_dist

    @property
    def uncond_component_dist(self):
        return self._component_dist

    @property
    def uncond_mixture_model(self):
        return self._mixture_model


class ConditionalGaussianMixtureModel(GaussianMixtureModel):
    def __init__(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        weights: torch.Tensor,
        cond: torch.Tensor,
    ):
        """
        A conditional Gaussian Mixture that is the result of 'cutting' the original mixture at the value of the conditioning vector. The new event dimension is the original dimension reduced by the feature dimension of the conditioning vector.

        See here for the math: https://stats.stackexchange.com/questions/348941/general-conditional-distributions-for-multivariate-gaussian-mixtures

        Args:
            means: shape (batch_size, num_components, event_dim)
            covariances: shape (batch_size, num_components, event_dim, event_dim)
            weights: shape (batch_size, num_components)
            cond: shape (batch_size, cond_dim), cond_dim < event_dim

        """
        super().__init__(means=means, covariances=covariances, weights=weights)

        cond_batch_size, self.cond_dim = cond.shape
        assert (
            cond_batch_size == self.batch_size
        ), f"Conditioning batch size does not match batch size: {cond_batch_size} =/= {self.batch_size}."
        assert (
            0 < self.cond_dim < self.event_dim
        ), f"Invalid dimensionality of the conditioning: {self.cond_dim}. Must be positive and lower than the original event dimension: {self.event_dim}"

        self.cond = cond

        # split moments using the conditioning dimension
        self.means_top, self.means_bot = self.split_means()
        (
            self.covs_top_left,
            self.covs_top_right,
            self.covs_bot_left,
            self.covs_bot_right,
        ) = self.split_covs()

        # compute conditional moments
        self.cond_means, helper_product = self.compute_conditional_means()
        self.cond_covs = self.compute_conditional_covs(helper_product)

        # sanity checks for conditional covariance
        # check that the conditional cov. is pos. definite, commented out for efficiency
        # assert torch.all(
        #     torch.linalg.eigvals(self.cond_covs).real > 0
        # ), "Non-positive definite conditional covariance."
        # check that the conditional cov. is approximately symmetric
        assert torch.allclose(
            self.cond_covs, torch.transpose(self.cond_covs, -1, -2), atol=1e-3
        ), "Non-symmetric conditional covariance."
        # in order to instantiate a MultivariateNormal below, it needs to be 'more' symmetric
        cond_covs_tril = torch.tril(self.cond_covs)
        self.cond_covs = cond_covs_tril + torch.triu(
            torch.transpose(cond_covs_tril, -1, -2), diagonal=1
        )
        assert torch.allclose(
            self.cond_covs, torch.transpose(self.cond_covs, -1, -2), atol=1e-8
        ), "Non-symmetric conditional covariance."

        # compute conditional weights and log probability of conditioning under the bottom-right distribution
        self.cond_weights, self._mix_cond_logprob = self.compute_conditional_weights()

        # renormalize the weights
        self.cond_weights /= torch.sum(self.cond_weights, dim=1, keepdim=True)

        # construct conditional mixture model using mixture dist and component dist
        self._mixture_dist = Categorical(probs=self.cond_weights, validate_args=False)
        self._component_dist = MultivariateNormal(
            loc=self.cond_means, covariance_matrix=self.cond_covs, validate_args=False
        )
        self._cond_mixture_model = MixtureSameFamily(
            mixture_distribution=self._mixture_dist,
            component_distribution=self._component_dist,
            validate_args=False,
        )

    @property
    def cond_mixture_dist(self):
        return self._mixture_dist

    @property
    def cond_component_dist(self):
        return self._component_dist

    @property
    def cond_mixture_model(self):
        return self._cond_mixture_model

    @property
    def mixture_cond_logprob(self):
        return self._mix_cond_logprob

    def split_means(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the mixture means according to the conditioning dimensionality:

        Example: event_dim = 3, cond_dim = 2
            mean = [.]
                    -
                   [.]
                   [.]
            mean_top = [.], means_top =  -
                        -               [.]
                                        [.]

        Computes:
            means_top: shape [batch_size, num_components, event_dim - cond_dim]
            means_bot: shape [batch_size, num_components, cond_dim]

        """
        means_top = self.means[:, :, : -self.cond_dim]
        means_bot = self.means[:, :, -self.cond_dim :]

        return means_top, means_bot

    def split_covs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the mixture covariances according to the conditioning dimensionality.

        Example: event_dim = 3, cond_dim = 2
            covs = [.|..]
                    ----
                   [.|..]
                   [.|..]
            covs_top_left = [.|, covs_top_right = |..], covs_bot_left =  --, covs_bot_right = ---
                             --                   ---                   [.|                   |..]
                                                                        [.|                   |..]
        Computes:
            covs_top_left: shape [batch_size, num_components, event_dim - cond_dim, event_dim - cond_dim]
            covs_top_right: shape [batch_size, num_components, event_dim - cond_dim, cond_dim]
            covs_bot_left: shape [batch_size, num_components, cond_dim, event_dim - cond_dim]
            covs_bot_right: shape [batch_size, num_components, cond_dim, cond_dim]

        """
        covs_top_left = self.covariances[:, :, : -self.cond_dim, : -self.cond_dim]
        covs_top_right = self.covariances[:, :, : -self.cond_dim, -self.cond_dim :]
        covs_bot_left = self.covariances[:, :, -self.cond_dim :, : -self.cond_dim]
        covs_bot_right = self.covariances[:, :, -self.cond_dim :, -self.cond_dim :]

        return covs_top_left, covs_top_right, covs_bot_left, covs_bot_right

    def compute_conditional_means(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the conditional mean according to the equation:

        mu = mu_top + Sigma_top_right @ Sigma_bot_right^-1 (cond - mu_bot)

        Attributes used:
            cond: shape [batch_size, cond_dim]
            means_top: shape [batch_size, num_components, event_dim - cond_dim]
            means_bot: shape [batch_size, num_components, cond_dim]
            covs_top_right: shape [batch_size, num_components, event_dim - cond_dim, cond_dim]
            covs_bot_right: shape [batch_size, num_components, cond_dim, cond_dim]

        Returns:
            cond_means: shape [batch_size, num_components, event_dim - cond_dim]
            prod_top_right_inv_bot_right: matrix product Sigma_top_right @ Sigma_bot_right^-1 that can be reused later, shape: [batch_size, num_components, event_dim - cond_dim, cond_dim]
        """
        # repeat conditioning num_components times
        cond_repeat = self.cond.unsqueeze(1).repeat(1, self.num_components, 1)

        # compute product Sigma_top_right @ Sigma_bot_right^-1, to be reused later
        prod_top_right_inv_bot_right = torch.matmul(
            self.covs_top_right, torch.linalg.inv(self.covs_bot_right)
        )

        cond_means = self.means_top + torch.matmul(
            prod_top_right_inv_bot_right, (cond_repeat - self.means_bot).unsqueeze(-1)
        ).squeeze(-1)

        return cond_means, prod_top_right_inv_bot_right

    def compute_conditional_covs(
        self, prod_top_right_inv_bot_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the conditional covariance according to the equation:

        Sigma = Sigma_top_left - Sigma_top_right @ Sigma_bot_right^-1 @ Sigma_bot_left

        Attributes used:
            covs_top_left: shape [batch_size, num_components, event_dim - cond_dim, event_dim - cond_dim]
            covs_bot_left: shape [batch_size, num_components, cond_dim, event_dim - cond_dim]

        Args:
            prod_top_right_inv_bot_right: matrix product Sigma_top_right @ Sigma_bot_right^-1 that can be reused later, shape: [batch_size, num_components, event_dim - cond_dim, cond_dim]

        Returns:
            cond_covs: shape [batch_size, num_components, event_dim - cond_dim, event_dim - cond_dim]
        """
        cond_covs = self.covs_top_left - torch.matmul(
            prod_top_right_inv_bot_right, self.covs_bot_left
        )

        return cond_covs

    def compute_conditional_weights(self) -> torch.Tensor:
        """
        Computes the conditional weights according to the Bayes rule:

        w_CGMM_i = wp_GMM_i(cond) / mix_cond_logprob --- cond. weight of component i of the conditional GMM

        wp_GMM_i(cond) = w_GMM_i * p_GMM_i(cond) --- weighted probability of the component i of the GMM at the conditioning value

        mix_cond_logprob = sum_i wp_GMM_i(cond) --- sum of all weighted probabilities of the GMM components at the conditioning value

        For numerical stability, computes logs:
        log w_CGMM_i = log(wp_GMM_i(cond) / mix_cond_logprob) = log(p_GMM_i(cond)) + log(w_GMM_i) - log(mix_cond_logprob)

        Attributes used:
            cond: shape [batch_size, cond_dim]
            means_bot: shape [batch_size, num_components, cond_dim]
            covs_bot_right: shape [batch_size, num_components, cond_dim, cond_dim]

        Returns:
            cond_weights: shape [batch_size, num_components]
            mix_cond_logprob: shape [batch_size]
        """
        bot_right_dist = MultivariateNormal(
            loc=self.means_bot,
            covariance_matrix=self.covs_bot_right,
            validate_args=False,
        )
        bot_right_dist_mix = MixtureSameFamily(
            mixture_distribution=Categorical(probs=self.weights, validate_args=False),
            component_distribution=bot_right_dist,
        )

        # evaluate numerator log prob
        cond_repeat = self.cond.unsqueeze(1).repeat(1, self.num_components, 1)
        comp_cond_logprob = bot_right_dist.log_prob(cond_repeat) + torch.log(
            self.weights
        )

        # evaluate denominator log prob
        mix_cond_logprob = bot_right_dist_mix.log_prob(self.cond)

        # compute ratio
        cond_weights = torch.exp(comp_cond_logprob - mix_cond_logprob.unsqueeze(-1))

        return cond_weights, mix_cond_logprob.squeeze(-1)

    def sample(self, num_batchwise_samples: int = 1):
        """
        Draws samples from the conditional Gaussian mixture for every batch element.

        Args:
            num_samples: batch-wise number of samples to take, num_samples=1 draws self.batch_size samples
        Returns:
            samples: shape [num_batchwise_samples, self.batch_size, self.event_dim - cond_dim]
        """
        return self.cond_mixture_model.sample([num_batchwise_samples])

    def sample_mean(self):
        """
        Takes the mean of the highest-weighted component from the conditional Gaussian mixture for every batch element.

        Returns:
            samples: shape [1, self.batch_size, self.event_dim - cond_dim]
        """
        _, top_component = torch.max(self.mixture_dist.probs, dim=1)

        # select mean of top component only
        top_component_mean = torch.take_along_dim(
            self.component_dist.loc, dim=1, indices=top_component[:, None, None]
        )

        # adapt shape for compatibility with sample()
        return top_component_mean.squeeze(1).unsqueeze(0)
