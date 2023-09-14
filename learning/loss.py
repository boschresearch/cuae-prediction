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

from typing import Callable, List

from learning.loss_base import (
    ADE_loss,
    FDE_loss,
    LossBase,
    dual_multimodality_adapter,
    kld_loss,
    minADE_loss,
    minFDE_loss,
    mixture_covar_nll_eval,
    multimodal_trajectory_spread,
    nll_attimestep,
    nll_mixture_attimestep,
    nll_trajectorywise,
    vae_covar_loss,
    vae_loss,
    vae_reconstruction_loss,
)


class OutputLabelKeyLossBase(LossBase):
    """
    Wrapper for losses that have only <output_label_key> as label name parameter
    """

    @property
    def label_names(self) -> List[str]:
        return [self._config.output_label_key]

    @property
    def output_names(self) -> List[str]:
        return [self._config.output_key]


class MinADELoss(OutputLabelKeyLossBase):
    """
    Config is
        {
            "output_key": ...,
            "output_label_key": ...,
            "order": ...,
        }
    """

    @property
    def _loss_fn(self) -> Callable:
        return minADE_loss


class MinFDELoss(OutputLabelKeyLossBase):
    """
    Config is
        {
            "output_key": ...,
            "output_label_key": ...,
            "order": ...,
        }
    """

    @property
    def _loss_fn(self) -> Callable:
        return minFDE_loss


class ADELoss(OutputLabelKeyLossBase):
    """
    Config is
        {
            "output_key": ...,
            "output_label_key": ...,
            "order": ...,
        }
    """

    @property
    def _loss_fn(self) -> Callable:
        return ADE_loss


class FDELoss(OutputLabelKeyLossBase):
    """
    Config is
        {
            "output_key": ...,
            "output_label_key": ...,
            "order": ...,
        }
    """

    @property
    def _loss_fn(self) -> Callable:
        return FDE_loss


class DualMultimodalityAdapter(LossBase):
    @property
    def label_names(self) -> List[str]:
        return self._config.label_names

    @property
    def output_names(self) -> List[str]:
        return self._config.output_names

    @property
    def _loss_fn(self) -> Callable:
        return dual_multimodality_adapter


class VAELoss(LossBase):
    @property
    def label_names(self) -> List[str]:
        return self._config.label_names

    @property
    def output_names(self) -> List[str]:
        return self._config.output_names

    @property
    def _loss_fn(self) -> Callable:
        return vae_loss


class VAEReconstructionLoss(LossBase):
    @property
    def label_names(self) -> List[str]:
        return self._config.label_names

    @property
    def output_names(self) -> List[str]:
        return self._config.output_names

    @property
    def _loss_fn(self) -> Callable:
        return vae_reconstruction_loss


class KLDLoss(LossBase):
    @property
    def label_names(self) -> List[str]:
        return [self._config.prior_mu_key, self._config.prior_cov_key]

    @property
    def output_names(self) -> List[str]:
        return [self._config.posterior_mu_key, self._config.posterior_cov_key]

    @property
    def _loss_fn(self) -> Callable:
        return kld_loss


class MultimodalTrajectorySpread(LossBase):
    @property
    def output_names(self) -> List[str]:
        return self._config.output_names

    @property
    def label_names(self) -> List[str]:
        return self._config.label_names

    @property
    def _loss_fn(self) -> Callable:
        return multimodal_trajectory_spread


class NLLTrajectorywise(LossBase):
    @property
    def label_names(self) -> List[str]:
        return [self._config.label_key]

    @property
    def output_names(self) -> List[str]:
        return [self._config.output_mean_key, self._config.output_std_key]

    @property
    def _loss_fn(self) -> Callable:
        return nll_trajectorywise


class NLLAtTimestep(LossBase):
    @property
    def label_names(self) -> List[str]:
        return [self._config.label_key]

    @property
    def output_names(self) -> List[str]:
        return [self._config.output_mean_key, self._config.output_std_key]

    @property
    def _loss_fn(self) -> Callable:
        return nll_attimestep


class VAECovarLoss(LossBase):
    @property
    def label_names(self) -> List[str]:
        return self._config.label_names

    @property
    def output_names(self) -> List[str]:
        return self._config.output_names

    @property
    def _loss_fn(self) -> Callable:
        return vae_covar_loss


class VAEMixtureCovarNLLLoss(LossBase):
    @property
    def label_names(self) -> List[str]:
        return self._config.label_names

    @property
    def output_names(self) -> List[str]:
        return self._config.output_names

    @property
    def _loss_fn(self) -> Callable:
        return mixture_covar_nll_eval


class MixtureNLLAtTimestep(LossBase):
    @property
    def label_names(self) -> List[str]:
        return [self._config.label_key]

    @property
    def output_names(self) -> List[str]:
        return [
            self._config.output_mean_key,
            self._config.output_std_key,
            self._config.output_weights.key,
        ]

    @property
    def _loss_fn(self) -> Callable:
        return nll_mixture_attimestep
