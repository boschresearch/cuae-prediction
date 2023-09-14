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

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning.policies import Policy, PolicyPin
from learning.policies_base import FlexibleMLP, LoopingGRU
from lib.action_constraints import ActionConstraints
from lib.matrix_utils import build_tril_matrix


class GRUPredictionHeadBase(Policy):
    def __init__(
        self,
        gru_params: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        input_lin_layers: List[Dict] = [],
        num_modes: int = 1,
        data_source: str = "default",
    ) -> None:
        assert len(inputs) == 2, "Only two input features are supported"
        assert len(outputs) == 2, "Only two output are supported"

        self.backbone_features_input: PolicyPin = None
        self.recurrent_features_input: PolicyPin = None
        self.recurrent_features_output: PolicyPin = None
        self.probability_output: PolicyPin = None

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            input_lin_layers=input_lin_layers,
            gru_params=gru_params,
            num_modes=num_modes,
            data_source=data_source,
        )

        assert (
            len(self.backbone_features_input.shape) == 1
            and len(self.recurrent_features_input.shape) == 1
        ), "Only 1-dim input feature vectors are supported. recurrent_features_input should be encoded into a single feature vector."

        if not self.recurrent_features_output.shape[2] == 2:
            raise ValueError(
                f"output shape = {self.recurrent_features_output.shape} must be of the form [num_modes, num_frames, 2]"
            )

        self.input_dim = np.prod(self.backbone_features_input.shape) + np.prod(
            self.recurrent_features_input.shape
        )
        self._num_modes = num_modes
        self.num_frames = self.recurrent_features_output.shape[1]
        self.feature_dim = self.recurrent_features_output.shape[2]
        self.output_dim = (
            self.num_frames * self.feature_dim * num_modes + num_modes
        )  # num_features + pobability for each mode

        self.input_lin = FlexibleMLP(
            self.input_dim, input_lin_layers, gru_params["input_dim"]
        )
        self.gru = LoopingGRU(
            in_dim=gru_params["input_dim"],
            hidden_dim=gru_params["hidden_dim"],
            num_layers=gru_params["num_layers"],
            num_loops=gru_params["num_loops"],
        )
        self.output_lin = nn.Linear(gru_params["input_dim"], self.output_dim)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        backbone_feats = x[self.backbone_features_input.key]
        backbone_feats = backbone_feats.view(backbone_feats.shape[0], -1)
        recurrent_features = x[self.recurrent_features_input.key]
        x = torch.cat((backbone_feats, recurrent_features), dim=1)
        x = self.input_lin(x)
        x = self.gru(x.unsqueeze(1))  # sequence of length 1
        x = self.output_lin(x.squeeze(1))
        return x

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(x[:, -self._num_modes :], dim=1)

    def get_recurrent_features_output(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, : -self._num_modes].view(
            -1,
            self.num_frames * self._num_modes,
            self.feature_dim,
        )

    def reshape_recurrent_features_output(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(
            -1,
            self._num_modes,
            self.num_frames,
            self.feature_dim,
        )


class GRUActionHead(GRUPredictionHeadBase):
    """
    Configurable ActionPredictor network:
        -> MLP &
        -> LoopingGRU
    Takes as input:
        context_embedding vector z.
            (Past context embeddings z_t or future and past context embeddings [z_t, z_t+1])
        past_actions_encoding vector a_t

    Predicts multimodal future_actions and a probability for each mode
        z,  a_t -> hat{a}_t+1

    Actions are a list [ac_0, ac_1, ..., ac_n]
    with ac_i = [a, delta_steer] and n = num_frames - 1

    Actions output shape: [batch_size, num_modes, num_frames, 2]
    Probabilities output shape: [batch_size, num_modes]
    """

    def __init__(
        self,
        gru_params: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        input_lin_layers: List[Dict] = [],
        num_modes: int = 1,
        data_source: str = "default",
    ) -> None:
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            input_lin_layers=input_lin_layers,
            gru_params=gru_params,
            num_modes=num_modes,
            data_source=data_source,
        )

        self.tanh = nn.Tanh()
        self.action_scaling = ActionConstraints(dataset=data_source)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.tanh(super().forward(x))

        mode_probabilities = self.get_probabilities(x)

        action_predictions = self.get_recurrent_features_output(x)
        action_predictions = self.action_scaling(action_predictions)
        action_predictions = self.reshape_recurrent_features_output(action_predictions)

        return {
            self.recurrent_features_output.key: action_predictions,
            self.probability_output.key: mode_probabilities,
        }


class RegularizableGRUActionHead(GRUActionHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def weights(self) -> torch.Tensor:
        layer_weights = []
        for layer in self.input_lin.layers:
            layer_weights.extend([param.flatten() for param in layer.parameters()])
        layer_weights.extend([param.flatten() for param in self.gru.gru.parameters()])
        layer_weights.extend([param.flatten() for param in self.gru.fc.parameters()])
        return torch.cat(layer_weights)


class NormalDistributionMLP(Policy):
    """
    Simple policy to predict moments of a multivariate Gaussian distribution.
    The MLP structure is shared apart from the last layer predicting means / log variances / correlations.
    If zero_correlation == True, then all correlations are set to zero and a diagonal-covariance distribution is modeled. Otherwise, full covariance matrix is modeled.

    The outputs are the mean vector and a lower triangular tril matrix L, Sigma=LL^T.
    """

    def __init__(
        self,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        hidden_layer_dims: List[Dict],
        zero_correlation: bool,
        data_source: str = "default",
    ) -> None:
        """
        Args:
            hidden_layer_dims: Tunes network linear layer structure, is a list of {"dim": linear layer dim, "norm": normalization layer, "activation": activation function}
            zero_correlation: Tunes diagonal or full covariance
        """
        self.conditional_input: PolicyPin = None
        self.mu_output: PolicyPin = None
        self.tril_output: PolicyPin = None

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            data_source=data_source,
        )

        assert (
            self.tril_output.shape[1] == self.tril_output.shape[0]
        ), "tril matrix must be quadratic"
        assert (
            self.tril_output.shape[0] == self.mu_output.shape[0]
        ), "dimension of the normal distribution must be equal for mu and tril"

        self.include_correlation = not zero_correlation

        latent_distribution_dim = self.mu_output.shape[0]

        self.layers = FlexibleMLP(
            input_dim=self.conditional_input.shape[0], lin_layers=hidden_layer_dims
        )

        self.mu_layer = nn.Linear(hidden_layer_dims[-1]["dim"], latent_distribution_dim)
        self.logvar_layer = nn.Linear(
            hidden_layer_dims[-1]["dim"], latent_distribution_dim
        )

        self.corr_dim = (latent_distribution_dim * (latent_distribution_dim - 1)) // 2
        if self.include_correlation:
            self.corr_layer = nn.Linear(hidden_layer_dims[-1]["dim"], self.corr_dim)

        self.tanh = nn.Tanh()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_tensor: torch.Tensor = x[self.conditional_input.key]

        layer_output = self.layers(x_tensor)
        mu = self.mu_layer(layer_output)
        logvar = self.logvar_layer(layer_output)

        if self.include_correlation:
            corr = self.tanh(self.corr_layer(layer_output))
        else:
            batch_size = layer_output.shape[0]
            corr = torch.zeros(
                size=[batch_size, self.corr_dim], device=layer_output.device
            )

        tril = build_tril_matrix(logvar=logvar, corr=corr)

        return {self.mu_output.key: mu, self.tril_output.key: tril}


class GMMDistributionPredictor(Policy):
    """
    Simple policy to predict parameters of a GMM of multivariate Gaussian distributions.
    Consists of n_components NormalDistributionsMLPs and one MLP to predict the components weights.

    The returned parameters are the mean vectors, lower triangular tril matrices L, Sigma=LL^T and the components weights
    """

    def __init__(
        self,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        hidden_layer_dims: List[Dict],
        zero_correlation: bool,
        n_components: int,
        data_source: str = "default",
        random_seed: Optional[int] = None,
    ) -> None:

        self.conditional_input: PolicyPin = None
        self.mu_output: PolicyPin = None
        self.tril_output: PolicyPin = None
        self.component_weights_output: PolicyPin = None

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            data_source=data_source,
            random_seed=random_seed,
        )

        self.component_normal_predictors = torch.nn.ModuleList(
            [
                NormalDistributionMLP(
                    inputs=[self.conditional_input],
                    outputs=[self.mu_output, self.tril_output],
                    hidden_layer_dims=hidden_layer_dims,
                    zero_correlation=zero_correlation,
                    data_source=data_source,
                )
                for _ in range(n_components)
            ]
        )

        self.components_weights_predictor = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=1),
            FlexibleMLP(
                input_dim=self.conditional_input.shape[0]
                + 2
                * self.mu_output.shape[
                    0
                ],  # weights predictor doesn't consider covariances, input: mean and std of each component [B, N, 2*D+C]
                lin_layers=hidden_layer_dims,
            ),
            nn.Unflatten(dim=0, unflattened_size=[-1, n_components]),
            nn.Linear(
                hidden_layer_dims[-1]["dim"], 1
            ),  # output: weight for each component [B, N, 1]
            nn.Flatten(
                start_dim=-2, end_dim=-1
            ),  # after flattening last dimension to [B,N] softmax is applied
            nn.Softmax(dim=-1),
        )

        self.n_components = n_components

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run module

        returns: mu: [B, N, D], tril: [B, N, D, D], components_weights: [B, N]
            N: n_components, D: dimension of latent distribution
        """
        # infer mixture components
        components_params = [cp.forward(x) for cp in self.component_normal_predictors]

        # stack loc and scale so that mixture component is rightmost dimension (i.e. dim=1)
        components_mu = torch.stack(
            [cp[self.mu_output.key] for cp in components_params], dim=1
        )
        components_tril = torch.stack(
            [cp[self.tril_output.key] for cp in components_params], dim=1
        )

        weigths_predictor_inputs = torch.cat(
            [
                components_mu,
                # components weights predictor doesn't consider correlations
                # only use diagonal of tril
                torch.diagonal(components_tril, dim1=-2, dim2=-1),
                torch.tile(
                    x[self.conditional_input.key][:, None, :], (1, self.n_components, 1)
                ),
            ],
            dim=-1,
        )
        components_weights = self.components_weights_predictor.forward(
            weigths_predictor_inputs
        )

        return {
            self.mu_output.key: components_mu,
            self.tril_output.key: components_tril,
            self.component_weights_output.key: components_weights,
        }
