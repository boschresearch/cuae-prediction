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

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from pycave.bayes.gmm import GaussianMixture
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal

from features_and_labels.features_generators import MapPolylineVectorFeaturesGenerator
from features_and_labels.labels_generators import TrafficPosesLabelsGenerator
from learning.clustering_utils import clustering_function
from learning.policies import (
    FromToPinConnection,
    Policy,
    PolicyPin,
    policy_from_policy_config,
)
from learning.prediction_models.encoders import StarGraphContextEncoderConfig
from lib.base_classes import EGO_STATE, STATE_EGO_ATTRIBUTES
from lib.dist_utils import ConditionalGaussianMixtureModel
from lib.kinematic_models import KinematicBicycleModel
from lib.matrix_utils import compute_sigma_points, sample_with_unscented_transform
from lib.utils import class_from_path, instantiate_from_config


@dataclass
class ContextEncoderConfig:
    path_to_class: str
    config: dict


class ActionSpacePredictorBase(Policy):

    """
    Policy base class for action-space-prediction models with following submodels:
        ContextEncoder from torchvision,
        ActionsEncoder, ActionsPredictor,
        KinematicModel
    """

    context_encoder_name = "context_encoder"
    actions_encoder_name = "actions_encoder"
    actions_predictor_name = "actions_predictor"
    kinematic_model_name = "kinematic_model"

    def __init__(
        self,
        submodel_configs: dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        inputs_sample_times: List[float],
        outputs_sample_times: List[float],
        num_modes: int = 1,
        num_output_timesteps: Optional[int] = None,
        data_source: str = "default",
        **kwargs,
    ) -> None:
        self.check_submodel_configs_exist(submodel_configs)

        self.past_actions_features_input: PolicyPin = None
        self.current_ego_state_features_input: PolicyPin = None

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            inputs_sample_times=inputs_sample_times,
            outputs_sample_times=outputs_sample_times,
            submodel_configs=submodel_configs,
            num_modes=num_modes,
            num_output_timesteps=num_output_timesteps,
            data_source=data_source,
            **kwargs,
        )

        self.data_source = data_source
        self.num_output_timesteps = num_output_timesteps

        self.num_modes = num_modes
        self._unimodal_waypoints_shape = (self.num_output_timesteps, 2)

        # ActionsEncoder
        self.actions_encoder_input = PolicyPin(
            name="actions_encoder_input",
            key="actions",
            shape=self.past_actions_features_input.shape,
        )

        self.actions_encoder = self.prepare_actions_encoder(
            submodel_configs[self.actions_encoder_name]
        )
        self._actions_encoding_shape = self.actions_encoder.forward(
            torch.rand(self.past_actions_features_input.shape).unsqueeze(
                0
            )  # random input data
        ).shape[1:]

        self.actions_encoder_output = PolicyPin(
            name="actions_encoder_output",
            key="actions_encoding",
            shape=self._actions_encoding_shape,
        )

        # ActionsPredictor
        # get context encoding shape as input to the actions predictor
        self._context_encoding_shape = (
            submodel_configs[self.context_encoder_name]["config"]["output_dim"],
        )

        actions_predictor_context_embedding_input_shape = (
            self._context_encoding_shape[0]
            * submodel_configs["actions_predictor"]["num_context_inputs"],
        )  # num_context_inputs -> to allow as input (past or future)=1 as well as (past & future)=2 context inputs

        self.actions_predictor_context_embedding_input = PolicyPin(
            name="backbone_features_input",
            key="context_embedding",
            shape=actions_predictor_context_embedding_input_shape,
        )

        self.actions_encoder_to_actions_predictor_mapping = (
            FromToPinConnection.create_from_from_pin(
                self.actions_encoder_output,
                to_name="recurrent_features_input",
            )
        )

        self.actions_predictor_probabilities_output = PolicyPin(
            name="probability_output",
            key="mode_probabilities",
            shape=(self.num_modes,),
        )

        self.actions_predictor_actions_output = PolicyPin(
            name="recurrent_features_output",
            key="multimodal_actions",
            shape=(self.num_modes,) + self._unimodal_waypoints_shape,
        )

        self.kinematic_model_actions_input = PolicyPin(
            name="actions_input",
            key="actions",
            shape=self._unimodal_waypoints_shape,
        )

        self.actions_predictor = self.prepare_actions_predictor(
            submodel_configs[self.actions_predictor_name],
            inputs=[
                self.actions_predictor_context_embedding_input,
                self.actions_encoder_to_actions_predictor_mapping.to_pin,
            ],
            outputs=[
                self.actions_predictor_actions_output,
                self.actions_predictor_probabilities_output,
            ],
            num_modes=self.num_modes,
            data_source=data_source,
        )

        # KinematicModel
        sample_time = outputs_sample_times[0]
        self.state_ego_attributes_to_kinematic_model_mapping = (
            FromToPinConnection.create_from_from_pin(
                self.current_ego_state_features_input,
                to_name="ego_state_features_input",
            )
        )
        self.kinematic_model_ego_state_output = PolicyPin(
            name="ego_states_output",
            key="ego_states",
            shape=(self.num_output_timesteps, len(EGO_STATE)),
        )
        self.kinematic_model = self.prepare_kinematic_model(
            submodel_configs[self.kinematic_model_name],
            inputs=[
                self.kinematic_model_actions_input,
                self.state_ego_attributes_to_kinematic_model_mapping.to_pin,
            ],
            outputs=[self.kinematic_model_ego_state_output],
            sample_time=sample_time,
        )

    @property
    def submodel_names(self) -> List[str]:
        return [
            self.context_encoder_name,
            self.actions_encoder_name,
            self.actions_predictor_name,
            self.kinematic_model_name,
        ]

    @property
    def num_output_timesteps(self):
        return self._num_output_timesteps

    @num_output_timesteps.setter
    def num_output_timesteps(self, num_output_timesteps: Union[int, None]):
        """
        Custom setter for `num_output_timesteps` in case it's not provided in the config. INTERACTION models should always predict 30 timesteps, otherwise it's inferred from the length of past actions features.
        """
        if num_output_timesteps is None:
            # INTERACTION models are always 3s prediction at 10Hz
            if self.data_source == "interaction":
                self._num_output_timesteps = 30
            else:
                # past actions contain one fewer timestep than predicted
                self._num_output_timesteps = (
                    self.past_actions_features_input.shape[0] + 1
                )
        else:
            self._num_output_timesteps = num_output_timesteps

    def check_submodel_configs_exist(self, submodel_configs: dict) -> None:
        provided_submodel_names = list(submodel_configs.keys())

        if not all(name in provided_submodel_names for name in self.submodel_names):
            raise Exception(
                f"Provided submodel configs: {provided_submodel_names}, but required are at least: {self.submodel_names}.\n"
            )

    def prepare_actions_encoder(self, actions_encoder_config: dict) -> torch.nn.Module:
        model_cf = actions_encoder_config["config"]
        return class_from_path(actions_encoder_config["path_to_class"]).from_config(
            model_cf
        )

    def prepare_actions_predictor(
        self,
        actions_predictor_config: dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        num_modes: int,
        data_source: str,
    ) -> torch.nn.Module:
        model_cf = actions_predictor_config["config"]
        model_cf["inputs"] = inputs
        model_cf["outputs"] = outputs
        model_cf["data_source"] = data_source
        model_cf["num_modes"] = num_modes
        return policy_from_policy_config(
            {
                "path_to_class": actions_predictor_config["path_to_class"],
                "config": model_cf,
            }
        )

    def prepare_kinematic_model(
        self,
        kinematic_model: dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        sample_time: float,
    ) -> torch.nn.Module:
        model_cf = kinematic_model["config"]
        model_cf["inputs"] = inputs
        model_cf["outputs"] = outputs
        model_cf["sample_time"] = sample_time
        model_cf["attributes"] = STATE_EGO_ATTRIBUTES
        return class_from_path(kinematic_model["path_to_class"]).from_config(model_cf)

    def get_waypoints_from_unimodal_states(
        self, ego_states: torch.Tensor
    ) -> torch.Tensor:
        """
        param: ego_states: [x, y, yaw, v], shape: [batch_size, num_timesteps, 4]
        return: positions [x, y], shape: [batch_size, num_timesteps, 2]
        """
        return ego_states[:, :, :2]

    def get_waypoints_from_multimodal_states(
        self, multimodal_ego_states: torch.Tensor
    ) -> torch.Tensor:
        """
        param: multimodal_ego_states: [x, y, yaw, v], shape: [batch_size, num_modes, num_timesteps, 4]
        return: ego_states [x, y], shape: [batch_size, num_modes, num_timesteps, 2]
        """
        return multimodal_ego_states[:, :, :, :2]

    def move_modes_dim_into_batch_dim(self, input: torch.Tensor) -> torch.Tensor:
        """
        Moves modes (dim=1) into the batch-dimension (dim=0)
        :param: input [batch_size, num_modes, sample_dim]
        :return: batched_input [batch_size*num_modes, sample_dim]
        """
        return input.view(-1, *input.shape[2:])

    def move_modes_from_batch_dim_into_mode_dim(
        self, input: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Moves modes from batch-dimension (dim=0) back into mode dimension (dim=1)
        :param: batched_input [batch_size*num_modes, sample_dim]
        :param: original batch_size (unbatched)
        :return: input [batch_size, num_modes, sample_dim]
        """
        return input.view(batch_size, -1, *input.shape[1:])

    def repeat_and_batch_tensor(self, input: torch.Tensor, factor: int) -> torch.Tensor:
        """
        Repeats an input tensor in the mode dimension (dim=1) factor times
        and moves the mode dimension afterwards into the batch dimension (dim=0) (batching).
        :param: input [batch_size, sample_dim]
        :param: factor: int: =>1, number of times each sample needs to be repeated
        :return: batched-repeated-tensor [batch_size*factor, sample_dim]
        """
        assert factor >= 1

        input_len = len(input.shape)
        # Repeat tensor along mode dimension [batch_size, factor, sample_dim]
        input = input.unsqueeze(1).repeat(
            1,
            factor,
            *[1] * (input_len - 1),
        )
        return self.move_modes_dim_into_batch_dim(input)

    def get_multimodal_ego_states_via_kinematic_model(
        self,
        multimodal_actions: torch.Tensor,
        current_state_ego_attributes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate each actions mode through a kinematic model and build multi-modal ego-states [x, y, yaw, v]
            The implementation is batched: Actions modes are moved temporarily into the batch-dimension
        param: multimodal_actions: [batch_size, num_modes, num_timesteps, 2]
        param: current_state_ego_attributes in own reference frame: [batch_size, 6]
        return: ego_states_modes: [x, y, yaw, v], shape: [batch_size, num_modes, num_timesteps, 4]
        """

        batch_size = multimodal_actions.shape[0]
        num_modes = multimodal_actions.shape[1]
        # batch
        multimodal_actions = self.move_modes_dim_into_batch_dim(multimodal_actions)
        current_state_ego_attributes = self.repeat_and_batch_tensor(
            current_state_ego_attributes, num_modes
        )  # create state sample for each mode

        ego_states = self.kinematic_model.forward(
            {
                self.kinematic_model_actions_input.key: multimodal_actions,
                self.state_ego_attributes_to_kinematic_model_mapping.key: current_state_ego_attributes,
            }
        )[self.kinematic_model_ego_state_output.key]

        return self.move_modes_from_batch_dim_into_mode_dim(
            ego_states, batch_size
        )  # unbatch

    def get_multimodal_waypoints_via_kinematic_model(
        self,
        multimodal_actions: torch.Tensor,
        current_state_ego_attributes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate each actions mode through a kinematic model and build multi-modal waypoints modes
        param: multimodal_actions: [batch_size, num_modes, num_timesteps, 2]
        param: current_state_ego_attributes: [batch_size, 6]
        return: multimodal_waypoints: [x, y, yaw, v], shape: [batch_size, num_modes, num_timesteps, 4]
        """
        multimodal_ego_states = self.get_multimodal_ego_states_via_kinematic_model(
            multimodal_actions, current_state_ego_attributes
        )
        multimodal_waypoints = self.get_waypoints_from_multimodal_states(
            multimodal_ego_states
        )
        return multimodal_waypoints


class StarNetFeedForwardActionSpacePredictor(ActionSpacePredictorBase):
    """
    Policy for action-space-prediction with following submodels:
        ContextEncoder StarNet,
        ActionsEncoder, ActionsPredictor,
        KinematicModel
    """

    def __init__(
        self,
        submodel_configs: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        inputs_sample_times: List[float],
        outputs_sample_times: List[float],
        num_modes: int = 1,
        num_output_timesteps: Optional[int] = None,
        data_source: str = "default",
        **kwargs,
    ) -> None:
        input_names = {input.name for input in inputs}
        assert (
            input_names == StarNetFeedForwardActionSpacePredictor.get_input_names()
        ), f"Model should have inputs: {StarNetFeedForwardActionSpacePredictor.get_input_names()}, received: {input_names}"
        output_names = {output.name for output in outputs}
        assert (
            output_names == StarNetFeedForwardActionSpacePredictor.get_output_names()
        ), f"Model should have outputs: {StarNetFeedForwardActionSpacePredictor.get_output_names()}, received: {output_names}"

        self.past_ego_waypoints_features_input: PolicyPin = None
        self.past_polyline_features_input: PolicyPin = None
        self.past_traffic_poses_features_input: PolicyPin = None
        self.multimodal_future_waypoints_output: PolicyPin = None

        self.check_submodel_configs_exist(submodel_configs)

        # only past context_encoding is used as context input for the ActionsPredictor
        submodel_configs["actions_predictor"]["num_context_inputs"] = 1

        super().__init__(
            submodel_configs=submodel_configs,
            inputs=inputs,
            outputs=outputs,
            inputs_sample_times=inputs_sample_times,
            outputs_sample_times=outputs_sample_times,
            num_modes=num_modes,
            num_output_timesteps=num_output_timesteps,
            data_source=data_source,
            **kwargs,
        )

        # Star Graph Context Encoder Inputs
        input_waypoints_num_timesteps = self.past_actions_features_input.shape[0] + 1
        star_graph_encoder_ego_waypoints_input_shape = (
            input_waypoints_num_timesteps,
            submodel_configs[self.context_encoder_name]["config"][
                "waypoints_encoder_config"
            ]["config"]["num_input_channels"],
        )
        self.star_graph_encoder_ego_waypoints_input = PolicyPin(
            name="ego_waypoints_input",
            key="ego_waypoints",
            shape=star_graph_encoder_ego_waypoints_input_shape,
        )
        star_graph_encoder_polyline_features_input_shape = (
            MapPolylineVectorFeaturesGenerator.MAX_NUM_POLYLINE_MAP_VECTORS,
            len(MapPolylineVectorFeaturesGenerator.MAP_POLYLINE_VECTOR_ATTRIBUTES),
        )
        self.star_graph_encoder_polyline_features_input = PolicyPin(
            name="polyline_features_input",
            key="polyline_features",
            shape=star_graph_encoder_polyline_features_input_shape,
        )
        star_graph_encoder_traffic_poses_input_shape = (
            star_graph_encoder_ego_waypoints_input_shape[0],
            TrafficPosesLabelsGenerator.MAX_TRAFFIC_OBJECTS,
            len(TrafficPosesLabelsGenerator.TRAFFIC_OBJECT_ATTRIBUTES),
        )
        self.star_graph_encoder_traffic_poses_input = PolicyPin(
            name="traffic_poses_input",
            key="traffic_poses",
            shape=star_graph_encoder_traffic_poses_input_shape,
        )

        # Star Graph Context Encoder Output
        star_graph_encoder_polyline_features_output_shape = (
            submodel_configs[self.context_encoder_name]["config"]["output_dim"],
        )
        self.star_graph_encoder_polyline_features_output = PolicyPin(
            name="context_features_output",
            key="context_features",
            shape=star_graph_encoder_polyline_features_output_shape,
        )

        # ContextEncoder
        self.context_encoder = self.prepare_context_encoder(
            submodel_configs[self.context_encoder_name],
            inputs=[
                self.star_graph_encoder_polyline_features_input,
                self.star_graph_encoder_ego_waypoints_input,
                self.star_graph_encoder_traffic_poses_input,
            ],
            outputs=[
                self.star_graph_encoder_polyline_features_output,
            ],
        )

    @staticmethod
    def get_input_names() -> Set[str]:
        return {
            "past_ego_waypoints_features_input",
            "past_polyline_features_input",
            "past_traffic_poses_features_input",
            "past_actions_features_input",
            "current_ego_state_features_input",
        }

    @staticmethod
    def get_output_names() -> Set[str]:
        return {"multimodal_future_waypoints_output"}

    def prepare_context_encoder(
        self,
        context_encoder_config: dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
    ) -> torch.nn.Module:
        context_encoder_config["config"]["inputs"] = inputs
        context_encoder_config["config"]["outputs"] = outputs
        star_graph_context_encoder_config = ContextEncoderConfig(
            path_to_class=context_encoder_config["path_to_class"],
            config=StarGraphContextEncoderConfig(**context_encoder_config["config"]),
        )
        return instantiate_from_config(star_graph_context_encoder_config)

    def call_context_encoder(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Wrapper for calling the context encoder forward() with the correct inputs.
        StarNetFeedForwardActionSpacePredictor() uses past ego waypoints, traffic poses, and polylines for the GNN encoder.
        """
        return self.context_encoder.forward(
            {
                self.star_graph_encoder_polyline_features_input.key: inputs[
                    self.past_polyline_features_input.key
                ],
                self.star_graph_encoder_ego_waypoints_input.key: inputs[
                    self.past_ego_waypoints_features_input.key
                ],
                self.star_graph_encoder_traffic_poses_input.key: inputs[
                    self.past_traffic_poses_features_input.key
                ],
            }
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # context_encoding via star graph encoder
        context_encoding = self.call_context_encoder(inputs)

        # actions encoding
        actions_encoding = self.actions_encoder(
            inputs[self.past_actions_features_input.key]
        )

        # predict actions
        multimodal_future_actions = self.actions_predictor.forward(
            {
                self.actions_predictor_context_embedding_input.key: context_encoding,
                self.actions_encoder_to_actions_predictor_mapping.key: actions_encoding,
            }
        )[self.actions_predictor_actions_output.key]

        current_state_ego_attributes = KinematicBicycleModel.get_state_ego_attributes(
            inputs[self.current_ego_state_features_input.key][:, 0]
        )

        # convert actions to waypoints
        multimodal_future_waypoints = self.get_multimodal_waypoints_via_kinematic_model(
            multimodal_actions=multimodal_future_actions,
            current_state_ego_attributes=current_state_ego_attributes,
        )

        return {
            self.multimodal_future_waypoints_output.key: multimodal_future_waypoints
        }


class CVAEActionSpacePredictor(Policy):
    """
    CVAE action-space model that expands one of the existing base models into a CVAE structure.
    Supported base models are FeedForwardActionSpacePredictor and StarNetFeedForwardActionSpacePredictor.
    Instantiates a base model delegate and depending on the functionality, calls the base model instance or the CVAE-inherent functions.
    For example, context encoding is done within the delegate base model and latent space sampling is done within the CVAE delegator.
    """

    posterior_predictor_name = "posterior_predictor"
    prior_predictor_name = "prior_predictor"
    gt_encoder_name = "gt_encoder"

    @property
    def submodel_names(self) -> List[str]:
        return [
            self.context_encoder_name,
            self.actions_encoder_name,
            self.gt_encoder_name,
            self.actions_predictor_name,
            self.kinematic_model_name,
            self.posterior_predictor_name,
            self.prior_predictor_name,
        ]

    def __init__(
        self,
        base_model_name: str,
        submodel_configs: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        inputs_sample_times: List[float],
        outputs_sample_times: List[float],
        latent_sampling: dict,
        zero_correlation: bool,
        num_z_samples: int,
        num_modes: int = 1,  # num modes per sample
        latent_z_dim: int = 64,
        data_source: str = "default",
        num_output_timesteps: Optional[int] = None,
        output_samples_handling: dict = {"method": "no_averaging"},
        **kwargs,
    ) -> None:
        SUPPORTED_BASE_MODELS = {
            "starnet": StarNetFeedForwardActionSpacePredictor,
        }
        SUPPORTED_SAMPLING_METHODS = {
            "random": self.random_sampling,
            "none": self.no_sampling,
            "unscented": self.unscented_sampling,
        }
        SUPPORTED_OUTPUT_SAMPLES_HANDLING = {
            "no_averaging": self.no_averaging,
            "averaging": self.average_outputs,
            "averaging_with_clustering": self.average_outputs_with_clustering,
        }

        assert "method" in latent_sampling.keys() and (
            latent_sampling["method"] in SUPPORTED_SAMPLING_METHODS.keys()
        ), f'latent_sampling has to contain "method" key taking one of: {SUPPORTED_SAMPLING_METHODS.keys()} values. Received latent_sampling: {latent_sampling}'
        assert "method" in output_samples_handling.keys() and (
            output_samples_handling["method"]
            in SUPPORTED_OUTPUT_SAMPLES_HANDLING.keys()
        ), f'output_samples_handling has to contain "method" key taking one of: {SUPPORTED_OUTPUT_SAMPLES_HANDLING.keys()} values. Received output_samples_handling: {output_samples_handling}'

        submodel_configs_ = copy.deepcopy(submodel_configs)

        self.latent_sampling = latent_sampling
        self.sample_z = SUPPORTED_SAMPLING_METHODS[self.latent_sampling["method"]]
        self.latent_z_dim = latent_z_dim
        self.num_z_samples = num_z_samples
        self.output_samples_handling = output_samples_handling
        self.handle_output_samples = SUPPORTED_OUTPUT_SAMPLES_HANDLING[
            self.output_samples_handling["method"]
        ]

        # context encoder policy pins
        if base_model_name == "ffasp":
            self.past_context_features_input: PolicyPin = None
        elif base_model_name == "starnet":
            self.past_ego_waypoints_features_input: PolicyPin = None
            self.past_polyline_features_input: PolicyPin = None
            self.past_traffic_poses_features_input: PolicyPin = None
        else:
            raise ValueError(
                f"Invalid base model: {base_model_name}. Supported base models: {SUPPORTED_BASE_MODELS.keys()}"
            )

        # action-space policy pins
        self.past_actions_features_input: PolicyPin = None
        self.current_ego_state_features_input: PolicyPin = None
        self.future_actions_features_input: PolicyPin = None
        self.future_waypoints_features_input: PolicyPin = None

        # CVAE-related policy pins
        self.multimodal_posterior_waypoints_output: PolicyPin = None
        self.multimodal_posterior_waypoints_std_output: PolicyPin = None
        self.multimodal_prior_waypoints_output: PolicyPin = None
        self.multimodal_prior_waypoints_std_output: PolicyPin = None
        self.posterior_mu_output: PolicyPin = None
        self.posterior_tril_output: PolicyPin = None
        self.prior_tril_output: PolicyPin = None
        self.prior_mu_output: PolicyPin = None

        # only past context_encoding is used as context input for the ActionsPredictor
        submodel_configs_["actions_predictor"]["num_context_inputs"] = 1
        submodel_configs_["prior_predictor"]["config"][
            "zero_correlation"
        ] = zero_correlation
        submodel_configs_["posterior_predictor"]["config"][
            "zero_correlation"
        ] = zero_correlation

        super().__init__(
            base_model_name=base_model_name,
            submodel_configs=submodel_configs_,
            inputs=inputs,
            outputs=outputs,
            inputs_sample_times=inputs_sample_times,
            outputs_sample_times=outputs_sample_times,
            num_output_timesteps=num_output_timesteps,
            latent_sampling=latent_sampling,
            num_z_samples=num_z_samples,
            num_modes=num_modes,  # num modes per sample
            zero_correlation=zero_correlation,
            latent_z_dim=latent_z_dim,
            output_samples_handling=output_samples_handling,
            data_source=data_source,
            **kwargs,
        )

        # delegate functionality to a base model instance
        base_model_inputs = [
            input
            for input in inputs
            if input.name in SUPPORTED_BASE_MODELS[base_model_name].get_input_names()
        ]
        # outputs need to be defined manually since CVAE and base model output names differ; TODO: hacky, find an alternative
        base_model_outputs = [
            PolicyPin(
                name="multimodal_future_waypoints_output",
                key="multimodal_future_waypoints",
            )
        ]
        self.base_model = SUPPORTED_BASE_MODELS[base_model_name](
            submodel_configs=submodel_configs_,
            inputs=base_model_inputs,
            outputs=base_model_outputs,
            inputs_sample_times=inputs_sample_times,
            outputs_sample_times=outputs_sample_times,
            num_output_timesteps=num_output_timesteps,
            num_modes=num_modes,
            data_source=data_source,
        )

        # GT Encoder
        self.gt_encoder = self.prepare_gt_encoder(
            submodel_configs_[self.gt_encoder_name]
        )

        # PosteriorPredictor
        self._gt_encoding_shape = self.gt_encoder.forward(
            torch.rand(self.future_actions_features_input.shape).unsqueeze(0)[
                :, 1:, :
            ]  # omit 0-th action to have equal size of past and future actions
        ).shape[1:]

        # Prior and Posterior Predictor
        posterior_predictor_input_shape = (
            self.base_model._context_encoding_shape[0]
            + self.base_model._actions_encoding_shape[0]
            + self._gt_encoding_shape[0],
        )
        prior_predictor_input_shape = (
            self.base_model._context_encoding_shape[0]
            + self.base_model._actions_encoding_shape[0],
        )
        self.posterior_flattened_embedding_input = PolicyPin(
            name="conditional_input",
            key="flattened_features",
            shape=posterior_predictor_input_shape,
        )
        self.prior_flattened_embedding_input = PolicyPin(
            name="conditional_input",
            key="flattened_features",
            shape=prior_predictor_input_shape,
        )
        self.posterior_mu_output = FromToPinConnection.create_from_to_pin(
            self.posterior_mu_output, from_name="mu_output"
        ).from_pin
        self.posterior_tril_output = FromToPinConnection.create_from_to_pin(
            self.posterior_tril_output, from_name="tril_output"
        ).from_pin

        self.prior_mu_output = FromToPinConnection.create_from_to_pin(
            self.prior_mu_output, from_name="mu_output"
        ).from_pin
        self.prior_tril_output = FromToPinConnection.create_from_to_pin(
            self.prior_tril_output, from_name="tril_output"
        ).from_pin

        self.posterior_predictor = self.prepare_prior_posterior_predictor(
            submodel_configs_[self.posterior_predictor_name],
            inputs=[self.posterior_flattened_embedding_input],
            outputs=[self.posterior_mu_output, self.posterior_tril_output],
        )
        self.prior_predictor = self.prepare_prior_posterior_predictor(
            submodel_configs_[self.prior_predictor_name],
            inputs=[self.prior_flattened_embedding_input],
            outputs=[self.prior_mu_output, self.prior_tril_output],
        )

        # Actions predictor
        # the actions_predictor_model of the superclass is overwritten so that it additionally takes the sampled z as input
        # Therefore, we slightly abuse the model and push encoded actions AND sampled z as flattened_vector into recurrent_features_input
        self.actions_predictor_flattened_z_actions_input = PolicyPin(
            name="recurrent_features_input",
            key="flattened_actions_z_input",
            shape=(self.base_model._actions_encoding_shape[0] + latent_z_dim,),
        )
        self.actions_predictor = self.base_model.prepare_actions_predictor(
            submodel_configs_[self.base_model.actions_predictor_name],
            inputs=[
                self.base_model.actions_predictor_context_embedding_input,
                self.actions_predictor_flattened_z_actions_input,
            ],
            outputs=[
                self.base_model.actions_predictor_actions_output,
                self.base_model.actions_predictor_probabilities_output,
            ],
            num_modes=num_modes,
            data_source=data_source,
        )

    def prepare_prior_posterior_predictor(
        self,
        posterior_predictor_config: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
    ) -> Policy:
        model_cf = posterior_predictor_config["config"]
        model_cf["inputs"] = inputs
        model_cf["outputs"] = outputs
        return policy_from_policy_config(
            {
                "path_to_class": posterior_predictor_config["path_to_class"],
                "config": model_cf,
            }
        )

    def prepare_gt_encoder(self, gt_encoder_config) -> torch.nn.Module:
        model_cf = gt_encoder_config["config"]
        return class_from_path(gt_encoder_config["path_to_class"]).from_config(model_cf)

    def encode_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Gathers the necessary encodings (context, actions, gt) used in the encoder, decoder, prior components of the CVAE. Includes the ego state among the encodings for convenience. Ego state is used to decode waypoints from action trajectories at multiple later instances.
        """
        encodings: Dict[str, torch.Tensor] = {
            "context_encoding": self.base_model.call_context_encoder(inputs),
            "actions_encoding": self.base_model.actions_encoder(
                inputs[self.past_actions_features_input.key]
            ),
            "gt_encoding": self.gt_encoder(
                # omit 0-th action to have equal size of past and future actions
                inputs[self.future_actions_features_input.key][:, 1:, :]
            ),
            "ego_state": inputs[self.current_ego_state_features_input.key],
        }

        return encodings

    def posterior_forward(
        self, context_encoding, actions_encoding, gt_encoding, ego_state
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Computes the forward pass of the posterior. Given a conditional input X, which is a context encoding and an actions encoding, as well as the gt input Y, which is the ground-truth trajectory encoding, calls the encoder and decoder components. The encoder constructs the posterior distribution given the conditional input and the gt encoding, P(Z|X, Y). This distribution is sampled, and samples are then reconstructed into trajectories via the deterministic decoder (which also receives the conditional input) f(X, Z).

        Args:
            context_encoding: context encoding part of the conditional input, shape [B, CD], where CD is the context encoding dimensionality
            actions_encoding: context encoding part of the conditional input, shape [B, AD], where AD is the actions encoding dimensionality
            gt_encoding: gt trajectory encoding part of the conditional input, shape [B, GD], where GD is the gt encoding dimensionality
            ego_state: prediction-ego state at the current timestep, used to reconstruct positions from predicted actions, shape [B, 6], 6 is [x, y, yaw, v, length, width]

        Returns:
            out_posterior: dict containing the posterior statistics and posterior samples reconstructions:
                key: posterior_future_waypoints: predicted future waypoints for each latent sample, [B, N, M, T, 2], where N corresponds to the number of modes from the latent sampling dimension, M is the number of modes from the decoder (since it can output multiple trajectory modes given an input, resulting in a dual N, M multi-modality), and T is the number of timesteps.
                key: posterior_future_waypoints_std: std of the predicted future waypoints for each latent sample, [B, N, M, T, 2]
                key: mu_posterior: posterior mean, [B, D]
                key: tril_posterior: posterior tril, [B, D, D]
            z_posterior: samples drawn from the latent posterior distribution, [B, N, D]
        """
        # compute conditional posterior given gt to compare with the prior
        # compute posterior inputs
        posterior_encodings = torch.cat(
            [context_encoding, actions_encoding, gt_encoding], dim=-1
        )

        # compute posterior statistics
        z_posterior, posterior_params = self.posterior_sampling(posterior_encodings)

        # compute posterior waypoints
        posterior_future_waypoints = self.decode_waypoints_from_z_samples(
            z=z_posterior,
            context_encoding=context_encoding,
            actions_encoding=actions_encoding,
            ego_state=ego_state,
        )

        out_posterior = posterior_params
        out_posterior.update(self.posterior_make_outputs(posterior_future_waypoints))

        return out_posterior, z_posterior

    def posterior_sampling(self, posterior_encodings: torch.Tensor) -> torch.Tensor:
        """
        Sample from the posterior distribution.
        :param posterior_encodings: (Tensor) Mean of the latent Gaussian [B x D]
        :return:
            z_prior (Tensor) [B x N_components x N_samples x D]
            prior_params[self.prior_mu_output.key]: means of mixture components, (Tensor) [B x K x D]
            prior_params[self.prior_tril_output.key]: trils of mixture components, (Tensor) [B x K x D x D]
            prior_params[self.prior_component_weights_output.key] weights of mixture components, (Tensor) [B x K]
        """
        posterior_params = self.posterior_predictor.forward(
            {self.posterior_flattened_embedding_input.key: posterior_encodings}
        )
        mu_posterior = posterior_params[self.posterior_mu_output.key]
        tril_posterior = posterior_params[self.posterior_tril_output.key]
        z_posterior = self.sample_z(mu_posterior, tril_posterior, self.num_z_samples)

        return z_posterior, posterior_params

    def posterior_make_outputs(self, posterior_future_waypoints):
        (
            posterior_future_waypoints_output,
            posterior_future_waypoints_std,
            _,
        ) = self.handle_output_samples(
            multimodal_trajectories=posterior_future_waypoints
        )

        return {
            self.multimodal_posterior_waypoints_output.key: posterior_future_waypoints_output,
            self.multimodal_posterior_waypoints_std_output.key: posterior_future_waypoints_std,
        }

    def prior_forward(
        self, context_encoding, actions_encoding, ego_state
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Computes the forward pass of the prior. Given a conditional input X, which is a context encoding and an actions encoding, calls the prior and decoder components. The prior constructs the prior distribution given the conditional input, P(Z|X). This distribution is sampled, and samples are then reconstructed into trajectories via the deterministic decoder (which also receives the conditional input) f(X, Z).

        Args:
            context_encoding: context encoding part of the conditional input, shape [B, CD], where CD is the context encoding dimensionality
            actions_encoding: context encoding part of the conditional input, shape [B, AD], where AD is the actions encoding dimensionality
            ego_state: prediction-ego state at the current timestep, used to reconstruct positions from predicted actions, shape [B, 6], 6 is [x, y, yaw, v, length, width]

        Returns:
            out_prior: dict containing the prior statistics and prior samples reconstructions:
                key: prior_future_waypoints: predicted future waypoints for each latent sample, [B, N, M, T, 2], where N corresponds to the number of modes from the latent sampling dimension, M is the number of modes from the decoder (since it can output multiple trajectory modes given an input, resulting in a dual N, M multi-modality), and T is the number of timesteps.
                key: prior_future_waypoints_std: std of the predicted future waypoints for each latent sample, [B, N, M, T, 2]
                key: mu_prior: prior mean, [B, D]
                key: tril_prior: prior tril, [B, D, D]
            z_prior: samples drawn from the latent prior distribution, [B, N, D]
        """
        prior_encodings = torch.cat([context_encoding, actions_encoding], dim=-1)

        # compute prior statistics
        z_prior, prior_params = self.prior_sampling(prior_encodings)

        prior_future_waypoints = self.decode_waypoints_from_z_samples(
            z=z_prior,
            context_encoding=context_encoding,
            actions_encoding=actions_encoding,
            ego_state=ego_state,
        )

        out_prior = prior_params
        out_prior.update(self.prior_make_outputs(prior_future_waypoints))

        return out_prior, z_prior

    def prior_sampling(self, prior_encodings: torch.Tensor) -> torch.Tensor:
        """
        Sample from the (un)conditional prior distribution.
        :param prior_encodings: (Tensor) Mean of the latent Gaussian [B x D]
        :return:
            z_prior (Tensor) [B x N_components x N_samples x D]
            prior_params[self.prior_mu_output.key]: means of mixture components, (Tensor) [B x K x D]
            prior_params[self.prior_tril_output.key]: trils of mixture components, (Tensor) [B x K x D x D]
            prior_params[self.prior_component_weights_output.key] weights of mixture components, (Tensor) [B x K]
        """
        prior_params = self.prior_predictor.forward(
            {self.prior_flattened_embedding_input.key: prior_encodings}
        )
        mu_prior = prior_params[self.prior_mu_output.key]
        tril_prior = prior_params[self.prior_tril_output.key]
        z_prior = self.sample_z(mu_prior, tril_prior, self.num_z_samples)

        return z_prior, prior_params

    def prior_make_outputs(self, prior_future_waypoints):
        (
            prior_future_waypoints_,
            prior_future_waypoints_std,
            _,
        ) = self.handle_output_samples(multimodal_trajectories=prior_future_waypoints)

        return {
            self.multimodal_prior_waypoints_output.key: prior_future_waypoints_,
            self.multimodal_prior_waypoints_std_output.key: prior_future_waypoints_std,
        }

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Realizes the forward pass, computing posterior and prior distribution statistics (mean and tril), sampling the distributions, and decoding trajectory samples from both distributions.
        """
        encodings = self.encode_inputs(inputs)

        out_posterior, _ = self.posterior_forward(
            encodings["context_encoding"],
            encodings["actions_encoding"],
            encodings["gt_encoding"],
            encodings["ego_state"],
        )
        out_prior, _ = self.prior_forward(
            encodings["context_encoding"],
            encodings["actions_encoding"],
            encodings["ego_state"],
        )

        out = {}
        out.update(out_prior)
        out.update(out_posterior)

        return out

    def decode_waypoints_from_z_samples(
        self,
        z: torch.Tensor,
        context_encoding: torch.Tensor,
        actions_encoding: torch.Tensor,
        ego_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calls the CVAE decoder in order to compute trajectory outputs given latent samples z, as well as the conditional input consisting of a context encoding feature vector and a actions encoding feature vector. Additionally takes in the current prediction-ego state in order to predict future actions from which a position trajectory is decoded.

        Args:
            z: latent samples z, shape [B, N, D], where B is the batch size, N is the number of samples and D is the latent dimensionality
            context_encoding: context encoding part of the conditional input, shape [B, CD], where CD is the context encoding dimensionality
            actions_encoding: context encoding part of the conditional input, shape [B, AD], where AD is the actions encoding dimensionality

        Returns:
            future_waypoints: predicted future waypoints for each latent sample, [B, N, M, T, 2], where N now corresponds to the number of modes from the latent sampling dimension, M is the number of modes from the decoder (since it can output multiple trajectory modes given an input, resulting in a dual N, M multi-modality), and T is the number of timesteps.
        """
        # we have N samples from p_z so we need to broadcast the other tensors to [B,N,..] and reshape to [\hat{B},..] with \hat{B}=B*N
        # Therby, we can prevent looping over samples
        _, num_samples, _ = z.shape

        context_encoding = (
            context_encoding.unsqueeze(1)
            .repeat(1, num_samples, 1)
            .reshape(-1, context_encoding.shape[-1])
        )
        actions_encoding = (
            actions_encoding.unsqueeze(1)
            .repeat(1, num_samples, 1)
            .reshape(-1, actions_encoding.shape[-1])
        )
        z_batched = z.reshape(-1, z.shape[-1])

        # the action prediction model takes only two inputs (context_embedding and actions embedding)
        # since all inputs are flattened and concatenated into a single vector, we can pass a third input (sampled z)
        # by slightly abusing the model by concatenating two of the inputs before passing them to the model
        actions_z = torch.cat([actions_encoding, z_batched], dim=1)

        # predict actions
        # We have N latent samples from p_z and predict M=num_nodes_per_sample modes for every sample
        # Therefore the actions predictor takes [\hat{B},..] and returns [\hat{B},M,..] containing B*N*M different modes
        future_actions = self.actions_predictor.forward(
            {
                self.base_model.actions_predictor_context_embedding_input.key: context_encoding,
                self.actions_predictor_flattened_z_actions_input.key: actions_z,
            }
        )[self.base_model.actions_predictor_actions_output.key]

        current_state_ego_attributes = KinematicBicycleModel.get_state_ego_attributes(
            ego_state[:, 0]
        )
        # broadcast ego states attributes from [B,6] to [\hat{B},6]
        current_state_ego_attributes = (
            current_state_ego_attributes.unsqueeze(1)
            .repeat(1, num_samples, 1)
            .reshape(-1, current_state_ego_attributes.shape[-1])
        )

        # decode waypoints
        future_waypoints = self.base_model.get_multimodal_waypoints_via_kinematic_model(
            multimodal_actions=future_actions,
            current_state_ego_attributes=current_state_ego_attributes,
        )

        # Before returning we have to reshape from [\hat{B}, M, T, 2] to [B, N, M, T, 2]
        # with N: number of latent samples and M: number of modes per sample
        future_waypoints = future_waypoints.unsqueeze(1).reshape(
            -1,
            num_samples,
            self.base_model.num_modes,
            self.base_model.num_output_timesteps,
            2,
        )

        return future_waypoints

    def random_sampling(
        self, mu: torch.Tensor, tril: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param tril: (Tensor) lower triangular covariance matrix of the latent Gaussian [B x D x D]
        :param num_samples: (int) number of samples N to draw from distribution
        :return: (Tensor) [B x N x D]
        """
        # repeat to mu [B, N, D] and tril [B, N, D, D]
        mu = mu.unsqueeze(1).repeat(1, num_samples, 1)
        tril = tril.unsqueeze(1).repeat(1, num_samples, 1, 1)
        eps = torch.randn_like(mu)
        return mu + torch.matmul(tril, eps.unsqueeze(-1)).squeeze(-1)

    def no_sampling(
        self, mu: torch.Tensor, tril: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """
        Used for constant-variance-posterior models such as RAE. Instead of sampling, just uses the mean as a latent code (i.e. variance=const).
        The exact same sample is repeated num_samples times without a warning! - num_samples should be set to 1 to avoid
        unnecessary computation.
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param tril [ignored]: (Tensor) lower triangular covariance matrix of the latent Gaussian [B x D x D]
        :param num_samples: (int) number of samples N to draw from distribution
        :return: (Tensor) [B x N x D]
        """
        return mu.unsqueeze(1).repeat(1, num_samples, 1)

    def unscented_sampling(
        self, mu: torch.Tensor, tril: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """
        Used for deterministic sampling models such as UAE.
        """
        heuristic = self.latent_sampling["heuristic"]
        return sample_with_unscented_transform(mu, tril, num_samples, heuristic)

    def no_averaging(
        self, multimodal_trajectories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns sample-based multi-modal trajectories.
        Args:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
        Returns:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
            trajectories_std: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2], cotains zeros
            None: empty probabilities for compatibility
        """
        trajectories_std = torch.zeros_like(multimodal_trajectories)

        return multimodal_trajectories, trajectories_std, None

    def average_outputs(
        self, multimodal_trajectories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Averages sample-based multi-modal trajectories and returns the trajectory mean.
        Args:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
        Returns:
            trajectories_mean: shape [batch_size, 1, num_decoder_modes, num_timesteps, dim=2]
            trajectories_std: shape [batch_size, 1, num_decoder_modes, num_timesteps, dim=2]
            None: empty probabilities for compatibility
        """
        assert (
            multimodal_trajectories.shape[2] == 1
        ), "Decoder multi-modality dimension must be 1"

        if self.training:
            trajectories_mean = multimodal_trajectories.mean(dim=1, keepdim=True)
            trajectories_std = torch.std(multimodal_trajectories, dim=1, keepdim=True)
        else:
            trajectories_mean = multimodal_trajectories
            trajectories_std = torch.zeros_like(multimodal_trajectories)

        return trajectories_mean, trajectories_std, None

    def average_outputs_with_clustering(
        self, multimodal_trajectories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Clusters sample-based multi-modal trajectories and returns cluster centroids.
        Args:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
        Returns:
            cluster_centroids: shape [batch_size, num_clusters, num_decoder_modes, num_timesteps, dim=2]
            cluster_stds: shape [batch_size, num_clusters, num_decoder_modes, num_timesteps, dim=2]
            cluster_probs: shape [batch_size, num_clusters]
        """
        assert (
            multimodal_trajectories.shape[2] == 1
        ), "Decoder multi-modality dimension must be 1"

        path_to_clustering_func = self.output_samples_handling[
            "path_to_clustering_func"
        ]
        num_clusters = self.output_samples_handling["num_clusters"]

        cluster_centroids, cluster_std, cluster_probs = clustering_function(
            unclustered_data=multimodal_trajectories.squeeze(2),
            path_to_clustering_func=path_to_clustering_func,
            num_clusters=num_clusters,
        )

        # add decoder multi-modality dimension
        cluster_centroids = cluster_centroids.unsqueeze(2)
        cluster_stds = cluster_std.unsqueeze(2)

        return cluster_centroids, cluster_stds, cluster_probs


class ExPostCVAEActionSpacePredictor(CVAEActionSpacePredictor):
    """
    CVAE model with (conditional) ex-post estimation in inference. Inherits from the base CVAE model.

    (conditional) ex-post estimation entails collecting posterior or posterior+prior embeddings during training and using them to build another distribution. This distribution is sampled in inference instead of the prior distribution, as usually done in CVAEs.
    """

    def __init__(
        self,
        base_model_name: str,
        submodel_configs: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        inputs_sample_times: List[float],
        outputs_sample_times: List[float],
        latent_sampling: dict,
        zero_correlation: bool,
        num_z_samples: int,
        inference_method: str,
        num_mixture_components: int,
        num_modes: int = 1,  # num modes per sample
        latent_z_dim: int = 64,
        data_source: str = "default",
        num_output_timesteps: Optional[int] = None,
        output_samples_handling: dict = {"method": "no_averaging"},
    ) -> None:

        SUPPORTED_INFERENCE_METHODS = {
            "expost": self.uncond_expost_sampling,
            "cond_expost": self.cond_expost_sampling,
        }

        assert (
            inference_method in SUPPORTED_INFERENCE_METHODS.keys()
        ), f"latent_sampling has to be in {super().SUPPORTED_INFERENCE_METHODS.keys()} but {inference_method} was passed"

        self.inference_method = inference_method
        self.sample_z_in_inference = SUPPORTED_INFERENCE_METHODS[self.inference_method]
        self.num_mixture_components = num_mixture_components

        # store trained mixture in a folder to load in evaluation
        self.stored_mixture_path = "./stored_expost_mixture/"

        # ex-post model related policy pins
        self.multimodal_mixture_waypoints_output: PolicyPin = None
        self.multimodal_mixture_waypoints_std_output: PolicyPin = None
        self.mixture_weights_output: PolicyPin = None

        super().__init__(
            base_model_name=base_model_name,
            submodel_configs=submodel_configs,
            inputs=inputs,
            outputs=outputs,
            inputs_sample_times=inputs_sample_times,
            outputs_sample_times=outputs_sample_times,
            num_output_timesteps=num_output_timesteps,
            latent_sampling=latent_sampling,
            num_z_samples=num_z_samples,
            num_modes=num_modes,  # num modes per sample
            zero_correlation=zero_correlation,
            latent_z_dim=latent_z_dim,
            output_samples_handling=output_samples_handling,
            data_source=data_source,
            num_mixture_components=num_mixture_components,
            inference_method=inference_method,
        )

        self.is_mixture_fit = False  # to track the mixture fitting logic
        self.mixture_latents = (
            []
        )  # to store the training {post, prior} latents for ex-post density estimation

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        returns:
            future_waypoints: (Tensor), B x N x M x T x 2 with N being number of latent samples and M being number of modes per sample
            latent mu: (Tensor) B x D with D being latent variables dimension estimated latent distribution
            latent tril: (Tensor) B x D x D estimated lower triangular covariance matrix
            z: (Tensor) B x N x D samples drawn from the latent distribution
        """
        encodings = self.encode_inputs(inputs)

        out_posterior, z_posterior = self.posterior_forward(
            encodings["context_encoding"],
            encodings["actions_encoding"],
            encodings["gt_encoding"],
            encodings["ego_state"],
        )
        out_prior, z_prior = self.prior_forward(
            encodings["context_encoding"],
            encodings["actions_encoding"],
            encodings["ego_state"],
        )

        if self.training:
            # accumulate latents in training for expost estimation
            self.expost_collection(z_posterior, z_prior)

        else:
            # inference trajectory is sampled from the mixture, no prior distribution
            (
                z_mixture_samples,  # [batch_size, num_mixture_samples, feat_dim]
                z_mixture_sigmas,  # [batch_size, num_components, num_sigmas, feat_dim]
                z_mixture_weights,  # [batch_size, num_components]
            ) = self.sample_z_in_inference(z_prior)
            batch_size = z_mixture_samples.shape[0]
            feat_dim = z_mixture_samples.shape[-1]
            num_sigmas = z_mixture_sigmas.shape[2]

            mixture_samples_future_waypoints = self.decode_waypoints_from_z_samples(
                z=z_mixture_samples,
                context_encoding=encodings["context_encoding"],
                actions_encoding=encodings["actions_encoding"],
                ego_state=inputs[self.current_ego_state_features_input.key],
            )
            (
                mixture_samples_future_waypoints,
                mixture_samples_future_waypoints_std,
                mixture_samples_future_waypoints_probs,
            ) = self.handle_output_samples(
                multimodal_trajectories=mixture_samples_future_waypoints
            )
            # naming out_prior and prior_waypoints is preserved for compatibility
            out_prior.update(
                {
                    self.multimodal_prior_waypoints_output.key: mixture_samples_future_waypoints,
                    self.multimodal_prior_waypoints_std_output.key: mixture_samples_future_waypoints_std,
                }
            )

            mixture_sigmas_future_waypoints = self.decode_waypoints_from_z_samples(
                z=z_mixture_sigmas.reshape(
                    batch_size, -1, feat_dim
                ),  # [batch_size, num_components * num_sigmas, feat_dim]
                context_encoding=encodings["context_encoding"],
                actions_encoding=encodings["actions_encoding"],
                ego_state=inputs[self.current_ego_state_features_input.key],
            )  # [batch_size, num_components * num_sigmas, num_decoder_modes, num_waypoints, dim]
            mixture_sigmas_future_waypoints = mixture_sigmas_future_waypoints.reshape(
                batch_size,
                self.num_mixture_components,
                num_sigmas,
                *mixture_sigmas_future_waypoints.shape[-3:],
            )
            mixture_sigmas_future_waypoints_mean = torch.mean(
                mixture_sigmas_future_waypoints, dim=2
            )
            mixture_sigmas_future_waypoints_std = torch.std(
                mixture_sigmas_future_waypoints, dim=2
            )
            out_prior.update(
                {
                    self.multimodal_mixture_waypoints_output.key: mixture_sigmas_future_waypoints_mean,
                    self.multimodal_mixture_waypoints_std_output.key: mixture_sigmas_future_waypoints_std,
                    self.mixture_weights_output.key: z_mixture_weights,
                }
            )

        # collect prior and posterior statistics, waypoints, and decoder weights
        out = {}
        out.update(out_prior)
        out.update(out_posterior)

        return out

    @torch.no_grad()
    def expost_collection(
        self, z_posterior: torch.Tensor, z_prior: torch.Tensor = None
    ):
        """
        Collects posterior and prior latents for expost mixture fitting. At the start of training in an epoch, clears latents from previous epoch.

        Args:
            z_posterior, z_prior: latent feature vectors of posterior and prior [B, S, D]
        """
        if self.is_mixture_fit:
            # at the start of epoch training stage, clear fitted mixture from previous epoch
            self.mixture_latents.clear()
            self.is_mixture_fit = False

        # during epoch training stage, collect latents to fit mixture in validation
        if self.inference_method == "expost":
            # add posterior latents
            self.mixture_latents.append(z_posterior.detach().cpu())
        elif self.inference_method == "cond_expost":
            # add concatenated posterior and prior latents

            batch_size, num_sigmas, feat_dim = z_posterior.shape

            # concatenate sigmas such that distances are minimized
            num_perms = 1000
            # find num_perms permutations of sigma indices
            index_perms = torch.stack(
                [
                    torch.arange(num_sigmas)[torch.randperm(num_sigmas)]
                    for _ in range(num_perms)
                ]
            ).to(z_posterior.device)
            # prepare z_prior for permuted selection, add new dimension of permutations at 1
            expanded_z_prior = z_prior.unsqueeze(1).expand(
                batch_size, num_perms, num_sigmas, feat_dim
            )
            # select sigma point permutations according to index_perms
            expanded_z_prior = expanded_z_prior.gather(
                dim=2,
                index=index_perms.unsqueeze(-1).expand(
                    batch_size, num_perms, num_sigmas, feat_dim
                ),
            )

            # find lowest differences between expanded posterior and prior sigma point permutations
            expanded_z_posterior = z_posterior.unsqueeze(1).expand(
                batch_size, num_perms, num_sigmas, feat_dim
            )
            perm_diffs = torch.linalg.vector_norm(
                (expanded_z_posterior - expanded_z_prior), dim=-1
            ).sum(
                dim=-1
            )  # [batch_size, num_perms]
            _, min_diffs = torch.min(perm_diffs, dim=1)  # [batch_size]

            # select prior sigma point permutation with lowest difference to posterior
            expanded_z_prior = torch.take_along_dim(
                expanded_z_prior, dim=1, indices=min_diffs[:, None, None, None]
            ).squeeze(
                1
            )  # [batch_size, num_sigmas, feat_dim]

            self.mixture_latents.append(
                torch.cat((z_posterior, expanded_z_prior), dim=2).detach().cpu()
            )

    @torch.no_grad()
    def expost_fit_mixture(self) -> GaussianMixture:
        """
        Fits a GMM using latent features. Suitable for both uncond and cond expost stages.

        Args:
            fitting_data [num_data_points, dim]
        """
        # prepare fitting data
        fitting_data = torch.cat(
            self.mixture_latents
        )  # [B * num_samples x D] join batch_size and sample dimensions
        fitting_data = fitting_data.reshape(-1, fitting_data.shape[-1])

        self.expost_mixture_params = {
            "num_components": self.num_mixture_components,
            "covariance_type": "full",
            "trainer_params": {
                "accelerator": "gpu",
                "devices": 1,
                "enable_progress_bar": False,
            },
            "batch_size": 32768,
        }

        self.expost_mixture = GaussianMixture(**self.expost_mixture_params)
        self.expost_mixture.fit(fitting_data)

        self.is_mixture_fit = True

        return self.expost_mixture

    @torch.no_grad()
    def uncond_expost_sampling(self, z_prior: torch.Tensor):
        """
        Sample from the fitted unconditional GMM distribution.

        Args:
            z_prior [B, encoding_dim], prior encodings are not used in the unconditional case
        Returns:
            mixture_samples [B, S, D], S latent feature samples from the mixture for each batch element B
        """
        # fit the mixture for the first time in the validation/testing stage
        if not self.is_mixture_fit:
            self.expost_fit_mixture()

        # sample from the fit posterior
        batch_size = z_prior.shape[0]
        mixture_samples, _ = self.expost_mixture.sample(batch_size * self.num_z_samples)
        mixture_samples = torch.tensor(
            mixture_samples, dtype=z_prior.dtype, device=z_prior.device
        ).view(batch_size, self.num_z_samples, -1)

        return mixture_samples

    @torch.no_grad()
    def cond_expost_sampling(self, z_prior_conditioning: torch.Tensor):
        """
        Sample from the fitted conditional GMM distribution.

        Args:
            z_prior [B, S, encoding_dim]
        Returns:
            mixture_samples [B, S, D], S latent feature samples from the mixture for each batch element B
        """
        # fit the mixture for the first time in the validation/testing stage
        if not self.is_mixture_fit:
            self.expost_fit_mixture()
            print("Computing conditional Gaussian Mixture...")

        (
            batch_size,
            num_cuts,
            feat_dim,
        ) = (
            z_prior_conditioning.shape
        )  # num cuts corresponds to num of sigmas taken from the prior
        z_prior_conditioning = z_prior_conditioning.reshape(
            batch_size * num_cuts, feat_dim
        )
        device = z_prior_conditioning.device

        # instantiate conditional mixture, repeat tensors batch_size * num_cuts times
        means = self.expost_mixture.model_.means.clone().detach().to(device)
        means = means.unsqueeze(0).repeat(batch_size * num_cuts, 1, 1)
        covs = self.expost_mixture.model_.covariances.clone().detach().to(device)
        covs = covs.unsqueeze(0).repeat(batch_size * num_cuts, 1, 1, 1)
        weights = self.expost_mixture.model_.component_probs.clone().detach().to(device)
        weights = weights.unsqueeze(0).repeat(batch_size * num_cuts, 1)
        # TODO: see if possible to achieve the same result without repeating joint mixture. maybe using expand()

        # construct conditional mixture using all the cuts
        gmm_cond = ConditionalGaussianMixtureModel(
            means=means, covariances=covs, weights=weights, cond=z_prior_conditioning
        )
        # find cuts with smallest neg. log prob
        gmm_cond_logprob = gmm_cond.mixture_cond_logprob.reshape(batch_size, num_cuts)
        _, indices = torch.min(gmm_cond_logprob, dim=1)

        # construct cond. mixtures using only best cuts in order to sample from them
        best_cut_mix_dist_weights = gmm_cond.cond_mixture_dist.probs.reshape(
            batch_size, num_cuts, self.num_mixture_components
        )  # [batch_size, num_cuts, num_components]
        best_cut_mix_dist_weights = torch.take_along_dim(
            best_cut_mix_dist_weights, dim=1, indices=indices[:, None, None]
        ).squeeze(1)
        (
            best_cut_comp_dist_means,
            best_cut_comp_dist_tril,
        ) = gmm_cond.cond_component_dist.mean.reshape(
            batch_size, num_cuts, self.num_mixture_components, feat_dim
        ), gmm_cond.cond_component_dist.scale_tril.reshape(
            batch_size, num_cuts, self.num_mixture_components, feat_dim, feat_dim
        )
        best_cut_comp_dist_means, best_cut_comp_dist_tril = torch.take_along_dim(
            best_cut_comp_dist_means, dim=1, indices=indices[:, None, None, None]
        ).squeeze(1), torch.take_along_dim(
            best_cut_comp_dist_tril, dim=1, indices=indices[:, None, None, None, None]
        ).squeeze(
            1
        )
        gmm_cond_best_cut = MixtureSameFamily(
            mixture_distribution=Categorical(
                probs=best_cut_mix_dist_weights, validate_args=False
            ),
            component_distribution=MultivariateNormal(
                loc=best_cut_comp_dist_means,
                scale_tril=best_cut_comp_dist_tril,
                validate_args=False,
            ),
        )

        # take samples from the best cut mixture for each batch element
        num_mixture_samples = 65
        best_cut_samples = gmm_cond_best_cut.sample(
            [num_mixture_samples]
        )  # [num_mixture_samples, batch_size, feat_dim]
        best_cut_samples = best_cut_samples.swapaxes(
            0, 1
        )  # [batch_size, num_mixture_samples, feat_dim]

        # best cut sigmas for every component
        best_cut_sigmas = compute_sigma_points(
            best_cut_comp_dist_means.reshape(-1, feat_dim),
            best_cut_comp_dist_tril.reshape(-1, feat_dim, feat_dim),
        )  # [batch_size * num_components, num_sigmas, feat_dim]
        best_cut_sigmas = best_cut_sigmas.reshape(
            batch_size, self.num_mixture_components, 2 * feat_dim + 1, feat_dim
        )

        return best_cut_samples, best_cut_sigmas, best_cut_mix_dist_weights

    def get_extra_state(self) -> Any:
        if (
            self.inference_method == "expost" or self.inference_method == "cond_expost"
        ) and self.is_mixture_fit:
            saved_mixture = {
                "params": self.expost_mixture.get_params(),
                "weights": self.expost_mixture.model_.component_probs,
                "means": self.expost_mixture.model_.means,
                "covariances": self.expost_mixture.model_.covariances,
            }
            self.expost_mixture.save(self.stored_mixture_path)
            print(f"Saving fit mixture to: {self.stored_mixture_path}")
            return saved_mixture
        else:
            pass

    def set_extra_state(self, state: Any):
        if self.inference_method == "expost" or self.inference_method == "cond_expost":
            self.expost_mixture = GaussianMixture.load(self.stored_mixture_path)
            self.is_mixture_fit = True


class CVAECovarActionSpacePredictor(CVAEActionSpacePredictor):
    def __init__(
        self,
        base_model_name: str,
        submodel_configs: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        inputs_sample_times: List[float],
        outputs_sample_times: List[float],
        latent_sampling: dict,
        zero_correlation: bool,
        num_z_samples: int,
        num_modes: int = 1,  # num modes per sample
        latent_z_dim: int = 64,
        data_source: str = "default",
        num_output_timesteps: Optional[int] = None,
        output_samples_handling: dict = {"method": "no_averaging"},
    ) -> None:

        self.multimodal_posterior_waypoints_covar_output: PolicyPin = None
        self.multimodal_prior_waypoints_covar_output: PolicyPin = None

        super().__init__(
            base_model_name,
            submodel_configs,
            inputs,
            outputs,
            inputs_sample_times,
            outputs_sample_times,
            latent_sampling,
            zero_correlation,
            num_z_samples,
            num_modes,  # num modes per sample
            latent_z_dim,
            data_source,
            num_output_timesteps,
            output_samples_handling,
        )

    def no_averaging(
        self, multimodal_trajectories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns sample-based multi-modal trajectories.

        Args:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
        Returns:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
            trajectories_covar: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2, dim=2], contains zeros
        """
        covar_shape = multimodal_trajectories.shape + (
            multimodal_trajectories.shape[-1],
        )
        trajectories_covar = torch.zeros(
            covar_shape,
            dtype=multimodal_trajectories.dtype,
            device=multimodal_trajectories.device,
        )

        return multimodal_trajectories, trajectories_covar

    def average_outputs(
        self, multimodal_trajectories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Averages sample-based multi-modal trajectories and returns the trajectory mean.

        Args:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
        Returns:
            trajectories_mean: shape [batch_size, 1, num_decoder_modes, num_timesteps, dim=2]
            trajectories_covar: shape [batch_size, 1, num_decoder_modes, num_timesteps, dim=2, dim=2]
        """
        assert (
            multimodal_trajectories.shape[2] == 1
        ), "Decoder multi-modality dimension must be 1"

        trajectories_reordered = multimodal_trajectories.permute(0, 2, 1, 3, 4)
        trajectories_mean, trajectories_covar = _find_covariance(trajectories_reordered)

        return trajectories_mean[:, None], trajectories_covar[:, None]

    def average_outputs_with_clustering(
        self, multimodal_trajectories: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Clusters sample-based multi-modal trajectories and returns cluster centroids.

        Args:
            multimodal_trajectories: shape [batch_size, num_samples, num_decoder_modes, num_timesteps, dim=2]
        Returns:
            cluster_centroids: shape [batch_size, num_clusters, num_decoder_modes, num_timesteps, dim=2]
            cluster_stds: shape [batch_size, num_clusters, num_decoder_modes, num_timesteps, dim=2]
        """
        raise NotImplementedError(
            "average_outputs_with_clustering not implemented with covar"
        )

    def posterior_make_outputs(self, posterior_future_waypoints):
        handle_outputs_methods_covar = {
            "no_averaging": self.no_averaging,
            "averaging": self.average_outputs,
        }
        assert (
            self.output_samples_handling["method"] in handle_outputs_methods_covar
        ), f"Unknown sample handling method: {self.output_samples_handling['method']}"

        (
            posterior_future_waypoints_,
            posterior_future_waypoints_covar,
        ) = handle_outputs_methods_covar[self.output_samples_handling["method"]](
            multimodal_trajectories=posterior_future_waypoints
        )

        return {
            self.multimodal_posterior_waypoints_output.key: posterior_future_waypoints_,
            self.multimodal_posterior_waypoints_covar_output.key: posterior_future_waypoints_covar,
        }

    def prior_make_outputs(self, prior_future_waypoints):
        handle_outputs_methods_covar = {
            "no_averaging": self.no_averaging,
            "averaging": self.average_outputs,
        }
        if self.output_samples_handling["method"] not in handle_outputs_methods_covar:
            raise RuntimeError(
                f"Unknown sample handling method: {self.output_samples_handling['method']}"
            )

        (
            prior_future_waypoints_,
            prior_future_waypoints_covar,
        ) = handle_outputs_methods_covar[self.output_samples_handling["method"]](
            multimodal_trajectories=prior_future_waypoints
        )

        return {
            self.multimodal_prior_waypoints_output.key: prior_future_waypoints_,
            self.multimodal_prior_waypoints_covar_output.key: prior_future_waypoints_covar,
        }


class GMMPriorCVAEActionSpacePredictor(CVAECovarActionSpacePredictor):
    def __init__(
        self,
        base_model_name: str,
        submodel_configs: Dict,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        inputs_sample_times: List[float],
        outputs_sample_times: List[float],
        latent_sampling: dict,
        zero_correlation: bool,
        num_z_samples: int,
        num_modes: int = 1,  # num modes per sample
        latent_z_dim: int = 64,
        data_source: str = "default",
        num_output_timesteps: Optional[int] = None,
        output_samples_handling: dict = {"method": "no_averaging"},
    ) -> None:

        self.prior_component_weights_output = PolicyPin(
            name="prior_components_weights_output",
            key="prior_components_weights",
        )
        self.posterior_component_weights_output = PolicyPin(
            name="posterior_components_weights_output",
            key="posterior_components_weights",
        )

        super().__init__(
            base_model_name=base_model_name,
            submodel_configs=submodel_configs,
            inputs=inputs,
            outputs=outputs,
            inputs_sample_times=inputs_sample_times,
            outputs_sample_times=outputs_sample_times,
            latent_sampling=latent_sampling,
            zero_correlation=zero_correlation,
            num_modes=num_modes,
            num_z_samples=num_z_samples,
            latent_z_dim=latent_z_dim,
            data_source=data_source,
            num_output_timesteps=num_output_timesteps,
            output_samples_handling=output_samples_handling,
        )

        self.prior_predictor.component_weights_output = (
            FromToPinConnection.create_from_from_pin(
                self.prior_component_weights_output, "components_weights_output"
            ).to_pin
        )
        self.posterior_predictor.component_weights_output = (
            FromToPinConnection.create_from_from_pin(
                self.posterior_component_weights_output, "components_weights_output"
            ).to_pin
        )

    def prior_sampling(
        self, prior_encodings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample from the (un)conditional prior distribution.
        :param prior_encodings: (Tensor) Mean of the latent Gaussian [B x D]
        :return:
            z_prior (Tensor) [B x N_components x N_samples x D]
            prior_params[self.prior_mu_output.key]: means of mixture components, (Tensor) [B x K x D]
            prior_params[self.prior_tril_output.key]: trils of mixture components, (Tensor) [B x K x D x D]
            prior_params[self.prior_component_weights_output.key] weights of mixture components, (Tensor) [B x K]
        """
        prior_params = self.prior_predictor.forward(
            {self.prior_flattened_embedding_input.key: prior_encodings}
        )

        mu_prior = prior_params[self.prior_mu_output.key]
        tril_prior = prior_params[self.prior_tril_output.key]

        if self.latent_sampling["method"] == "random":
            z_prior = self.random_sampling_gmm(
                mu=mu_prior,
                tril=tril_prior,
                num_samples=self.num_z_samples,
            )
        elif self.latent_sampling["method"] == "unscented":
            z_prior = self.unscented_sampling_gmm(
                mu=mu_prior,
                tril=tril_prior,
                num_samples=self.num_z_samples,
            )
        else:
            raise NotImplementedError
        return z_prior, prior_params

    def posterior_sampling(
        self, posterior_encodings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample from the posterior distribution.
        :param posterior_encodings: (Tensor) Mean of the latent Gaussian [B x D]
        :return:
            z_prior (Tensor) [B x N_components x N_samples x D]
            prior_params[self.prior_mu_output.key]: means of mixture components, (Tensor) [B x K x D]
            prior_params[self.prior_tril_output.key]: trils of mixture components, (Tensor) [B x K x D x D]
            prior_params[self.prior_component_weights_output.key] weights of mixture components, (Tensor) [B x K]
        """
        posterior_params = self.posterior_predictor.forward(
            {self.posterior_flattened_embedding_input.key: posterior_encodings}
        )

        mu_posterior = posterior_params[self.posterior_mu_output.key]
        tril_posterior = posterior_params[self.posterior_tril_output.key]

        if self.latent_sampling["method"] == "random":
            z_posterior = self.random_sampling_gmm(
                mu=mu_posterior,
                tril=tril_posterior,
                num_samples=self.num_z_samples,
            )
        elif self.latent_sampling["method"] == "unscented":
            z_posterior = self.unscented_sampling_gmm(
                mu=mu_posterior,
                tril=tril_posterior,
                num_samples=self.num_z_samples,
            )
        else:
            raise NotImplementedError
        return z_posterior, posterior_params

    def random_sampling_gmm(
        self,
        mu: torch.Tensor,
        tril: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Take a number of samples from each component. Return with components and samples combined in one sample
        dimension

        :param mu: Mean values of distribution, shape (B, N_components, Dim)
        :param tril: Triangular matrix of distribution, shape (B, N_components, Dim, Dim)
        :param num_samples: Number of samples to take.
        :return:
                z_prior (Tensor), shape: [B, N_components x N_samples, D].
        """
        components_distributions = MultivariateNormal(
            loc=mu,
            scale_tril=tril,
        )
        z = components_distributions.sample([num_samples])
        z = z.permute(1, 2, 0, 3)
        z = torch.flatten(z, start_dim=1, end_dim=2)
        return z

    def unscented_sampling_gmm(
        self,
        mu: torch.Tensor,
        tril: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """
        Take a number of samples from each component, using unscented sampling. Return with components and samples
        combined in one sample dimension

        :param mu: Mean values of distribution, shape (B, N_components, Dim)
        :param tril: Triangular matrix of distribution, shape (B, N_components, Dim, Dim)
        :param num_samples: Number of samples to take.
        :return:
                z (Tensor) (B, N_components x N_samples, D)
        """
        heuristic = self.latent_sampling["heuristic"]
        n_components = mu.shape[1]
        per_component_samples = []
        for c in range(n_components):
            per_component_samples.append(
                sample_with_unscented_transform(
                    mu[:, c, :], tril[:, c, :, :], num_samples, heuristic
                )
            )
        z = torch.stack(per_component_samples, dim=1)
        # we sample n_samples from each component and
        # squeeze them all in the num_samples dimension
        z = torch.flatten(z, start_dim=1, end_dim=2)

        return z

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # sample prior with extra component dim
        # flatten into num_samples

        # forward base model
        out = super().forward(inputs)
        # unsqueeze the samples to (B, N_component, N_samples, D)
        prior_trajectories: torch.Tensor = out[
            "multimodal_vae_prior_future_waypoints"
        ].unflatten(dim=1, sizes=[-1, self.num_z_samples])
        posterior_trajectories: torch.Tensor = out[
            "multimodal_vae_posterior_future_waypoints"
        ].unflatten(dim=1, sizes=[-1, self.num_z_samples])

        # remove extra dimension (representing clusters/samples or something like that?)
        assert prior_trajectories.shape[3] == 1
        prior_trajectories_ = prior_trajectories[:, :, :, 0, :, :]

        # return prior waypoints and covar based on mean/covar of each mode
        prior_mode_centers, prior_mode_covar = _find_covariance(prior_trajectories_)
        out["multimodal_vae_prior_future_waypoints"] = prior_mode_centers[:, :, None]
        out["multimodal_vae_prior_future_waypoints_covar"] = prior_mode_covar[
            :, :, None
        ]

        # also find covariance representation of posterior positions
        # posterior_trajectories_ is shape (batch, modes=1, samples, time, dims=2)
        posterior_trajectories_ = posterior_trajectories[:, :, :, 0, :, :]
        posterior_mode_centers, posterior_mode_covar = _find_covariance(
            posterior_trajectories_
        )
        out["multimodal_vae_posterior_future_waypoints"] = posterior_mode_centers[
            :, :, None
        ]
        out["multimodal_vae_posterior_future_waypoints_covar"] = posterior_mode_covar[
            :, :, None
        ]

        return out


def _find_covariance(
    trajectories: torch.Tensor,
    mode_centers: Optional[torch.Tensor] = None,
    min_std: Optional[float] = 0.1,
):
    """
    Find covariance matrix for given trajectories, by finding the covariance of the set of samples for each component
    for each timestep

    :param trajectories: Trajectory positions defined over modes, shape [batch_size, modes, samples, timesteps, dim=2]
    :param mode_centers: Mode mean trajectories, shape [batch_size, modes, timesteps, dim=2]. This is calculated if
        not defined.
    :param min_std: Minimum standard deviation for each component.  If provided, the covariances calculated from the
        samples are modified so any that have principle component standard deviations below the threshold are replaced
        with a constant size distribution.
    :returns: Mode centers (calculated or passed-through), and Mode covariances, shape (batch_size, modes, timesteps,
        dims=2, dims=2)
    """
    if mode_centers is None:
        # calculate centroid positions for each timestep
        mode_centers = trajectories.mean(axis=2)

    position_diffs = trajectories - mode_centers[:, :, None, :, :]
    # position_diffs_cross is shape (batch, modes, num_samples, timesteps, dims=2, dims=2)
    position_diffs_cross = (
        position_diffs[:, :, :, :, :, None] * position_diffs[:, :, :, :, None, :]
    )
    # mode_covar is shape (batch, modes, timesteps, dims=2, dims=2)
    mode_covar = position_diffs_cross.mean(axis=2)

    if min_std is not None:
        # check for singularity in matrix, and replace any singular matrices with a fixed covar matrix
        # with constant std
        covar_det = torch.det(mode_covar)
        const_covar = min_std * torch.eye(
            mode_covar.shape[-1], dtype=mode_covar.dtype, device=mode_covar.device
        )
        singular_elements = torch.eq(covar_det, 0.0)
        mode_covar = torch.where(
            singular_elements[:, :, :, None, None],
            const_covar[None, None, None, :, :],
            mode_covar,
        )

        # ensure stddev is greater than the minimum
        eigvals, eigvecs = torch.linalg.eig(mode_covar)
        stddevs = torch.sqrt(torch.real(eigvals))
        small_dists = torch.any(torch.lt(stddevs, min_std), dim=-1)
        mode_covar = torch.where(
            small_dists[:, :, :, None, None],
            const_covar[None, None, None, :, :],
            mode_covar,
        )

    return mode_centers, mode_covar
