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

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union

import torch

from features_and_labels.generators_base import (
    GeneratorOutputInfo,
    make_generator_output_info,
)
from lib.utils import class_from_path, get_path_to_class


@dataclass
class PolicyPin(dict):
    """
    A connection that defines an input to or an output from a policy
    by mapping between the key for incoming/outgoing data in the input/output dict
    and an internally used member (the name) for accessing the data.
    """

    name: str
    key: str
    info: GeneratorOutputInfo

    def __init__(self, name: str, key: str, **kwargs):
        self.name = name
        self.key = key
        self.info = make_generator_output_info(**kwargs)

    @property
    def shape(self):
        return self.info.shape

    @shape.setter
    def shape(self, value):
        self.info.shape = value

    def to_config(self) -> dict:
        return asdict(self)


@dataclass
class FromToPinConnection:
    """
    Connects two policy pins
    """

    from_name: str
    to_name: str
    key: str
    info: GeneratorOutputInfo

    def __init__(self, from_name: str, to_name: str, key: str, **kwargs):
        self.from_name = from_name
        self.to_name = to_name
        self.key = key
        self.info = make_generator_output_info(**kwargs)

    @property
    def shape(self):
        return self.info.shape

    @shape.setter
    def shape(self, value):
        self.info.shape = value

    @property
    def from_pin(self) -> PolicyPin:
        return PolicyPin(name=self.from_name, key=self.key, info=self.info)

    @property
    def to_pin(self) -> PolicyPin:
        return PolicyPin(name=self.to_name, key=self.key, info=self.info)

    @classmethod
    def create_from_from_pin(cls, from_pin: PolicyPin, to_name: str):
        return cls(
            from_name=from_pin.name,
            to_name=to_name,
            key=from_pin.key,
            info=from_pin.info,
        )

    @classmethod
    def create_from_to_pin(cls, to_pin: PolicyPin, from_name: str):
        return cls(
            from_name=from_name,
            to_name=to_pin.name,
            key=to_pin.key,
            info=to_pin.info,
        )


@dataclass
class PolicyConfig:
    """
    Dataclass for specifying the policy config
    """

    inputs: List[PolicyPin]
    outputs: List[PolicyPin]
    additional_parameters: dict = field(default_factory=dict)

    def __init__(
        self,
        inputs,
        outputs,
        additional_parameters=dict(),
        **kwargs,
    ):
        """
        additional_parameters can be specified in two ways:
        - kwargs: to allow natural policy config dicts
        - additional_parameters: to allow conversion Policy_config <-> dict
        """
        self.inputs = [
            i if isinstance(i, PolicyPin) else PolicyPin(**i) for i in inputs
        ]
        self.outputs = [
            o if isinstance(o, PolicyPin) else PolicyPin(**o) for o in outputs
        ]
        self.additional_parameters = {**additional_parameters, **kwargs}


@dataclass
class PolicyClassConfigBase:
    path_to_class: str
    config: Any
    is_expert: bool


@dataclass
class PolicyClassConfig(PolicyClassConfigBase):
    """
    Dataclass for specifying the policy class, config, and state dict
    """

    config: PolicyConfig
    path_to_state_dict: str = None
    is_expert: bool = False

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = PolicyConfig(**self.config)


class Policy(torch.nn.Module):
    """
    Base class for NN policies.
    Usage: output_features = Policy.forward(input_features)
    """

    def __init__(
        self,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        **kwargs,
    ) -> None:
        # can be used to reconstruct the policy
        self.config = PolicyConfig(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # torch.nn.Module.__init__: parameter and buffer registration
        super().__init__()

        self._init_inputs_outputs(inputs, outputs)

    def _init_policy_pin(self, policy_pin: PolicyPin) -> None:
        """
        This function can be used to initialize the external inputs to and the outputs from the policy.
        The convention is as follows
        1. You define the pins for the inputs and outputs before super().__init__, e.g.
                self.binary_grid_input: PolicyPin = None
                self.waypoints_output: PolicyPin = None
        2. You call super().__init__
        3. You create connections for your internal components, e.g. (here we already have the to_pin defined)
            my_input_pin_to_my_head_connection = FromToPinConnection.create_from_from_pin(
                from_pin=my_pin_to_policy, to_name="internal_variable_name"
            )
            self.internal_head = MyHead(
                inputs=[
                    my_input_pin_to_my_head_connection.to_pin
                ],
                outputs=[
                    # more code here
                ]
            )
        4. You use your defined keys to work with features and outputs, e.g.
                def forward(self, x: Dict[str,  torch.Tensor]) -> Dict[str, torch.Tensor]:
                    my_input_tensor = x[my_input_pin_to_my_head_connection.key]
                    ....
                    return {my_head_to_my_output_connection.key: my_output_tensor}
        """
        assert hasattr(
            self, policy_pin.name
        ), f"Mapping {policy_pin.name} is not defined"
        self.__setattr__(
            policy_pin.name,
            policy_pin,
        )

    def _init_inputs_outputs(
        self, inputs: List[PolicyPin], outputs: List[PolicyPin]
    ) -> None:
        """
        Convenience function to initialize all inputs and outputs
        """
        self.input_keys = []
        self.output_keys = []
        for input_pin in inputs:
            self._init_policy_pin(input_pin)
            self.input_keys.append(input_pin.key)
        for output_pin in outputs:
            self._init_policy_pin(output_pin)
            self.output_keys.append(output_pin.key)

    def save_state_dict(self, path_to_state_dict_file: str) -> None:
        torch.save(self.state_dict(), path_to_state_dict_file)

    def load_state_dict(self, path_to_state_dict_file: str) -> None:
        if os.path.exists(path_to_state_dict_file):
            if torch.cuda.is_available():
                state_dict = torch.load(path_to_state_dict_file)
            else:
                state_dict = torch.load(path_to_state_dict_file, map_location="cpu")
            super(Policy, self).load_state_dict(state_dict)
        else:
            raise ValueError(
                f"Given path to policy state-dict does not exist ({path_to_state_dict_file})"
            )
        self.eval()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: PolicyConfig, path_to_state_dict: str = None):
        policy = cls(
            inputs=config.inputs,
            outputs=config.outputs,
            **(config.additional_parameters),
        )
        if path_to_state_dict is not None:
            policy.load_state_dict(path_to_state_dict)
        return policy

    def to_config(self) -> PolicyClassConfig:
        return PolicyClassConfig(
            path_to_class=get_path_to_class(self.__class__), config=self.config
        )


##################################################################################################################################
# CONFIG HANDLING
##################################################################################################################################


def policy_from_policy_config(policy_config: Union[PolicyClassConfig, dict]) -> Policy:
    """
    Instantiates a policy from configuration. If the policy is a NN policy and the path to stored weights is specified, the
    weights are loaded into the policy.
    """
    if isinstance(policy_config, dict):
        policy_config = PolicyClassConfig(
            config=policy_config["config"],
            path_to_class=policy_config["path_to_class"],
            path_to_state_dict=policy_config["path_to_state_dict"]
            if "path_to_state_dict" in policy_config.keys()
            else None,
            is_expert=policy_config["is_expert"]
            if "is_expert" in policy_config.keys()
            else False,
        )

    policy_cls = class_from_path(policy_config.path_to_class)
    return policy_cls.from_config(
        policy_config.config, policy_config.path_to_state_dict
    )
