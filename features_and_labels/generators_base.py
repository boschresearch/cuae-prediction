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

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from gym import spaces

from features_and_labels.mission import Mission
from lib.frame import Frame
from lib.utils import ensure_init_type, get_path_to_class, instantiate_from_config


def punch_past_and_future_frames(
    frames_list: List[Frame], idx_reference_frame: int, len_tail: int, len_head: int
) -> Tuple[List[Frame], List[Frame]]:
    """
    Punches <past_frames> and <future_frames> for the frame at <idx_reference_frame>
    """
    past_frames = frames_list[idx_reference_frame - len_tail : idx_reference_frame + 1]
    future_frames = frames_list[
        idx_reference_frame + 1 : idx_reference_frame + len_head + 1
    ]
    return past_frames, future_frames


def slice_future_frames(
    future_frames: List[Frame], stride_future_frames: int
) -> List[Frame]:
    """
    Slices the future frames list, e.g. num_future_frames = 3, stride_future_frames = 2

    future_frames:                 f[1]   f[2]   f[3]   f[4]   f[5]   f[6]
                                           |             |             |
    sliced_future_frames:                f_[0]         f_[1]         f_[3]
    """
    len_head = len(future_frames)
    return future_frames[
        slice(stride_future_frames - 1, len_head + 1, stride_future_frames)
    ]


def slice_past_frames(past_frames: List[Frame], stride_past_frames: int) -> List[Frame]:
    """
    Slices the past frames list, e.g. num_past_frames = 3, stride_past_frames = 2

    past_frames:             f[-4]   f[-3]   f[-2]   f[-1]   f[0]
                               |               |               |
    sliced_past_frames:      f_[0]           f_[1]           f_[3]
    """
    assert len(past_frames) > 0, "Received empty list for slicing"
    len_tail = len(past_frames) - 1
    return past_frames[slice(0, len_tail + 1, stride_past_frames)]


@dataclass
class SegmentsSlicerConfig:
    sim_num_frames: int
    sim_stride: int


@dataclass
class GeneratorOutputInfo:
    type: str
    details: dict
    shape: Tuple[int]

    def __post_init__(self):
        if self.shape is not None:
            self.shape = tuple(self.shape)


def make_generator_output_info(
    info: Optional[GeneratorOutputInfo] = None,
    type: Optional[str] = None,
    shape: Optional[tuple] = None,
    details: Optional[dict] = None,
) -> GeneratorOutputInfo:
    """
    Convenience function to produce GeneratorOutputInfo
    """
    if info is not None:
        assert type is None and shape is None and details is None
        return GeneratorOutputInfo(**info) if isinstance(info, dict) else info
    else:
        return GeneratorOutputInfo(
            type=type,
            shape=shape,
            details=details,
        )


@dataclass
class SingleOutputGeneratorConfig:
    """
    Dataclass for holding the config of a features or a labels generator,
    that produces exactly one type of features or labels
    """

    output_name: str
    additional_parameters: dict

    def __init__(
        self,
        output_name: str,
        additional_parameters=dict(),
        **kwargs,
    ):
        """
        additional_parameters can be specified in two ways:
        - kwargs: to allow natural config dicts
        - additional_parameters: to allow conversion FeaturesGenerator_config <-> dict
        """
        self.output_name = output_name
        self.additional_parameters = {**additional_parameters, **kwargs}

    def to_dict(self) -> dict:
        return {"output_name": self.output_name, **self.additional_parameters}


@dataclass
class SingleOutputGeneratorClassConfig:
    """
    Dataclass for specifying the features or labels generator class, and config
    """

    path_to_class: str
    config: SingleOutputGeneratorConfig

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = SingleOutputGeneratorConfig(**self.config)


class SingleOutputGeneratorBase:
    """
    Base class for input features and labels generators.
    """

    def __init__(self, output_name: str):
        self._output_name = output_name

        self.mission = None
        self.ego_id = None

    @property
    def len_head(self) -> int:
        """
        See definition in __call__
        """
        raise NotImplementedError

    @property
    def len_tail(self) -> int:
        """
        See definition in __call__
        """
        raise NotImplementedError

    @property
    def stride(self) -> int:
        """
        Stride of the generated data.
        """
        raise NotImplementedError

    @property
    def space(self) -> spaces.Space:
        """
        Generator output space
        """
        raise NotImplementedError

    @property
    def output_info(self) -> GeneratorOutputInfo:
        raise NotImplementedError

    @property
    def output_name(self) -> str:
        """
        Generator output name
        """
        return self._output_name

    def __call__(self, past_frames: List[Frame], future_frames: List[Frame]):
        """
        past_frames and future_frames both include the frame from the current time step:

                                 past_frames                         future_frames
                 --------------------------------------------   -------------------------
        Frames:  f[-5]   f[-4]   f[-3]   f[-2]   f[-1]   f[0]   f[1]   f[2]   f[3]   f[4]
                 -------------------------------------          -------------------------
                              len_tail                                   len_head
        Current time step: 0

        NOTE: f[0] = past_frames[-1] which is the current frame

        Determining len_head and len_tail:
        * len_tail = (num_past_frames-1)*stride
        * len_head = num_future_frames*stride
        """
        raise NotImplementedError

    def reset(self, mission: Mission, ego_id: Any) -> None:
        self.mission = mission
        self.ego_id = ego_id

    @classmethod
    def from_config(cls, config: SingleOutputGeneratorConfig):
        return cls(**config.to_dict())

    def to_config(self) -> SingleOutputGeneratorClassConfig:
        return SingleOutputGeneratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=SingleOutputGeneratorConfig(output_name=self.output_name),
        )

    def check_frame_lengths(func):
        def _impl(self, past_frames: List[Frame], future_frames: List[Frame]):
            assert (
                len(future_frames) == self.len_head
            ), f"Expected future_frames to have a length of {self.len_head}, but got {len(future_frames)}"
            assert (
                len(past_frames) == self.len_tail + 1
            ), f"Expected past_frames to have a length of {self.len_tail + 1}, but got {len(past_frames)}"
            return func(self, past_frames, future_frames)

        return _impl


@dataclass
class MultiOutputGeneratorConfig:
    generators: List[SingleOutputGeneratorClassConfig]

    def __post_init__(self):
        # Dicts are a cleaner way to define generators,
        # because the key (output_name) is not a member of the generator itself.
        # TODO refactor towards dict
        if isinstance(self.generators, dict):
            self.generators = [
                self.generators[key] for key in sorted(self.generators.keys())
            ]

        if not isinstance(self.generators, list):
            raise ValueError("Generator configs must be provided as a list")
        self.generators = [
            ensure_init_type(config, SingleOutputGeneratorClassConfig)
            for config in self.generators
        ]


@dataclass
class MultiOutputGeneratorClassConfig:
    path_to_class: str
    config: MultiOutputGeneratorConfig

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = MultiOutputGeneratorConfig(**self.config)


class MultiOutputGenerator:
    """
    A class for handling multiple generators. We require a list of configs
    even if only one generator is used.
    It is always used as a top layer wrapper when generating a dataset.
    """

    def __init__(self, generators: List[SingleOutputGeneratorBase]):
        self.generators = generators
        self._len_tail = max([gen.len_tail for gen in self.generators])
        self._len_head = max([gen.len_head for gen in self.generators])
        self._outputs_names = [gen.output_name for gen in self.generators]

    @property
    def len_tail(self):
        return self._len_tail

    @property
    def len_head(self):
        return self._len_head

    @property
    def outputs_names(self):
        return self._outputs_names

    @property
    def outputs_infos(self) -> Dict[str, GeneratorOutputInfo]:
        return {gen.output_name: gen.output_info for gen in self.generators}

    def __call__(self, past_frames: List[Frame], future_frames: List[Frame]) -> dict:
        """
        Generates a dict with features for one time step
        """
        outputs = {
            gen.output_name: gen(
                past_frames=past_frames[len(past_frames) - gen.len_tail - 1 :],
                future_frames=future_frames[: gen.len_head],
            )
            for gen in self.generators
        }
        return outputs

    def reset(self, mission: Mission, ego_id: Any):
        for gen in self.generators:
            gen.reset(mission=mission, ego_id=ego_id)

    @classmethod
    def from_config(cls, config: MultiOutputGeneratorConfig):
        generators = [
            instantiate_from_config(generator_config)
            for generator_config in config.generators
        ]
        return cls(
            generators=generators,
        )

    def to_config(self) -> MultiOutputGeneratorClassConfig:
        return MultiOutputGeneratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=MultiOutputGeneratorConfig(
                [gen.to_config() for gen in self.generators]
            ),
        )

    def to_config_dict(self):
        return asdict(self.to_config())


class GeneratorBase:
    """
    A base class for a generator. It produces its output from a sequence of
    (already sliced) frames.
    """

    @property
    def space(self) -> spaces.Space:
        """
        Definition of the output space of the generator.
        """
        raise NotImplementedError

    @property
    def type(self) -> str:
        """
        Generator output type, e.g. "waypoints", "grid"
        """
        raise NotImplementedError

    @property
    def details(self) -> dict:
        """
        Additional details on the generator output such as info on state space semantics
        """
        raise NotImplementedError

    def __call__(*args, **kwargs):
        """
        Generates outputs from a sequence of frames
        """
        raise NotImplementedError


class CurrentFrameSingleOutputGeneratorBase(SingleOutputGeneratorBase):
    """
    Base class for generators that generate features/labels only from the current frame,
    i.e. it does not receive any historical or future frames.
    Required e.g. for the current ego pose.
    """

    def __init__(self, output_name: str):
        super().__init__(output_name)

    @property
    def len_tail(self) -> int:
        return 0

    @property
    def len_head(self) -> int:
        return 0

    @property
    def stride(self) -> int:
        return 0

    @property
    def space(self) -> spaces.Space:
        raise NotImplementedError

    def output_info(self) -> GeneratorOutputInfo:
        raise NotImplementedError

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(self, past_frames: List[Frame], future_frames: List[Frame]):
        raise NotImplementedError


class NumFramesSingleOutputGeneratorBase(SingleOutputGeneratorBase):
    def __init__(self, output_name: str, num_frames: int, stride: int):
        super().__init__(output_name=output_name)
        self.num_frames = num_frames
        self._stride = stride

    @property
    def len_head(self) -> int:
        raise NotImplementedError

    @property
    def len_tail(self) -> int:
        raise NotImplementedError

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def output_info(self) -> GeneratorOutputInfo:
        return GeneratorOutputInfo(
            type=self.gen.type,
            details={
                **self.gen.details,
                "num_frames": self.num_frames,
                "stride": self.stride,
            },
            shape=self.space.shape,
        )

    @property
    def space(self) -> spaces.Space:
        return self.gen.space

    def __call__(self, past_frames: List[Frame], future_frames: List[Frame]):
        raise NotImplementedError

    def to_config(self) -> SingleOutputGeneratorClassConfig:
        return SingleOutputGeneratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=SingleOutputGeneratorConfig(
                output_name=self.output_name,
                num_frames=self.num_frames,
                stride=self.stride,
            ),
        )


class FutureFramesSingleOutputGeneratorBase(NumFramesSingleOutputGeneratorBase):
    def __init__(self, output_name: str, num_frames: int, stride: int):
        super().__init__(output_name=output_name, num_frames=num_frames, stride=stride)

    @property
    def len_head(self) -> int:
        return self.num_frames * self.stride

    @property
    def len_tail(self) -> int:
        return 0

    def __call__(self, past_frames: List[Frame], future_frames: List[Frame]):
        raise NotImplementedError


class PastFramesSingleOutputGeneratorBase(NumFramesSingleOutputGeneratorBase):
    def __init__(self, output_name: str, num_frames: int, stride: int):
        super().__init__(output_name=output_name, num_frames=num_frames, stride=stride)

    @property
    def len_head(self) -> int:
        return 0

    @property
    def len_tail(self) -> int:
        return (self.num_frames - 1) * self.stride

    def __call__(self, past_frames: List[Frame], future_frames: List[Frame]):
        raise NotImplementedError
