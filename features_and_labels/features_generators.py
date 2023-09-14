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

import hashlib
from typing import List, Union

import numpy as np
from gym import spaces

from features_and_labels.generators_base import (
    CurrentFrameSingleOutputGeneratorBase,
    FutureFramesSingleOutputGeneratorBase,
    GeneratorBase,
    GeneratorOutputInfo,
    PastFramesSingleOutputGeneratorBase,
    SingleOutputGeneratorBase,
    SingleOutputGeneratorClassConfig,
    SingleOutputGeneratorConfig,
    slice_future_frames,
    slice_past_frames,
)
from features_and_labels.map_base import SampledBoundary
from lib.base_classes import FULL_EGO_ATTRIBUTES
from lib.frame import Frame
from lib.geometric_transforms import (
    transform_acc_to_local_frame,
    transform_points_into_frame,
    transform_pose_into_frame,
    transform_velocity_to_local_frame,
)
from lib.kinematic_models import KinematicBicycleModel
from lib.spaces import Box
from lib.utils import get_path_to_class, pad_to_shape


class EgoStateFeaturesGenerator(GeneratorBase):
    def __init__(self, num_frames: int, attributes: List[str] = FULL_EGO_ATTRIBUTES):
        self._num_outputs = num_frames
        self._attributes = attributes

    @property
    def space(self) -> spaces.Space:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self._num_outputs,
                len(self._attributes),
            ),
            dtype=np.float64,
        )

    @property
    def type(self) -> str:
        return "ego_states"

    @property
    def details(self) -> dict:
        return {"attributes": self._attributes, "num_outputs": self._num_outputs}

    @staticmethod
    def _transform_ego_state_into_ref_frame(
        ego_state: np.array, yaw: float, offset: np.array
    ) -> np.array:
        ego_state[3:5] = transform_velocity_to_local_frame(ego_state[3:5], yaw)
        # TODO: calculate yaw rate to have more accurate values
        ego_state[5:7] = transform_acc_to_local_frame(
            ego_state[5:7],
            ego_state[3:5],
            yaw,
            0.0,  # zero yaw-rate approximation
        )
        ego_state[:3] = transform_pose_into_frame(ego_state[:3], offset, yaw)

        return ego_state

    def __call__(
        self, sliced_frame_list: List[Frame], ego_id: int, reference_frame: Frame = None
    ) -> np.ndarray:
        # TODO: use explicit attributes, i.e. make obj.dynamic_state also able to return the sematics of the state - ["x", "y", "yaw", ...]
        ego_state_semantics = FULL_EGO_ATTRIBUTES

        ego_params = getattr(
            sliced_frame_list[0].get_object(ego_id).meta, "obj_dimensions"
        )

        dynamic_states = [
            frame.get_object(ego_id).dynamic_state for frame in sliced_frame_list
        ]
        ego_states = [
            np.append(dyn_state.to_array(), ego_params) for dyn_state in dynamic_states
        ]

        if reference_frame is not None:
            # TODO: rename attributes v_x->v_lon, etc and adapt usage in KinematicModels.forward()
            ego_ref_state = reference_frame.get_object(ego_id).dynamic_state.to_array()
            for idx, ego_state in enumerate(ego_states):
                ego_states[idx] = self._transform_ego_state_into_ref_frame(
                    ego_state, ego_ref_state[2], ego_ref_state[:2]
                )

        ego_states = np.array(ego_states)

        assert set(self._attributes) <= set(
            ego_state_semantics
        ), f"Attribute set {self._attributes} is not a subset of supported state semantics {ego_state_semantics}"

        # return only the requested state variables
        ego_states = ego_states[
            :, [ego_state_semantics.index(attribute) for attribute in self._attributes]
        ]

        return ego_states[np.newaxis, ...]


class PastEgoStateFeaturesGenerator(PastFramesSingleOutputGeneratorBase):
    """
    Features generator that creates num_frames past_ego_states in reference to an ego state in a local or the global frame.
    Used for prediction and reconstruction via a kinematic model.
        Note:
            - Select the global reference frame through: and reference_frame_idx=None
            - The reference_frame_idx indicates the frame from which the local ego reference pose is chosen
            e.g.  reference_frame_idx=-1, num_frames=6, stride=1

                            past_frames
                ----------------------------------------
                    past_ego_state_feature_frames
                ----------------------------------------
    Frames:      f[-5]  f[-4]  f[-3]  f[-2]  f[-1]  f[0]
                --------------------------------------|-
                                                      |
                                        reference_frame (current frame)


            e.g.  reference_frame_idx=0, num_frames=6, stride=1

                            past_frames
                ----------------------------------------
                    past_ego_state_feature_frames
                ----------------------------------------
    Frames:     f[-5]  f[-4]  f[-3]  f[-2]  f[-1]  f[0]
                --|-------------------------------------
                  |
            reference_frame

    :return past_ego_state_features, shape [num_frames, num_ego_state_attributes]
    """

    def __init__(
        self,
        output_name: str,
        num_frames: int,
        stride: int,
        attributes: List[str] = FULL_EGO_ATTRIBUTES,
        reference_frame_idx: Union[
            int, None
        ] = -1,  # current frame (last frame in the past); None skips any transformation
    ):
        super().__init__(output_name=output_name, num_frames=num_frames, stride=stride)
        self.gen = EgoStateFeaturesGenerator(
            num_frames=num_frames, attributes=attributes
        )
        self._reference_frame_idx = reference_frame_idx

    @property
    def output_info(self) -> GeneratorOutputInfo:
        return GeneratorOutputInfo(
            type=self.gen.type,
            details={
                **self.gen.details,
                "num_frames": self.num_frames,
                "stride": self.stride,
                "reference_frame_idx": self._reference_frame_idx,
            },
            shape=self.gen.space.shape,
        )

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        sliced_past_frame_list = slice_past_frames(past_frames, self.stride)

        # TODO: Refactor toward dependency inversion
        if self._reference_frame_idx is None:
            reference_frame = None
        else:
            reference_frame = sliced_past_frame_list[self._reference_frame_idx]

        return self.gen(sliced_past_frame_list, self.ego_id, reference_frame)


class MapPolylineVectorFeaturesGenerator(CurrentFrameSingleOutputGeneratorBase):
    """
    Features generator that generates map polyline features consisting of vectors of map points within a certain distance to the ego. The features contain the xy positions of the vector, polyline type (solid, broken, etc) represented by a one-hot encoding, and the index of the polyline the vector belongs to. The points are either in the local ego or global coordinate frame. Features are padded to the maximum number of map points in a scene.
    """

    MAP_LINES_ID_MAPPING = {"solid": 0, "broken": 1, "unpaved": 2}

    MAP_POLYLINE_VECTOR_ATTRIBUTES = [
        "x1",
        "y1",
        "x2",
        "y2",
        *[0] * len(MAP_LINES_ID_MAPPING),  # one-hot encoding
        "polyline_index",
    ]  # len(self.MAP_LINES_ID_MAPPING) gives length of one-hot encoding
    MAX_NUM_POLYLINE_MAP_VECTORS = (
        10000  # From waymo, features are padded with zeros until this point
    )

    def __init__(
        self,
        output_name: str,
        ref_frame: str = "local",
        max_distance_to_ego: float = None,  # [m], More distant points to the ego are excluded (None <=> take all points)
    ):
        if ref_frame not in ["local", "global"]:
            raise ValueError(
                f"Unknown reference frame={ref_frame}, allowed values: [local, global]"
            )

        self._ref_frame = ref_frame
        self._max_distance_to_ego = (
            np.inf if max_distance_to_ego is None else max_distance_to_ego
        )

        # hashing function for encoding polyline information
        self.hashing_func = hashlib.md5

        # store mapping of [str] hash: [int] id since hash is too big to be used as a torch tensor
        self.poly_hash_id_mapping = {}

        self.shape = (
            self.MAX_NUM_POLYLINE_MAP_VECTORS,
            len(self.MAP_POLYLINE_VECTOR_ATTRIBUTES),
        )

        super().__init__(
            output_name=output_name,
        )

    @property
    def space(self) -> spaces.Space:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=self.shape,
            dtype=np.float64,
        )

    @property
    def output_info(self) -> GeneratorOutputInfo:
        return GeneratorOutputInfo(
            type="map_polyline_vector_features",
            details={
                "attributes": self.MAP_POLYLINE_VECTOR_ATTRIBUTES,
            },
            shape=self.space.shape,
        )

    def get_unique_polyline_id(self, location: str, polyline: SampledBoundary) -> int:
        """
        Generates a unique id given the polyline location, lane index, side, and type. The hashing algorithm performs the following:
        1. generates UTF-8 string representation
        2. hashes the given string using the given algorithm
        3. retrives its hex hash tag
        4. converts hex to integer
        """
        polyline_info = (
            location + str(polyline.lane_index) + polyline.side + polyline.type
        )
        polyline_hashed_info = self.hashing_func(polyline_info.encode()).hexdigest()
        if polyline_hashed_info not in self.poly_hash_id_mapping:
            self.poly_hash_id_mapping[polyline_hashed_info] = len(
                self.poly_hash_id_mapping
            )
        return self.poly_hash_id_mapping[polyline_hashed_info]

    @staticmethod
    def generate_polyline_type_onehot_encoding(polyline: SampledBoundary) -> List[int]:
        onehot_enc = np.zeros(
            len(MapPolylineVectorFeaturesGenerator.MAP_LINES_ID_MAPPING)
        )
        if polyline.type in ["unknown", "concretebarrier"]:
            print(
                f"Warning: Attempt to add '{polyline.type}' to MAP_LINES_ID_MAPPING is currently unsupported. This action leads to incompatible shapes in the data."
            )
            print(
                f"TODO: Recommendation: Enable the encoder to effectively handle '{polyline.type}' elements."
            )
            type_id = MapPolylineVectorFeaturesGenerator.MAP_LINES_ID_MAPPING["solid"]
        else:
            type_id = MapPolylineVectorFeaturesGenerator.MAP_LINES_ID_MAPPING[
                polyline.type
            ]
        onehot_enc[type_id] = 1
        return onehot_enc

    @staticmethod
    def enforce_polyline_points_contiguity(
        are_pts_close_to_ego: np.ndarray,
    ) -> np.ndarray:
        """
        Enforces that the polyline points considered close to ego are all contiguous (no jumps).
        :Args
            are_pts_close_to_ego: bools array indicating polyline points close to ego, shape [num_polyline_pts]
        :Returns
            are_pts_close_to_ego: bools array indicating polyline points close to ego after contiguity enforcement,
                shape [num_polyline_pts]
        """
        (indices_points_close_to_ego,) = np.where(are_pts_close_to_ego)
        if indices_points_close_to_ego.size == 0:
            return are_pts_close_to_ego
        if indices_points_close_to_ego.size > 1:
            first_idx = indices_points_close_to_ego[0]
            last_idx = indices_points_close_to_ego[-1]
            are_pts_close_to_ego[first_idx : last_idx + 1] = True
        return are_pts_close_to_ego

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        """
        Iterates through polylines in the map and represents them as a sequence of vectors. Vector attributes are xy positions of start and end point, one-hot encoding of the polyline type, and a unique (in the entire dataset) polyline id.
        :return padded_map_polyline_vectors, shape [1, max_num_vectors, 4+num_poly_types+1]
        """
        reference_frame = past_frames[0]
        ego_ref_state = reference_frame.get_object(self.ego_id).dynamic_state.to_array()

        polylines = self.mission.map_info.border_lines
        # TODO hacky to set location to empty string. Consider switching to None and adapting the rest of the code.
        if hasattr(self.mission.map_info, "location"):
            location = self.mission.map_info.location
        else:
            location = ""
        map_polyline_vectors = []  # (point1, point2, polyline_type, polyline_index)

        for polyline in polylines:
            polyline_pts = np.asarray(polyline.xy)  # [num_points, dim]

            # compute distance to ego
            dist_to_ego = np.linalg.norm(polyline_pts - ego_ref_state[:2], axis=1)

            # compute distance treshold mask
            are_pts_close_to_ego = dist_to_ego < self._max_distance_to_ego

            # check that points satisfying distance condition are contiguous (no jumps)
            are_pts_close_to_ego = (
                MapPolylineVectorFeaturesGenerator.enforce_polyline_points_contiguity(
                    are_pts_close_to_ego
                )
            )

            # filter points according to mask
            polyline_pts = polyline_pts[are_pts_close_to_ego, :]

            # skip polyline if contains a single point
            if polyline_pts.shape[0] <= 1:
                continue

            # transform to local frame
            if self._ref_frame == "local":
                polyline_pts = transform_points_into_frame(
                    polyline_pts,
                    offset=ego_ref_state[:2],
                    rotation_angle=ego_ref_state[2],
                )

            # generate polyline vectors
            polyline_vectors = np.concatenate(
                (polyline_pts[:-1, :], polyline_pts[1:, :]), axis=1
            )

            # generate auxiliary info
            one_hot_enc = np.tile(
                MapPolylineVectorFeaturesGenerator.generate_polyline_type_onehot_encoding(
                    polyline
                ),
                (polyline_vectors.shape[0], 1),
            )  # repeat for num vectors along new axis
            polyline_id = np.tile(
                np.array(float(self.get_unique_polyline_id(location, polyline))),
                (polyline_vectors.shape[0], 1),
            )

            # concatenate into polyline features
            polyline_feature = np.concatenate(
                (polyline_vectors, one_hot_enc, polyline_id), axis=1
            )
            map_polyline_vectors.append(polyline_feature)

        map_polyline_vectors = np.concatenate(map_polyline_vectors, axis=0)
        padded_map_polyline_vectors = pad_to_shape(
            map_polyline_vectors, self.shape, padder=np.nan
        )

        return padded_map_polyline_vectors[np.newaxis, ...]  # add batch dim

    def to_config(self) -> SingleOutputGeneratorClassConfig:
        return SingleOutputGeneratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=SingleOutputGeneratorConfig(
                ref_frame=self._ref_frame,
                output_name=self.output_name,
                max_distance_to_ego=self._max_distance_to_ego,
            ),
        )


class ActionsFeaturesGenerator(GeneratorBase):
    """
    Features generator that provides actions (acceleration, steering angle) over a number of timesteps.
    """

    attributes = ["a", "delta"]  # acc, steering angle

    def __init__(
        self,
        sample_time: float,
    ):
        self._sample_time = sample_time

    @property
    def space(self) -> spaces.Space:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.attributes),),
            dtype=np.float64,
        )

    @property
    def type(self) -> str:
        return "actions"

    @property
    def details(self) -> dict:
        return {
            "attributes": self.attributes,
            "sample_time": self._sample_time,
        }

    def __call__(self, sliced_frame_list: List[Frame], ego_id: int) -> np.ndarray:
        """
        Generates actions (acceleration, steering angle) features.
        Each action is computed using two consecutive states, with the resulting action tensor of length sliced_frame_list - 1.
        :return actions, shape [batch_size, sliced_frame_list - 1, 2]
        """
        actions = np.empty((len(sliced_frame_list) - 1, len(self.attributes)))

        # compute actions via inverse kinematic model
        ego_objects = [frame.get_object(ego_id) for frame in sliced_frame_list]
        ego_states = [obj.get_state() for obj in ego_objects]
        ego_params = getattr(
            sliced_frame_list[0].get_object(ego_id).meta, "obj_dimensions"
        )

        # iterate over states chronologically
        for i in range(len(ego_states) - 1):
            actions[i, :] = KinematicBicycleModel.compute_actions(
                ego_states[i],  # state k
                ego_states[i + 1],  # state k+1
                ego_params,
                self._sample_time,
            )

        return actions[np.newaxis, ...]


class FutureActionsFeaturesGenerator(FutureFramesSingleOutputGeneratorBase):
    """
    Features generator that provides future actions (acceleration, steering angle) over a number of timesteps.
    """

    def __init__(
        self,
        output_name: str,
        num_frames: int,
        stride: int,
        data_frequency: float,
    ):
        super().__init__(
            output_name=output_name,
            num_frames=num_frames,
            stride=stride,
        )
        self.num_actions = num_frames
        self.data_frequency = data_frequency
        self.gen = ActionsFeaturesGenerator(
            sample_time=self.stride / self.data_frequency,
        )

    @property
    def _data_frequency(self):
        return self.data_frequency

    @property
    def space(self) -> spaces.Space:
        space = self.gen.space
        space.prepend_dims((self.num_actions,))
        return space

    @property
    def output_info(self) -> GeneratorOutputInfo:
        return GeneratorOutputInfo(
            type=self.gen.type,
            details={**self.gen.details},
            shape=self.space.shape,
        )

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        """
        Generates chronological future actions (acceleration, steering angle) features.
        Each action is computed using two consecutive states, with the resulting action tensor of length num_frames

        Example num_frames = 4

        past_frames                            future_frames
         ---------   ------------------------------------------------------------------> t
         state_(t)   state_(t+1)   state_(t+2)   state_(t+3)   state_(t+4)   state_(t+5)
             |       /
         action(t)
                     |             /
                     action(t+1)
                                 |              /
                                 action(t+2)
                                                 |            /
                                                 action(t+3)
                                                             |            /
                                                             action(t+4)
         current ----------------------------------------------------------------------

        :return future_actions, shape [batch_size, num_timesteps, 2]
        """
        sliced_future_frame_list = slice_future_frames(future_frames, self.stride)
        return self.gen(past_frames + sliced_future_frame_list, self.ego_id)

    def to_config(self) -> SingleOutputGeneratorClassConfig:
        return SingleOutputGeneratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=SingleOutputGeneratorConfig(
                output_name=self.output_name,
                num_frames=self.num_frames,
                stride=self.stride,
                data_frequency=self._data_frequency,
            ),
        )


class PastActionsFeaturesGenerator(PastFramesSingleOutputGeneratorBase):
    """
    Features generator that provides the past actions (acceleration, steering angle) over a number of timesteps.
    """

    def __init__(
        self,
        output_name: str,
        num_frames: int,
        stride: int,
        data_frequency: float,
    ):
        super().__init__(
            output_name=output_name,
            num_frames=num_frames,
            stride=stride,
        )
        self.num_actions = num_frames - 1
        self.data_frequency = data_frequency
        self.gen = ActionsFeaturesGenerator(
            sample_time=self.stride / self.data_frequency,
        )

    @property
    def _data_frequency(self):
        return self.data_frequency

    @property
    def space(self) -> spaces.Space:
        space = self.gen.space
        space.prepend_dims((self.num_actions,))
        return space

    @property
    def output_info(self) -> GeneratorOutputInfo:
        return GeneratorOutputInfo(
            type=self.gen.type,
            details={**self.gen.details},
            shape=self.space.shape,
        )

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        """
        Generates chronological past actions (acceleration, steering angle) features.
        Each action is computed using two consecutive states, with the resulting action tensor of length num_frames - 1.

        Example num_frames = 4

                                past_frames
        -------------------------------------------------------------------> t
        state_(t-4)   state_(t-3)   state_(t-2)   state_(t-1)   state_(t)
        |             /
        action(t-4)
                      |             /
                      action(t-3)

                                    |             /
                                    action(t-2)
                                                  |             /
                                                  action(t-1)
        ------------------------------------------------------  current
        :return past_actions, shape [batch_size, num_timesteps - 1, 2]
        """
        sliced_past_frame_list = slice_past_frames(past_frames, self.stride)
        return self.gen(sliced_past_frame_list, self.ego_id)

    def to_config(self) -> SingleOutputGeneratorClassConfig:
        return SingleOutputGeneratorClassConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=SingleOutputGeneratorConfig(
                output_name=self.output_name,
                num_frames=self.num_frames,
                stride=self.stride,
                data_frequency=self._data_frequency,
            ),
        )
