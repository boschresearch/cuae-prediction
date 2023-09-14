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

from typing import List, Optional

import numpy as np
from gym import spaces

from features_and_labels.generators_base import (
    CurrentFrameSingleOutputGeneratorBase,
    FutureFramesSingleOutputGeneratorBase,
    GeneratorBase,
    GeneratorOutputInfo,
    PastFramesSingleOutputGeneratorBase,
    SingleOutputGeneratorBase,
    slice_future_frames,
    slice_past_frames,
)
from features_and_labels.utils import get_n_closest_objects
from lib.frame import Frame
from lib.geometric_transforms import (
    transform_point_into_frame,
    transform_pose_into_frame,
)
from lib.spaces import Box


class WaypointsLabelsGenerator(GeneratorBase):
    def __init__(self, num_frames: int, is_past_waypoints: bool = False):
        self._num_outputs = num_frames
        self.is_past_waypoints = is_past_waypoints

    @property
    def space(self) -> spaces.Space:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self._num_outputs,
                2,
            ),
            dtype=np.float64,
        )

    @property
    def type(self) -> str:
        if self.is_past_waypoints:
            return "past_waypoints"
        else:
            return "waypoints"

    @property
    def details(self) -> dict:
        return {"num_outputs": self._num_outputs}

    def __call__(
        self, sliced_frame_list: List[Frame], reference_frame: Frame, ego_id: int
    ) -> np.ndarray:
        """
        Creates waypoints from a sequence of sliced_frames w.r.t. to a the reference frame ego position
        """

        waypoints_global_frame = [
            frame.get_object(ego_id).get_pose()[:2] for frame in sliced_frame_list
        ]
        waypoints = [
            transform_point_into_frame(
                point,
                reference_frame.get_object(ego_id).get_pose()[:2],
                reference_frame.get_object(ego_id).get_pose()[2],
            )
            for point in waypoints_global_frame
        ]

        return np.array(waypoints)[np.newaxis, ...]  # adding batch dimension


class PastWaypointsLabelsGenerator(PastFramesSingleOutputGeneratorBase):
    def __init__(self, output_name: str, num_frames: int, stride: int):
        super().__init__(output_name=output_name, num_frames=num_frames, stride=stride)
        self.gen = WaypointsLabelsGenerator(self.num_frames, is_past_waypoints=True)

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        """
        Slices waypoints from a sequence of past frames, e.g. num_past_frames = 3, stride_past_frames = 2

        past_frames:                         f[-4]   f[-3]   f[-2]   f[-1]   f[0]
                                               |               |              |
        sliced_past_frames:               waypoints[0]    waypoints[1]   waypoints[2]
        Ego position at current time step: 0 is chosen as reference.
        """

        sliced_past_frames = slice_past_frames(past_frames, self.stride)

        return self.gen(sliced_past_frames, past_frames[-1], self.ego_id)


class FutureWaypointsLabelsGenerator(FutureFramesSingleOutputGeneratorBase):
    def __init__(self, output_name: str, num_frames: int, stride: int):
        super().__init__(output_name=output_name, num_frames=num_frames, stride=stride)
        self.gen = WaypointsLabelsGenerator(self.num_frames, is_past_waypoints=False)

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        """
        Slices waypoints from a sequence of future frames, e.g. num_future_frames = 3, stride_future_frames = 2

        future_frames:                 f[1]   f[2]   f[3]   f[4]   f[5]   f[6]
                                               |             |             |
        sliced_future_frames:            waypoints[0]   waypoints[1]  waypoints[2]
        Ego position at current time step: 0 (included in past_frames) is chosen as reference.
        """

        sliced_future_frames = slice_future_frames(future_frames, self.stride)

        return self.gen(sliced_future_frames, past_frames[-1], self.ego_id)


class EgoStateLabelsGenerator(CurrentFrameSingleOutputGeneratorBase):
    def __init__(
        self,
        output_name: str,
    ):
        super().__init__(
            output_name=output_name,
        )
        self.attributes = ["x", "y", "yaw", "v_x", "v_y", "a_x", "a_y"]

    @property
    def space(self) -> spaces.Space:
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.attributes),),
            dtype=np.float64,
        )

    @property
    def output_info(self) -> GeneratorOutputInfo:
        return GeneratorOutputInfo(
            type="ego_state",
            details={
                "attributes": self.attributes,
            },
            shape=self.space.shape,
        )

    @SingleOutputGeneratorBase.check_frame_lengths
    def __call__(
        self, past_frames: List[Frame], future_frames: List[Frame]
    ) -> np.ndarray:
        ego_dynamic_state = past_frames[-1].get_object(self.ego_id).dynamic_state
        return np.array([getattr(ego_dynamic_state, attr) for attr in self.attributes])[
            np.newaxis, ...
        ]  # adding batch dimension


class TrafficPosesLabelsGenerator(GeneratorBase):
    """
    Features generator that generates traffic objects (non-ego) features for objects within a certain distance to the ego. The distance threshold is tuned by max_distance_to_ego.
    Poses are in coordinate frame of ego in reference Frame.
    The features contain the objects' pose and an existence flag indicating whether the object is present at a certain time-step. In this case, the positions are set to zero.
    Objects can be of any class, e.g. passenger_car, bicycle, pedestrian, etc. Features are padded to the maximum assumed number of objects in a sequence of frames.
    """

    TRAFFIC_OBJECT_ATTRIBUTES = ["exists", "x", "y", "yaw"]
    MAX_TRAFFIC_OBJECTS = 64

    def __init__(
        self,
        num_frames: int,
        max_distance_to_ego: Optional[float] = np.inf,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.max_distance_to_ego = max_distance_to_ego

    @property
    def space(self) -> spaces.Space:
        return Box(
            low=-np.infty,
            high=np.infty,
            shape=(
                self.num_frames,
                self.MAX_TRAFFIC_OBJECTS,
                len(self.TRAFFIC_OBJECT_ATTRIBUTES),
            ),
            dtype=float,
        )

    @property
    def type(self) -> str:
        return "traffic_poses"

    @property
    def details(self) -> dict:
        return {
            "traffic_object_attributes": self.TRAFFIC_OBJECT_ATTRIBUTES,
            "max_traffic_objects": self.MAX_TRAFFIC_OBJECTS,
            "max_distance_to_ego": self.max_distance_to_ego,
        }

    def __call__(
        self, reference_frame: Frame, sliced_frames_list: List[Frame], ego_id: int
    ):
        """
        Generates a tensor of traffic objects' poses (in the desired ego reference frame) and existence flags.

        The tensor can be sparse, each unique obj has its own column. Example (cells contain attributes):

                | obj 1 | obj 2 | obj 3 | obj 4 |
        -----------------------------------------
        frame 0 |   X   |   X   |   X   |       |
        frame 1 |   X   |   X   |   X   |  X    |
        frame 2 |       |   X   |   X   |       |
        frame 3 |       |   X   |       |  X    |

        Returns:
            traffic_poses [num_frames, self.MAX_TRAFFIC_OBJECTS, len(self.TRAFFIC_OBJECT_ATTRIBUTES)]
        """
        reference_pose = reference_frame.get_object(ego_id).get_pose()

        traffic_poses = np.zeros(self.space.shape)

        # Cut to self.MAX_TRAFFIC_OBJECTS objects by finding the closest ones.
        closest_objects = get_n_closest_objects(
            sliced_frames_list, reference_pose, ego_id, self.MAX_TRAFFIC_OBJECTS
        )

        # create set of unique traffic_ids
        traffic_ids = {obj.meta.obj_id for obj in closest_objects}

        # assign position in traffic_poses tensor for each id via a dict of {id: position}
        traffic_ids = list(traffic_ids)
        traffic_ids = dict(zip(traffic_ids, range(len(traffic_ids))))

        # fill output tensor with transformed poses
        for i, traffic_frame in enumerate(sliced_frames_list):
            for obj in traffic_frame.objects:
                if obj.meta.obj_id == ego_id:
                    continue
                pose = transform_pose_into_frame(
                    obj.get_pose(), reference_pose[:2], reference_pose[2]
                )

                # skip object if farther than max distance to ego at origin
                dist_to_ego = np.linalg.norm(pose[:2], axis=0)
                if dist_to_ego > self.max_distance_to_ego:
                    continue

                # skip objects which were pruned due to limiting to N closest ones
                if obj.meta.obj_id not in traffic_ids:
                    continue

                traffic_poses[i, traffic_ids[obj.meta.obj_id], 0] = 1
                traffic_poses[i, traffic_ids[obj.meta.obj_id], 1:] = pose

        return traffic_poses[np.newaxis, ...]  # adding batch dimension


class PastTrafficPosesLabelsGenerator(PastFramesSingleOutputGeneratorBase):
    """
    Labels generator that provides past traffic poses over a number of time-steps. Holds an instance of TrafficPosesLabelsGenerator.
    """

    def __init__(
        self,
        output_name: str,
        num_frames: int,
        stride: int,
        max_distance_to_ego: float = np.inf,
    ):
        super().__init__(
            output_name=output_name,
            num_frames=num_frames,
            stride=stride,
        )

        self.gen = TrafficPosesLabelsGenerator(
            num_frames=self.num_frames, max_distance_to_ego=max_distance_to_ego
        )

    @property
    def space(self) -> spaces.Space:
        space = self.gen.space
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
        Generates past traffic poses in the coordinate frame of the current ego pose (which is the coordinate frame of predicted waypoints).
        """
        sliced_past_frames_list = slice_past_frames(past_frames, self.stride)
        return self.gen(
            reference_frame=past_frames[-1],
            sliced_frames_list=sliced_past_frames_list,
            ego_id=self.ego_id,
        )
