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
import json
import pickle
from dataclasses import dataclass
from itertools import groupby
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from dacite import from_dict as dacite_from_dict

from features_and_labels.map_base import MapConfig
from lib.frame import Frame
from lib.utils import ensure_init_type, to_python_standard


@dataclass
class StaticRolloutInfo:
    data_frequency: float  # data frequency in Hz
    map_info: MapConfig
    ego_id: Optional[int] = None
    objective: Optional[str] = None
    meta: Optional[dict] = None

    def __post_init__(self):
        self.map_info = ensure_init_type(self.map_info, MapConfig)


class Rollout:
    """
    Class for storing data generated during a rollout.
    Conventions:
    - frames: List[Frame]
    - static_info: List[dict] contains additional information (if available) such as map, control commands, etc ...
    """

    KEYS = ["frames", "static_info"]
    SUPPORTED_FILE_FORMATS = ["json", "pickle"]

    def __init__(self, static_info: StaticRolloutInfo, frames: List[Frame]) -> None:
        self.frames = frames
        self.static_info = static_info

    @classmethod
    def _check_file_format_supported(cls, path) -> None:
        if path.split(".")[-1] not in cls.SUPPORTED_FILE_FORMATS:
            raise ValueError(
                f"Unrecognized file extension in {path}! Only {cls.SUPPORTED_FILE_FORMATS} supported."
            )

    @classmethod
    def from_json_file(cls, json_file_path: str) -> "Rollout":
        with open(json_file_path, "r") as fp:
            data = json.load(fp)
        return cls.from_dict(data)

    @classmethod
    def from_pickle_file(cls, pickle_file_path: str) -> "Rollout":
        try:
            with open(pickle_file_path, "rb") as fp:
                data = pickle.load(fp)
            return cls.from_dict(data)
        except EOFError as e:
            print(f"Error loading file: {pickle_file_path}")
            raise e  # Re-raise the caught exception

    @classmethod
    def from_file(cls, file_path: str) -> "Rollout":
        """
        Load rollout from file, determine format from file extension.
        """
        cls._check_file_format_supported(file_path)
        if file_path.endswith(".json"):
            return cls.from_json_file(file_path)
        elif file_path.endswith(".pickle"):
            return cls.from_pickle_file(file_path)

    def save_as_json(self, path: str) -> None:
        """
        Save rollout to json file.
        """
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=2)

    def save_as_pickle(self, path: str) -> None:
        """
        Save rollout to pickle file.
        """
        with open(path, "wb") as fp:
            pickle.dump(self.to_dict(), fp, pickle.HIGHEST_PROTOCOL)

    def save(self, path: str) -> None:
        """
        Save rollout, determine format from file extension.
        """
        self._check_file_format_supported(path)
        if path.endswith(".json"):
            self.save_as_json(path)
        elif path.endswith(".pickle"):
            self.save_as_pickle(path)

    def to_dict(self) -> dict:
        """
        Casts the Rollout instance to a python standard data type dictionary
        """
        data = {
            "frames": to_python_standard(self.frames, dataclass_to_dict=True),
            "static_info": to_python_standard(self.static_info, dataclass_to_dict=True),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Rollout":
        frames = [dacite_from_dict(Frame, frame) for frame in data["frames"]]
        static_info = dacite_from_dict(StaticRolloutInfo, data["static_info"])
        return cls(static_info=static_info, frames=frames)

    @classmethod
    def empty_rollout_dict(cls) -> dict:
        dict_ = {}
        for key in cls.KEYS:
            dict_[key] = []
        return dict_

    def __getitem__(self, item: Union[slice, int]) -> "Rollout":
        if isinstance(item, int):
            item = slice(item, item + 1 if item != -1 else None)

        rollout_new = self.__class__(
            static_info=copy.deepcopy(self.static_info),
            frames=copy.deepcopy(self.frames[item]),
        )

        return rollout_new

    def __len__(self) -> int:
        return len(self.frames)

    def get_unique_object_ids(
        self,
        exclude_ego: bool = False,
        exclude_classes: List[str] = [],
        exclude_invalid_obj: bool = False,
    ) -> List[int]:
        """
        Iterates through the agents present in a rollout and returns all encountered unique ids.

        Args:
            exclude_ego: tunes whether to omit ego
            exclude_classes: tunes whether to omit certain classes
            exclude_invalid_obj: tunes whether to omit objects with set invalid flag
        Returns:
            unique_object_ids: list of unique object ids
        """
        unique_object_ids: set[int] = set(
            [
                obj.meta.obj_id
                for frame in self.frames
                for obj in frame.objects
                if (
                    obj.meta.obj_id is not None
                    and obj.meta.obj_class not in exclude_classes
                    and (not exclude_invalid_obj or obj.meta.obj_valid is True)
                )
            ]
        )
        if exclude_ego and self.static_info.ego_id:
            assert (
                self.static_info.ego_id in unique_object_ids
            ), f"Ego with id: {self.static_info.ego_id} does not exist in the rollout."
            unique_object_ids.remove(self.static_info.ego_id)
        return list(unique_object_ids)

    def get_frames_with_object(
        self,
        obj_id: int,
    ) -> Tuple[List[Frame], bool]:
        """
        Returns all frames in which the object with matching obj_id is present
        return: frames_with_object: frame-list
        return: is_only_consecutive_frames: boolean, true if all object frames are consecutive frames
        """
        frames_with_object = []
        results_is_object_in_frame = []

        for frame in self.frames:
            is_object_in_frame = frame.is_object_in_frame(obj_id)
            results_is_object_in_frame.append(is_object_in_frame)

            if is_object_in_frame:
                frames_with_object.append(frame)

        is_only_consecutive_frames = (
            True
            if len([boolean for boolean in groupby(results_is_object_in_frame)]) <= 1
            else False
        )

        return frames_with_object, is_only_consecutive_frames

    def get_object_time_series(self, object_id: int) -> pd.Series:
        if object_id is None:
            raise ValueError(f"Invalid object_id = {object_id}")
        series = pd.Series(
            data=[frame.get_object(object_id) for frame in self.frames],
            index=[frame.timestamp for frame in self.frames],
        )
        return series

    def clone(self) -> "Rollout":
        return copy.deepcopy(self)


def cut_rollout(
    rollout: Rollout,
    ego_id: int,
    strip_traveled_metres_behind: float = 0,
    strip_traveled_metres_ahead: float = 0,
) -> Rollout:
    """
    Only keep frames where ego_id is present.
    Strip frames in the beginning until ego has traveled at least strip_traveled_metres_behind.
    Strip frames in the end with cumulated ego travel distance strip_traveled_metres_ahead.
    """
    static_info = copy.deepcopy(rollout.static_info)
    static_info.ego_id = ego_id

    # avoid jumps in time and only get first series of consecutive frames
    frame_ixs = [
        ix for ix, frame in enumerate(rollout.frames) if frame.get_object(ego_id)
    ]
    time_jump_ixs = np.where(np.diff(frame_ixs) > 1)[0]
    if len(time_jump_ixs):
        frame_ixs = frame_ixs[
            : time_jump_ixs[0] + 1
        ]  # +1 because np.diff reduces len by 1
    frames = rollout.frames[frame_ixs[0] : frame_ixs[-1] + 1]

    traveled_metres = np.cumsum(
        [
            np.linalg.norm(
                frame.get_object(ego_id).get_pose()[:2]
                - next_frame.get_object(ego_id).get_pose()[:2]
            )
            for frame, next_frame in zip([frames[0]] + frames[:-1], frames)
        ]
    )
    assert traveled_metres.ndim == 1  # check if dim. is corr.
    indices = np.where(
        np.logical_and(
            traveled_metres >= strip_traveled_metres_behind,
            traveled_metres <= traveled_metres[-1] - strip_traveled_metres_ahead,
        )
    )[0]

    frames_filtered = [frames[i] for i in indices]
    return Rollout(
        static_info=static_info,
        frames=frames_filtered,
    )
