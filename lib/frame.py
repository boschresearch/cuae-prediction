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

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np

from lib.utils import ensure_init_type


class DictClass:
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def to_dict(self):
        return asdict(self)


@dataclass
class DynamicState(DictClass):
    """
    Class for representing the dynamic state of an object
    Frame convention
    ^
    |           v_y      object
    |             ^ @@@@
    |             |@@@@
    |           @@|@@--------> v_x
    |          @@@@
    |
    |------------------------------>  global frame
    """

    x: float
    y: float
    yaw: float
    v_x: float
    v_y: float
    a_x: float
    a_y: float

    @classmethod
    def get_empty(cls):
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def get_pose(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw])

    def set_pose(self, pose: np.ndarray) -> None:
        self.x, self.y, self.yaw = pose

    def get_absolute_velocity(self) -> float:
        return np.linalg.norm([self.v_x, self.v_y])

    def get_absolute_acceleration(self) -> float:
        return np.linalg.norm([self.a_x, self.a_y])

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.get_absolute_velocity()])

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.yaw, self.v_x, self.v_y, self.a_x, self.a_y]
        )


class ObjectClasses(str, Enum):
    passenger_car = "passenger_car"
    large_vehicle = "large_vehicle"
    tractor_vehicle = "tractor_vehicle"
    pedestrian = "pedestrian"
    bicycle = "bicycle"
    motorbike = "motorbike"
    static_object = "static_object"
    unset = "unset"


@dataclass
class Meta(DictClass):
    """
    For the rollout format the object classes listed in ObjectClasses are admissible
    """

    obj_class: Union[str, ObjectClasses] = field(
        default=ObjectClasses.unset, compare=False
    )
    # you can use obj_valid to exclude unwanted objects (e.g. unwanted ego objects) via the
    # method Rollout.get_unique_object_ids. But invalid objects are still a part of the environment context.
    obj_valid: Optional[bool] = field(default=True, compare=False)
    obj_id: Optional[int] = field(default=None, compare=False)
    obj_dimensions: Union[Tuple[float, float], list, Tuple[None, None]] = field(
        default=(None, None), compare=False
    )
    info: Optional[dict] = field(default=None, compare=False)

    def __post_init__(self):
        assert (
            self.obj_class in ObjectClasses.__members__
        ), f"Invalid object class: {self.obj_class}"

    @classmethod
    def get_empty(cls):
        return cls()


@dataclass
class Obj(DictClass):
    dynamic_state: DynamicState = field(compare=False)
    meta: Meta = field(compare=False)

    def __post_init__(self):
        self.dynamic_state = ensure_init_type(self.dynamic_state, DynamicState)
        self.meta = ensure_init_type(self.meta, Meta)

    @classmethod
    def get_empty(cls):
        return Obj(DynamicState.get_empty(), Meta.get_empty())

    def get_pose(self) -> np.ndarray:
        return self.dynamic_state.get_pose()

    def set_pose(self, pose: np.ndarray) -> None:
        self.dynamic_state.set_pose(pose)

    def get_absolute_velocity(self) -> float:
        return self.dynamic_state.get_absolute_velocity()

    def get_state(self) -> float:
        return self.dynamic_state.get_state()


@dataclass
class Frame(DictClass):
    """ """

    timestamp: Optional[float] = None
    objects: List[Obj] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.objects = [ensure_init_type(obj, Obj) for obj in self.objects]

    @classmethod
    def get_empty(cls):
        return cls(None, [Obj.get_empty()], {})

    def get_object(self, object_id: int) -> Optional[Obj]:
        for obj in self.objects:
            if obj.meta.obj_id == object_id:
                return obj
        return None

    def is_object_in_frame(self, object_id: int) -> bool:
        return self.get_object(object_id) is not None
