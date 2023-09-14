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

from typing import List

import numpy as np

from lib.frame import Frame, Obj


def get_n_closest_objects(
    frame_list: List[Frame], reference_pose: np.ndarray, ego_id: int, n: int
) -> List[Obj]:
    """
    Returns the n closest objects to reference_pose, excluding ego_id.
    "Closest" here means euclidean distance to reference pose,
    w.r.t. to the earliest frame the object appears in.
    """
    objects = {}
    for frame in frame_list:
        for obj in frame.objects:
            if obj.meta.obj_id == ego_id:
                continue
            if obj.meta.obj_id not in objects:
                objects[obj.meta.obj_id] = obj

    object_list = [obj for _, obj in objects.items()]
    return sorted(
        object_list,
        key=lambda obj: np.linalg.norm(
            reference_pose[:2] - np.asarray([obj.dynamic_state.x, obj.dynamic_state.y])
        ),
    )[:n]
