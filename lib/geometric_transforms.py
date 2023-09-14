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

import numpy as np


def wrap_angle(angle):
    """
    Normalize angle to [-pi, pi)
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _rotation_matrix(rotation_angle):
    c = np.cos(rotation_angle)
    s = np.sin(rotation_angle)
    return np.array(
        [
            [c, s],
            [-s, c],
        ]
    )


def transform_point_into_frame(
    point: np.ndarray, offset: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """
    Transform point (x,y) -> (X,Y)
    such that offset (x,y) -> (X,Y) = (0,0) and rotation_angle theta -> Theta = 0
    in the new coordinate system.
    """
    t = -np.array(offset)
    r = _rotation_matrix(rotation_angle)
    return np.dot(r, point + t)


def transform_points_into_frame(
    points: np.ndarray, offset: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """
    Similar to transform_point, with points.shape = (#points, 2).
    """
    offset_points = points - np.expand_dims(offset[:], axis=0)
    r = _rotation_matrix(rotation_angle)
    transformed_points = np.dot(r, offset_points.T).T
    return transformed_points


def transform_points_out_of_frame(
    points: np.ndarray, offset: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """
    Similar to transform_point_out_of_frame, with points.shape = (#points, 2).
    """
    transformed_points = transform_points_into_frame(points, [0, 0], -rotation_angle)
    transformed_points = transform_points_into_frame(
        transformed_points, -1 * np.array(offset), 0
    )
    return transformed_points


def transform_pose_into_frame(
    pose: np.ndarray, offset: np.ndarray, rotation_angle: float
) -> np.ndarray:
    """
    Transform pose (x,y,theta) -> (X,Y,Theta)
    such that offset (x,y) -> (X,Y) = (0,0) and rotation_angle theta -> Theta = 0
    in the new coordinate system.
    """
    return np.concatenate(
        (
            transform_point_into_frame(pose[:2], offset, rotation_angle),
            [wrap_angle(pose[2] - rotation_angle)],
        )
    )


def transform_velocity_to_local_frame(vel_global: np.ndarray, yaw: float) -> np.ndarray:
    """
    Transforms velocity vector from global frame orientation to local frame orientation.

    y                   v_lon
    ^     v_lat  v_y    ^
    |         ^   ^    /  object
    |          \  |  @@@@
    |           \ |@@@@
    |           @@|@@--------> v_x
    |          @@@@
    |
    |------------------------------>x

    v_x = v_lon * cos(yaw) - v_lat * sin(yaw)
    v_y = v_lon * sin(yaw) + v_lat * cos(yaw)
    """  # noqa W605
    R = np.transpose(_rotation_matrix(yaw))
    return np.linalg.solve(R, vel_global)


def transform_acc_to_local_frame(
    acc_global: np.ndarray, vel_local: np.ndarray, yaw: float, yaw_dot: float
) -> np.ndarray:
    """
    Transforms acceleration vector from global frame orientation to local frame orientation.

    y                    a_lon
    ^     a_lat  a_y    ^
    |         ^   ^    /  object
    |          \  |  @@@@
    |           \ |@@@@
    |           @@|@@--------> a_x
    |          @@@@
    |
    |------------------------------>x

    a_x = a_lon * cos(yaw) - v_lon * sin(yaw) * yaw_dot - a_lat * sin(yaw) - v_lat * cos(yaw) * yaw_dot
    a_y = a_lon * sin(yaw) + v_lon * cos(yaw) * yaw_dot + a_lat * cos(yaw) - v_lat * sin(yaw) * yaw_dot
    """  # noqa W605
    R = np.transpose(_rotation_matrix(yaw))
    b = np.array(
        [
            -vel_local[0] * np.sin(yaw) * yaw_dot
            - vel_local[1] * np.cos(yaw) * yaw_dot,
            vel_local[0] * np.cos(yaw) * yaw_dot - vel_local[1] * np.sin(yaw) * yaw_dot,
        ]
    )
    b = acc_global - b

    return np.linalg.solve(R, b)
