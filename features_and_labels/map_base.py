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
from typing import Any, Dict, List, Tuple

import numpy as np

BOUNDARY_SIDE_TO_STRING = {
    0: "left",
    1: "right",
}


@dataclass
class SampledBoundary:
    lane_index: Any
    side: str  # left, right
    type: str  # in grid_features.base.MAP_LINES
    xy: List[Tuple[float, float]] = field(default_factory=list)

    def to_config(self) -> dict:
        return asdict(self)

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)


@dataclass
class SampledBoundariesPadded:
    """
    The class contains List[SampledBoundary] as numpy arrays. If the border-lines have different size, the points are padded with np.inf.
    This class is needed for run-time optimized boundary handling.
    """

    lane_indices: np.ndarray = np.array([])
    sides: np.ndarray = np.array([])
    types: np.ndarray = np.array([])
    point_arrays_padded: np.ndarray = np.array([])


@dataclass
class BorderPolygonsPadded:
    """
    The class contains border-polygons as numpy arrays. If the polygons have different size, the points are padded with np.inf.
    This is needed for run-time optimized polygon handling.
    """

    lane_indices: np.ndarray = np.array([])
    border_polygons_padded: np.ndarray = np.array([])


@dataclass
class LaneInfo:
    length: float
    speed_limit: float
    priority: int = 0


@dataclass
class MapConfig:
    path_to_class: str
    config: Any


class MapBase:
    """
    Base class for map.
    """

    def __init__(self, road_graph: Any):
        self._road_graph = road_graph
        self._border_lines = self._extract_border_lines()
        self._border_lines_numpy = self._convert_border_lines_numpy(
            border_lines=self.border_lines
        )
        self._border_polygons = self._extract_border_polygons(self.border_lines)
        self._border_polygons_numpy = self._convert_border_polygons_numpy(
            border_polygons=self._border_polygons
        )

    def _extract_border_polygons(
        self, border_lines: List[SampledBoundary]
    ) -> Dict[Any, List[np.ndarray]]:
        border_polygons = {}
        lane_indices = list(set([b.lane_index for b in border_lines]))
        for lane_index in lane_indices:
            interior_boundaries = []
            exterior_boundaries = {"left": [], "right": []}
            for b in border_lines:
                if b.lane_index == lane_index:
                    if b.side == "interior":
                        interior_boundaries.append(b.xy)
                    else:
                        exterior_boundaries[b.side].extend(b.xy)

            if (
                len(exterior_boundaries["left"]) == 0
                or len(exterior_boundaries["right"]) == 0
            ):
                raise ValueError(
                    "A lane segment must have at least one left and one right exterior boundary."
                )

            border_polygons.update(
                {
                    lane_index: [
                        np.concatenate(
                            (
                                exterior_boundaries["right"],
                                exterior_boundaries["left"][::-1],
                            )
                        )
                    ]
                    + interior_boundaries
                }
            )

        return border_polygons

    def _convert_border_lines_numpy(
        self, border_lines: List[SampledBoundary]
    ) -> SampledBoundariesPadded:
        if not border_lines:
            return SampledBoundariesPadded()

        max_border_line_points = np.max([len(bl.xy) for bl in border_lines])
        border_lines_xy = np.stack(
            [
                np.pad(
                    bl.xy,
                    ((0, max_border_line_points - len(bl.xy)), (0, 0)),
                    constant_values=(np.inf, np.inf),
                )
                for bl in border_lines
            ]
        )

        # lane-indices can be of Any type (e.g. tuples) and, thus, cannot be converted directly.
        lane_indices_numpy = np.empty(len(border_lines), dtype=object)
        lane_indices_numpy[:] = [bl.lane_index for bl in border_lines]

        return SampledBoundariesPadded(
            lane_indices=lane_indices_numpy,
            sides=np.array([bl.side for bl in border_lines]),
            types=np.array([bl.type for bl in border_lines]),
            point_arrays_padded=border_lines_xy,
        )

    def _convert_border_polygons_numpy(
        self, border_polygons: Dict[str, List[np.ndarray]]
    ) -> BorderPolygonsPadded:
        """
        Convert border_polygons as dict of {lane_id: list of polygon} into an array of lane-ids and an array of polygons.
        Since the polygons have different length (number of points), the array of polygons is padded with np.inf.

        Args:
            - border_polygons: dict of {lane_id: list of polygons} in which the list of polygons can be inhomogeneous.
        Returns:
            - array of lane-ids
            - array of homogeneous polygons (padded with np.inf)
        """
        if not border_polygons:
            return BorderPolygonsPadded()

        lane_indices, border_polygons_padded = zip(*border_polygons.items())
        border_polygons_padded = [
            np.stack([xy for seg in bp for xy in seg], axis=0)
            for bp in border_polygons_padded
        ]

        max_polygon_points = np.max([len(p) for p in border_polygons_padded])
        border_polygons_padded = np.stack(
            [
                np.pad(
                    bp,
                    ((0, max_polygon_points - len(bp)), (0, 0)),
                    constant_values=(np.inf, np.inf),
                )
                for bp in border_polygons_padded
            ]
        )

        # lane-indices can be of Any type (e.g. tuples) and, thus, cannot be converted directly.
        lane_indices_numpy = np.empty(len(lane_indices), dtype=object)
        lane_indices_numpy[:] = lane_indices

        return BorderPolygonsPadded(
            lane_indices=lane_indices_numpy,
            border_polygons_padded=border_polygons_padded,
        )

    @property
    def border_lines(self) -> List[SampledBoundary]:
        return self._border_lines

    @property
    def border_lines_numpy(self) -> SampledBoundariesPadded:
        return self._border_lines_numpy

    @property
    def border_polygons(self) -> Dict[str, List[np.ndarray]]:
        return self._border_polygons

    @property
    def border_polygons_numpy(self) -> BorderPolygonsPadded:
        return self._border_polygons_numpy

    def lane_index(self, pose: np.ndarray, margin=0.0) -> Any:
        """
        Lane association of the given pose.
        No association if |Frenet d| > |lane boundary d| + margin.
        """
        raise NotImplementedError

    def frenet_coordinates(
        self, pose: np.ndarray, margin=0.0
    ) -> Tuple[Any, float, float]:
        """
        Local Frenet coordinates of the given pose.
        No association if |Frenet d| > |lane boundary d| + margin.

        :return: Tuple (lane_index, s, d)
        """
        raise NotImplementedError

    def lane_info(self, lane_index: Any) -> LaneInfo:
        """
        Lane Information of the given lane index.
        """
        raise NotImplementedError

    def _extract_border_lines(
        self, sample_spacing: float = 10
    ) -> List[SampledBoundary]:
        """
        Extract border lines from road graph.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict):
        raise NotImplementedError

    def to_config(self) -> MapConfig:
        raise NotImplementedError
