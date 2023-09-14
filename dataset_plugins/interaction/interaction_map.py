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

import json
from typing import List

from dataset_plugins.interaction.interaction_globals import LIST_OF_LOCATIONS
from features_and_labels.map_base import MapBase, MapConfig, SampledBoundary
from lib.utils import get_path_to_class


class InteractionMap(MapBase):
    def __init__(
        self, path_to_interaction: str, path_to_parsed_map: str, location: str
    ):
        if location not in LIST_OF_LOCATIONS:
            raise ValueError(
                f"unknown location {location}. Map data is only available for the following locations {self._LIST_OF_LOCATIONS}"
            )
        self.location = location
        self._config = {
            "path_to_interaction": path_to_interaction,
            "path_to_parsed_map": path_to_parsed_map,
            "location": self.location,
        }
        with open(path_to_parsed_map, "r") as fp:
            map_data = json.load(fp)
        super().__init__(road_graph=map_data[location])

    def _extract_border_lines(
        self, sample_spacing: float = 10
    ) -> List[SampledBoundary]:
        sampled_boundaries = []
        for lane_id, lane in self._road_graph.items():
            for boundary in lane["boundaries"].values():
                sampled_boundaries.append(
                    SampledBoundary(
                        lane_id,
                        boundary["side"],
                        boundary["type"],
                        boundary["boundary_points"],
                    )
                )
        return sampled_boundaries

    @classmethod
    def from_config(cls, config: dict):
        if isinstance(config, dict):
            return InteractionMap(**config)
        return cls(
            path_to_interaction=config.config["path_to_interaction"],
            path_to_parsed_map=config.config["path_to_parsed_map"],
            location=config.config["location"],
        )

    def to_config(self) -> MapConfig:
        return MapConfig(
            path_to_class=get_path_to_class(self.__class__),
            config=self._config,
        )
