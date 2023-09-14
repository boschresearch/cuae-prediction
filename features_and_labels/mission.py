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

from features_and_labels.map_base import MapBase
from features_and_labels.map_config_handling import map_from_config
from lib.rollout import StaticRolloutInfo


class Mission:
    """
    Mission wraps a map.
    """

    def __init__(
        self,
        map_info: MapBase,
    ):
        """
        Args:
            - map_info
        """
        self._map_info = map_info

    @property
    def map_info(self) -> MapBase:
        return self._map_info


class DummyMission(Mission):
    """
    Reduced mission that only provides the map.
    """

    def __init__(self, map_info: MapBase):
        super().__init__(map_info=map_info)


def mission_from_static_rollout_info(static_rollout_info: StaticRolloutInfo):
    map_info = map_from_config(static_rollout_info.map_info)

    mission = DummyMission(map_info=map_info)

    return mission
