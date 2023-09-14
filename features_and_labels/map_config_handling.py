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

from typing import Union

from features_and_labels.map_base import MapBase, MapConfig
from lib.utils import class_from_path, ensure_init_type


def map_from_config(map_config: Union[dict, MapConfig]) -> MapBase:
    map_config = ensure_init_type(map_config, MapConfig)
    cls = class_from_path(map_config.path_to_class)
    return cls.from_config(map_config.config)
