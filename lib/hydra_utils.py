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
import os
from typing import Type, TypeVar

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from lib.utils import get_root_dir

PATH_TO_HYDRA_CONFIGS = os.path.join(get_root_dir(), "hydra_configs")

T = TypeVar("T")  # Declare type variable


def load_dataset_info(cfg: DictConfig) -> DictConfig:
    """
    Replace cfg.dataset_info which is given by a json filename
    by the contents of that file (only info key).

    Args:
        cfg: the config

    Returns:
        the modified config
    """
    with open(cfg.dataset_info) as fh:
        dataset_info = json.load(fh)
    cfg.dataset_info = OmegaConf.create(dataset_info).info
    return cfg


def instantiate_dataclass_from_omegaconf(
    cfg: DictConfig, cls: Type[T], blacklist_keys: list = ["custom_help"]
) -> T:
    """
    Hydra produces an omegaconf-type config, convert to dataclass here.

    Args:
        cfg: The config produced by hydra.
        cls: The dataclass type that is returned.
        blacklist_keys: A list of all top-level keys that are ignored.

    Returns:
        The initialised dataclass.
    """
    config_dict = {
        k: v
        for k, v in OmegaConf.to_container(cfg, throw_on_missing=True).items()
        if k not in blacklist_keys
    }
    return cls(**config_dict)
