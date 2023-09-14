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

import argparse
import dataclasses
import enum
import importlib
import json
import os
import subprocess
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Type, TypeVar, Union

import numpy as np
import torch

AnyType = TypeVar("AnyType")  # Declare type variable


class ConfigType(enum.Enum):
    """
    Holds the possible locations for configuration files
    """

    DATASET = "dataset_configs"
    EVALUATION = "evaluation_configs"
    TRAINING = "training_configs"


def to_python_standard(
    arg: Union[dict, list, np.ndarray], dataclass_to_dict: bool = False
) -> Union[list, dict]:
    """
    Casts a nested dict or array of np.ndarrays to python standard data types.
    :param arg: object to cast to python standard
    :param dataclass_to_dict: whether or not to convert dataclasses to dict
    :return: nested list or dict of python standard data types
    """
    if isinstance(arg, torch.Tensor):
        arg = arg.detach().cpu().numpy()
    if isinstance(arg, dict):
        return {
            key: to_python_standard(value, dataclass_to_dict=dataclass_to_dict)
            for key, value in arg.items()
        }
    elif isinstance(arg, np.ndarray):
        return to_python_standard(arg.tolist(), dataclass_to_dict=dataclass_to_dict)
    elif isinstance(arg, list):
        return [
            to_python_standard(elem, dataclass_to_dict=dataclass_to_dict)
            for elem in arg
        ]
    elif isinstance(arg, np.bool_):
        return bool(arg)
    elif dataclass_to_dict and dataclasses.is_dataclass(arg):
        return to_python_standard(dataclasses.asdict(arg))
    elif isinstance(arg, np.int64):
        return int(arg)
    else:
        return arg


def ensure_init_type(obj: AnyType, cls: Type[AnyType]) -> AnyType:
    """
    Casts dict obj to 'cls' type.

    Args:
        obj: object to be cast
        cls: type of the class that's being cast

    Returns:
        obj of type cls
    """
    if isinstance(obj, cls):
        return obj
    elif isinstance(obj, dict):
        try:
            obj = to_python_standard(obj)
            if hasattr(cls, "from_dict"):
                return cls.from_dict(obj)
            else:
                return cls(**obj)
        except TypeError:
            raise TypeError(
                f"Unexpected keyword arguments in {obj} dict to {cls} casting."
            )
    else:
        raise RuntimeError(
            f"Unexpected initialization type for {obj}, dict or {cls} expected."
        )


def current_time() -> str:
    """
    Get current time as string in the format 'Year-Month-Day, Hours:Minutes:Seconds'
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_root_dir() -> str:
    print("__file__: ", __file__)
    directory_path = os.path.abspath(os.path.join(__file__, "..", ".."))

    if not os.path.isdir(directory_path):
        raise ValueError(f'The directory "{directory_path}" does not exist.')

    print("directory_path: ", directory_path)
    return directory_path


def config_from_file(absolute_file_path: str) -> dict:
    """
    Parse config file (json)

    :param absolute_file_path: absolute path to the config file
    :return: loaded json dict
    """
    if not os.path.exists(absolute_file_path):
        raise ValueError(f"File '{absolute_file_path}' does not exist")

    with open(absolute_file_path, "r") as f:
        config = json.load(f)
    return config


def git_revision_hash() -> str:
    """
    Get current commit hash if available
    """
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            .decode("utf-8")
            .replace("\n", "")
        )
    except subprocess.CalledProcessError:
        git_hash = ""
    return git_hash


def git_diff() -> str:
    """
    Returns git diff.
    """
    try:
        git_diff = subprocess.check_output(["git", "diff"]).decode("utf-8")
    except subprocess.CalledProcessError:
        git_diff = ""
    return git_diff


# TODO: make function name more general since it can be used to convert strings to function calls as well
def class_from_path(path: str) -> Callable:
    return class_from_path_with_legacy_map_support(path=path)


def instantiate_from_config(config: Any) -> Any:
    """
    Creates and object from the config.
    Config must be a dataclass with properties
        * path_to_class
        * config
    """
    cls = class_from_path(config.path_to_class)
    return cls.from_config(config.config)


def class_from_path_with_legacy_map_support(path: str) -> Callable:
    if path == "highway_env_plugin.env.highway_env_map_info.HighwayEnvMapInfo":
        warnings.warn(
            "Deprecated map class <highway_env_plugin.env.highway_env_map_info.HighwayEnvMapInfo> is used. Please substitute with <highway_env_plugin.env.highway_env_map.HighwayEnvMapWithRouting>"
        )
        path = "highway_env_plugin.env.highway_env_map.HighwayEnvMapWithRouting"

    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def get_path_to_class(cls) -> str:
    return cls.__module__ + "." + cls.__qualname__


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def remove_keys_from_dict(dict: dict, keys: List[Union[str, int, bool, float]]) -> dict:
    """
    Returns a copy of the dict without the specified keys
    """
    new_dict = dict.copy()
    for key in keys:
        new_dict.pop(key)
    return new_dict


def get_script_dir_path() -> str:
    """
    Returns absolute path of script directory
    """
    abs_script_dir_path = os.path.abspath(os.path.join(get_root_dir(), "scripts"))
    if not os.path.isdir(abs_script_dir_path):
        raise ValueError(f'The script dir "{abs_script_dir_path}" does not exist.')
    return abs_script_dir_path


def pprint_args(args: argparse.Namespace):
    print(" ".join(f"{key}={value}" for key, value in vars(args).items()))


def raise_if_dir_does_not_exist(
    dir_path: str,
):
    if not os.path.isdir(dir_path):
        msg = f'The dir "{dir_path}" does not exist and needs to be created beforehand.'
        raise ValueError(msg)


def raise_if_dir_is_not_empty(dir_path: str):
    if len(os.listdir(dir_path)) != 0:
        msg = f'The dir "{dir_path}" needs to be empty. It contains these files: {os.listdir(dir_path)}'
        raise ValueError(msg)


def create_eval_dir_with_timestamp(base_dir: str) -> str:
    eval_dir_with_timestamp_path = os.path.join(
        base_dir, "evaluation_" + current_time()
    )
    os.makedirs(eval_dir_with_timestamp_path, exist_ok=True)
    return eval_dir_with_timestamp_path


def map_classes_to_rollouts_classes(class_map: Dict[str, set], obj_class: str) -> str:
    """
    Maps the object class <obj_class> to the corresponding class in <class_map>.
    Class map is a dict of type
    object_class: set_of_mapped_classes
    """
    for k, v in class_map.items():
        if obj_class in v:
            return k
    raise RuntimeError(f"Unhandled class mapping for {obj_class}")


def pad_to_shape(array: np.ndarray, desired_shape: tuple, padder: object = np.nan):
    """
    Pads an array to the desired shape. padder is always added to upper axis side
        E.g. padder=0,  desired_shape=(3, 3),   array=[[1, 1], [1, 1]]
          -> padded_array = [[1, 1, 0],
                            [1, 1, 0],
                            [0, 0, 0]]
    padder: padding value
    """
    pad_widths = [
        (0, desired_shape[axis] - array.shape[axis])
        for axis in range(len(desired_shape))
    ]
    return np.pad(array, pad_widths, mode="constant", constant_values=padder)


def create_empty_directory(
    path: str, exist_ok: bool = False, ignore_hydra_subdir: bool = False
) -> None:
    """
    Similar to os.makedirs, but an error is thrown if the directory is not empty.

    Args:
        path: path of the directory to be created
        exist_ok: if True, do not throw an error if the directory already exists
        ignore_hydra_subdir: if True, do not throw an error if there is a subdir named "hydra"
    """
    os.makedirs(path, exist_ok=exist_ok)

    allowed_subdirs = set()
    if ignore_hydra_subdir:
        allowed_subdirs.add("hydra")

    if not set(os.listdir(path)) <= allowed_subdirs:
        raise ValueError(
            f"Attempting to create directory {path}, which is not empty: {os.listdir(path)}"
        )
