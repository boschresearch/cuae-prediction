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
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from features_and_labels.features_labels_generator import (
    FeaturesLabelsGenerator,
    FeaturesLabelsGeneratorConfig,
)
from features_and_labels.generators_base import MultiOutputGeneratorClassConfig
from lib.rollout import Rollout
from lib.utils import (
    create_empty_directory,
    ensure_init_type,
    git_revision_hash,
    to_python_standard,
)


@dataclass
class DatasetPathConfig:
    path: str
    sampling_weight: float = 1.0  # (see https://pytorch.org/docs/stable/data.html)
    name: Optional[str] = None


class Dataset(torch.utils.data.Dataset):
    """
    Dataset that contains meta.json, and hdf5 file with input features and labels.

    dataset_name is typically "train_x" or "dev_x". (x: int > 0)

    Downsampling (in range (0, 1]) allows to reduce the dataset size to this fraction.
    This is helpful if we want to extend a dataset with only a fraction of another one.
    """

    expected_dataset_meta_keys = [
        "features_labels_infos",
        "dataset_config",
        "data_frequency",
        "features_labels_generator_config",
    ]

    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
    ):
        with open(os.path.join(dataset_path, "meta.json")) as fp:
            _meta = json.load(fp)
            if any([k not in _meta["info"] for k in self.expected_dataset_meta_keys]):
                Warning(
                    f"Dataset meta info does not contain expected keys. Expected={self.expected_dataset_meta_keys}, received={_meta['info']}"
                )

            def _read_from_meta_info_else_none(meta: dict, key: str):
                return meta["info"][key] if key in _meta["info"].keys() else None

            self._features_labels_infos = _meta["info"]["features_labels_infos"]
            self._features_labels_names = list(self._features_labels_infos.keys())
            self._data_source = (
                _meta["info"]["dataset_config"]["config"]["data_source"]
                if "dataset_config" in _meta["info"].keys()
                else None
            )
            self._data_frequency = (
                _meta["data_frequency"]
                if "data_frequency" in _meta["info"].keys()
                else None
            )
            self._features_labels_generator_config = _read_from_meta_info_else_none(
                _meta, "features_labels_generator_config"
            )

        data_file = h5py.File(os.path.join(dataset_path, f"data_{dataset_name}.h5"))

        self._features_labels = {
            name: data_file[name] for name in self.features_labels_names
        }

        self.total_length = len(next(iter(self._features_labels.values())))

    @property
    def features_labels_names(self) -> List[str]:
        return self._features_labels_names

    @property
    def features_labels_infos(self) -> dict:
        return self._features_labels_infos

    @property
    def data_source(self) -> str:
        return self._data_source

    @property
    def data_frequency(self) -> float:
        return self._data_frequency

    @property
    def features_labels_generator_config(self) -> dict:
        return self._features_labels_generator_config

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int) -> tuple:
        # Do return a plain list of labels such that it can be batched
        return {
            key: copy.deepcopy(
                self._features_labels[key][idx]
                if not key.startswith("str::")
                else self._features_labels[key][idx].tolist()
            )
            for key in self.features_labels_names
        }


@dataclass
class DatasetGenerationAdditionalConfig:
    """
    Parameters specific to dataset generation.

    Attributes:
      rollouts_path_in: Path to the directory containing the rollouts.
      dataset_path_out: Path to the directory where the dataset will be stored.
      split: Partitioning ratios of data subsets, a dict<name, ratio>.
             If set to "from_rollout_subfolders", ratios are inferred from rollout subfolders.
      rollout_stride: Frame shift between two samples from the same rollout.
      num_splits: Number of parts (hdf5 files) in which the dataset will be split.
      num_workers: Number of workers to generate the dataset. Each worker generates a split of the dataset.
    """

    rollouts_path_in: str
    dataset_path_out: str
    split: Union[Dict[str, float], str]
    rollout_stride: int = 1
    data_source: str = "default"
    num_splits: int = 1
    num_workers: int = 1

    def __post_init__(self):
        if isinstance(self.split, dict):
            if not sum(self.split.values()) == 1:
                raise ValueError(
                    f"Given dataset split = {self.split} does not sum to 1."
                )
        elif self.split == "from_rollout_subfolders":
            pass
        else:
            raise ValueError(f"Unsupported dataset split = {self.split}")

        assert isinstance(self.rollouts_path_in, str)
        assert isinstance(self.dataset_path_out, str)
        assert isinstance(self.rollout_stride, int)
        assert isinstance(self.data_source, str)
        assert isinstance(self.num_splits, int)
        assert isinstance(self.num_workers, int)


@dataclass
class DatasetGenerationAdditionalClassConfig:
    path_to_class: str
    config: DatasetGenerationAdditionalConfig

    def __post_init__(self):
        if isinstance(self.config, dict):
            self.config = DatasetGenerationAdditionalConfig(**self.config)


@dataclass
class DatasetGenerationConfig:
    dataset_config: DatasetGenerationAdditionalClassConfig
    features_labels_generator_config: MultiOutputGeneratorClassConfig

    def __post_init__(self):
        self.dataset_config = ensure_init_type(
            self.dataset_config, DatasetGenerationAdditionalClassConfig
        )
        self.features_labels_generator_config = ensure_init_type(
            self.features_labels_generator_config, MultiOutputGeneratorClassConfig
        )


def generate_dataset_from_rollouts(
    config: DatasetGenerationConfig,
) -> None:
    """
    Transform a set of rollout json's into single input features and labels, which is supported by Dataset.

    The rollouts are split into separate sets according to the split argument, e.g. {"train": 0.8, "dev": 0.2}, or directly inferred from train/dev subfolders, and separate datasets are created.
    """
    dataset_path_out = config.dataset_config.config.dataset_path_out
    create_empty_directory(
        path=dataset_path_out, exist_ok=True, ignore_hydra_subdir=True
    )

    assert (
        config.dataset_config.config.num_splits
        >= config.dataset_config.config.num_workers
    ), "Having more workers than splits does not lead to faster dataset generation!"

    start_time = datetime.now()

    feature_label_gen_conf = FeaturesLabelsGeneratorConfig(
        features_labels_generator_config=config.features_labels_generator_config,
    )

    features_labels_gen = FeaturesLabelsGenerator.from_config(feature_label_gen_conf)
    dataset_config = config.dataset_config

    # split rollouts into subsets
    rollout_subsets = dict()
    rollouts_path_in = config.dataset_config.config.rollouts_path_in
    if dataset_config.config.split == "from_rollout_subfolders":
        # get train/dev subset from rollouts subfolders <- for fixed, 'early' split
        for subset in ["train", "dev"]:
            assert subset in os.listdir(
                rollouts_path_in
            )  # must be a subfolder in rollouts
            all_subset_files = [
                os.path.join(rollouts_path_in, subset, f)
                for f in os.listdir(os.path.join(rollouts_path_in, subset))
                if f != "meta.json"
            ]
            np.random.shuffle(all_subset_files)
            rollout_subsets[subset] = all_subset_files
    else:
        # Dict[str, float]
        all_rollout_files = [
            os.path.join(rollouts_path_in, f)
            for f in os.listdir(rollouts_path_in)
            if f != "meta.json" and os.path.isfile(os.path.join(rollouts_path_in, f))
        ]

        np.random.shuffle(all_rollout_files)
        split_acc = 0
        for dataset_name, split_value in dataset_config.config.split.items():
            len_subset = max(int(split_value * len(all_rollout_files)), 1)
            rollout_subsets[dataset_name] = all_rollout_files[
                split_acc : split_acc + len_subset
            ]
            split_acc += len_subset

    # infer meta data from a batch; trim rollout to a single sample for features and labels
    rollouts_paths_subset = next(iter(rollout_subsets.values()))
    for rollout_path in rollouts_paths_subset:
        sample_rollout = Rollout.from_file(rollout_path)
        if (
            len(sample_rollout.frames)
            > features_labels_gen.len_head + features_labels_gen.len_tail
        ):
            break

        print(
            f"Rollout too short for features/labels generation => skip (file = {rollout_path})"
        )
        if rollout_path == rollouts_paths_subset[-1]:
            raise ValueError("No rollout with sufficient length found.")

    sample_rollout.frames = sample_rollout.frames[
        : features_labels_gen.len_head + features_labels_gen.len_tail + 1
    ]
    _ = features_labels_gen(
        sample_rollout,
        dataset_config.config.rollout_stride,
    )
    features_labels_infos = features_labels_gen.features_labels_infos

    # add sample time to each feature/label info type
    data_frequency = sample_rollout.static_info.data_frequency
    for key, gen in zip(
        features_labels_infos, features_labels_gen.features_labels_generator.generators
    ):
        features_labels_infos[key].details["sample_time"] = gen.stride / data_frequency

    process_results = []

    # torch.multiprocessing.set_start_method("spawn", force=True)
    with torch.multiprocessing.Pool(config.dataset_config.config.num_workers) as pool:
        for split_idx in range(config.dataset_config.config.num_splits):
            split_subset = {
                dataset_name: rollout_files[
                    split_idx :: config.dataset_config.config.num_splits
                ]
                for dataset_name, rollout_files in rollout_subsets.items()
            }

            process_result = pool.apply_async(
                gen_dataset_split,
                kwds=dict(
                    split_idx=split_idx,
                    dataset_path_out=dataset_path_out,
                    rollout_split=split_subset,
                    features_labels_shapes={
                        k: v.shape for k, v in features_labels_infos.items()
                    },
                    features_labels_gen=features_labels_gen,
                    rollout_stride=dataset_config.config.rollout_stride,
                ),
            )
            process_results.append(process_result)

        created_data_files = []

        pool.close()
        pool.join()

    for process_result in process_results:
        created_data_files.extend(process_result.get())

    cast_config: dict = to_python_standard(config, dataclass_to_dict=True)

    # write metadata to .json
    dataset_meta = {
        "info": {
            **cast_config,
            "features_labels_infos": to_python_standard(
                features_labels_infos, dataclass_to_dict=True
            ),
            "commit": git_revision_hash(),
            "rollouts": os.path.abspath(rollouts_path_in),
        },
        "data": created_data_files,
        "data_frequency": data_frequency,
    }
    dataset_meta["info"]["runtime"] = (datetime.now() - start_time).total_seconds()
    with open(os.path.join(dataset_path_out, "meta.json"), "w") as fp:
        json.dump(dataset_meta, fp, indent=2)


def gen_dataset_split(
    split_idx: int,
    dataset_path_out: str,
    rollout_split: Dict[str, List[str]],
    features_labels_shapes: Dict[str, tuple],
    features_labels_gen: Callable,
    rollout_stride: int,
) -> List[str]:
    """
    Generates a hdf5-dataset/ file for each element in the rollout_split
    :param: split_idx: index of the split
    :param: rollout_split: Dict with dataset-name (key) and list of rollout-file-paths (values)
    :param: features_labels_shapes: Dict with features and labels names (key) and the corresponding shapes (values) to be generated
    :param: Callable that generates the features/ labels from the rollouts
    :param: rollout_stride: frames shift between two samples
    :return: List of generated hdf5 files e.g ["data_train_5.h5", "data_dev_5.h5"]
    """
    created_data_files = []
    # write .h5 contents
    for dataset_name, rollout_files in rollout_split.items():
        data_file = f"data_{dataset_name}_{split_idx}.h5"
        created_data_files.append(data_file)

        with h5py.File(os.path.join(dataset_path_out, data_file), "w") as data_file:
            features_labels_dataset = create_h5py_dataset(
                data_file,
                data_shapes=features_labels_shapes,
            )

            row_count = 0

            for rollout_file in tqdm(
                rollout_files,
                desc=f"Processed {dataset_name} rollouts of split {split_idx}",
                position=split_idx,
            ):
                tqdm.write(f"Processing file: {rollout_file}")
                rollout = Rollout.from_file(rollout_file)
                if (
                    len(rollout.frames)
                    < features_labels_gen.len_head + features_labels_gen.len_tail + 1
                ):
                    print(
                        f"Rollout too short for features/labels generation => skip (file = {rollout_file})"
                    )
                    continue

                features_labels_ = features_labels_gen(rollout, rollout_stride)

                # skip if rollout yields empty features/labels
                if not (features_labels_.keys()):
                    continue

                assert (
                    features_labels_dataset.keys() == features_labels_.keys()
                ), f"Expected features and labels keys: {features_labels_dataset.keys()}, received features and labels keys: {features_labels_.keys()}"
                row_count = dump_to_h5py(
                    features_labels_dataset, features_labels_, row_count
                )

    return created_data_files


def create_h5py_dataset(
    data_file: h5py.File, data_shapes: Dict[str, tuple]
) -> Dict[str, h5py.Dataset]:
    """
    Creates a dataset for features and labels (data) in the provided h5py file
    """
    return {
        dataset_name: data_file.create_dataset(
            dataset_name,
            shape=(0,) + data_shape,
            maxshape=(None,) + data_shape,
            chunks=(1,) + data_shape,
            compression="gzip",
            dtype=np.float32
            if not dataset_name.startswith("str::")
            else h5py.special_dtype(vlen=str),
        )
        for dataset_name, data_shape in data_shapes.items()
    }


def dump_to_h5py(
    datasets: Dict[str, h5py.Dataset], data: Dict[str, np.ndarray], row_count: int
) -> int:
    """
    Write features and labels into the corresponding dataset
    """
    for dataset_name, new_data in data.items():
        assert datasets[dataset_name].shape[0] == row_count
        datasets[dataset_name].resize(row_count + len(new_data), axis=0)
        if not dataset_name.startswith("str::"):
            new_data = new_data.astype(np.float32)
        datasets[dataset_name][row_count:] = new_data
    return row_count + len(new_data)
