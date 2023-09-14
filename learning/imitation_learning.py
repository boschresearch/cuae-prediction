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
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from typing import Optional, Tuple

import mlflow
import numpy as np
import torch
from dacite import from_dict
from tqdm import tqdm

from features_and_labels.generators_base import MultiOutputGeneratorClassConfig
from learning.data_curator_config import DataCuratorClassConfig
from learning.learning_utils import LearningRateSchedulerConfig, scheduler_from_config
from learning.loss_utils import ExponentialMovingAverage
from learning.policies import Policy
from learning.policy_training_wrappers import (
    PolicyTrainingWrapper,
    PolicyTrainingWrapperClassConfig,
    get_matching_config_for_class,
    policy_training_wrapper_from_config,
)
from lib.data_handling import Dataset
from lib.random import init_global_seeds
from lib.utils import (
    current_time,
    ensure_init_type,
    git_diff,
    git_revision_hash,
    remove_prefix,
    remove_suffix,
)

ML_FLOW_MAX_TAG_LENGTH = 500
STATE_DICT = "state_dict"


@dataclass(kw_only=True)
class ILTrainingConfigBase:
    epochs: int
    learning_rate: float
    learning_rate_scheduler: dict
    weight_decay: float
    use_gpu: bool
    clip_grad_norm_to: Optional[float]
    loss_config: list
    additional_dev_loss_config: Optional[list]
    additional_train_loss_config: Optional[list]
    evaluate_dev_loss_every_nth_epoch: int = 1
    policy_training_wrapper_config: dict
    data_curator_config: DataCuratorClassConfig
    # Turning the store_state_dict_after_each_epoch to True does not/should not
    # change the default behavior of storing state dicts.
    store_state_dict_after_each_epoch: bool = False

    def __post_init__(self):
        self.data_curator_config = ensure_init_type(
            self.data_curator_config, DataCuratorClassConfig
        )

        # Dicts are a cleaner way to define losses,
        # because the key (name) is not a member of the loss itself.
        # TODO refactor towards dict
        if isinstance(self.loss_config, dict):
            self.loss_config = [
                self.loss_config[key] for key in sorted(self.loss_config.keys())
            ]
        if isinstance(self.additional_dev_loss_config, dict):
            self.additional_dev_loss_config = [
                self.additional_dev_loss_config[key]
                for key in sorted(self.additional_dev_loss_config.keys())
            ]
        if isinstance(self.additional_train_loss_config, dict):
            self.additional_train_loss_config = [
                self.additional_train_loss_config[key]
                for key in sorted(self.additional_train_loss_config.keys())
            ]


@dataclass
class ILTrainingConfig(ILTrainingConfigBase):
    experiment_name: str
    mlflow_directory: str
    policy_directory: str
    mlflow_run_name: str


def handle_training_configs(
    config: dict,
) -> Tuple[ILTrainingConfig, PolicyTrainingWrapperClassConfig]:
    """
    Update input and output entries based on dataset params.
    """
    cfg = config["training_config"]
    training_config = ILTrainingConfig(
        epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        learning_rate_scheduler=cfg["learning_rate_scheduler"],
        weight_decay=cfg["weight_decay"],
        use_gpu=cfg["use_gpu"],
        clip_grad_norm_to=cfg["clip_grad_norm_to"],
        loss_config=cfg["loss_config"],
        additional_dev_loss_config=cfg["additional_dev_loss_config"],
        additional_train_loss_config=cfg["additional_train_loss_config"],
        evaluate_dev_loss_every_nth_epoch=cfg["evaluate_dev_loss_every_nth_epoch"],
        policy_training_wrapper_config=cfg["policy_training_wrapper_config"],
        data_curator_config=cfg["data_curator_config"],
        experiment_name=cfg["experiment_name"],
        mlflow_directory=cfg["mlflow_directory"],
        policy_directory=cfg["policy_directory"],
        mlflow_run_name=cfg["mlflow_run_name"],
    )
    wrapper_config = training_config.policy_training_wrapper_config["config"]
    policy_config = config["policy_config"]

    # Dataset only loaded to infer feature and label properties
    dataset = Dataset(
        training_config.data_curator_config.config.dataset_paths[0].path,
        "train_0",
    )

    # TODO: needs refactoring
    # update feature infos
    def _infer_input_info(policy_input: dict, dataset: Dataset) -> dict:
        """
        Infers input info from dataset
        """

        policy_input["info"] = dataset.features_labels_infos[policy_input["key"]]

        return policy_input

    wrapper_config["params_from_dataset"]["inputs"] = [
        _infer_input_info(policy_input, dataset)
        if policy_input["info"] == "INFER"
        else policy_input
        for policy_input in wrapper_config["params_from_dataset"]["inputs"]
    ]

    # handle uni-modal labels naming, label shapes, multi-modal trajectory and distribution outputs
    def _infer_output_shape(policy_output: dict, dataset: Dataset) -> dict:
        """
        Infers output shape from dataset
        """
        name_unimodal = remove_prefix(
            policy_output["key"], "multimodal_"
        )  # returns name if doesn't start w/ prefix

        if name_unimodal in dataset.features_labels_names:
            shape = dataset.features_labels_infos[name_unimodal]["shape"]

            if policy_output["key"].startswith("multimodal_"):
                shape = [policy_config["config"]["num_modes"], *shape]

            policy_output["shape"] = shape
            return policy_output
        if policy_output["key"].startswith("multimodal_vae"):
            name_unimodal = remove_prefix(policy_output["key"], "multimodal_vae_")
            name_unimodal = remove_prefix(name_unimodal, "prior_")
            name_unimodal = remove_prefix(name_unimodal, "posterior_")
            name_unimodal = remove_prefix(name_unimodal, "mixture_")
            name_unimodal = remove_suffix(name_unimodal, "_std")
            if name_unimodal.endswith("_covar"):
                base_name = name_unimodal.removesuffix("_covar")
                waypoints_shape = dataset.features_labels_infos[base_name]["shape"]
                # add extra dimension to produce DxD covar matrix
                waypoints_shape.append(waypoints_shape[-1])
            else:
                waypoints_shape = dataset.features_labels_infos[name_unimodal]["shape"]
            policy_output["shape"] = [
                policy_config["config"]["num_z_samples"],
                policy_config["config"]["num_modes"],
                *waypoints_shape,
            ]
            return policy_output
        if policy_output["key"].startswith("posterior_tril") or policy_output[
            "key"
        ].startswith("prior_tril"):
            policy_output["shape"] = [policy_config["config"]["latent_z_dim"]] * 2
            return policy_output
        if policy_output["key"].startswith("posterior_mu") or policy_output[
            "key"
        ].startswith("prior_mu"):
            policy_output["shape"] = [
                policy_config["config"]["latent_z_dim"],
            ]
            return policy_output
        if policy_output["key"] == "mixture_weights":
            policy_output["shape"] = policy_config["config"]["num_components"]

        raise RuntimeError(
            f"Cannot infer output shape from dataset; output config: {policy_output}"
        )

    wrapper_config["params_from_dataset"]["outputs"] = [
        _infer_output_shape(policy_output, dataset)
        if policy_output["shape"] == "INFER"
        else policy_output
        for policy_output in wrapper_config["params_from_dataset"]["outputs"]
    ]

    # add feature sample times
    if "inputs_sample_times" in wrapper_config["params_from_dataset"]:
        if wrapper_config["params_from_dataset"]["inputs_sample_times"] == "INFER":
            wrapper_config["params_from_dataset"]["inputs_sample_times"] = []
            for policy_input in wrapper_config["params_from_dataset"]["inputs"]:
                wrapper_config["params_from_dataset"]["inputs_sample_times"].append(
                    dataset.features_labels_infos[policy_input["key"]]["details"][
                        "sample_time"
                    ]
                )

    # add label sample times
    if "outputs_sample_times" in wrapper_config["params_from_dataset"]:
        if wrapper_config["params_from_dataset"]["outputs_sample_times"] == "INFER":
            outputs_sample_times = []
            for policy_output in wrapper_config["params_from_dataset"]["outputs"]:
                name_unimodal = remove_prefix(policy_output["key"], "multimodal_")
                name_unimodal = remove_prefix(name_unimodal, "vae_")
                name_unimodal = remove_prefix(name_unimodal, "prior_")
                name_unimodal = remove_prefix(name_unimodal, "posterior_")
                if name_unimodal in dataset.features_labels_names:
                    outputs_sample_times.append(
                        dataset.features_labels_infos[name_unimodal]["details"][
                            "sample_time"
                        ]
                    )

            wrapper_config["params_from_dataset"][
                "outputs_sample_times"
            ] = outputs_sample_times

    # add dataset
    if "data_source" in wrapper_config["params_from_dataset"]:
        if wrapper_config["params_from_dataset"]["data_source"] == "INFER":
            wrapper_config["params_from_dataset"]["data_source"] = dataset.data_source

    # assemble PolicyTrainingWrapperClassConfig
    path_to_class = training_config.policy_training_wrapper_config["path_to_class"]
    policy_training_wrapper_class = get_matching_config_for_class(path_to_class)
    policy_training_wrapper_class_config = PolicyTrainingWrapperClassConfig(
        path_to_class=path_to_class,
        config=policy_training_wrapper_class(
            params_from_dataset=wrapper_config["params_from_dataset"],
            policy_config=policy_config,
            **{
                f.name: (
                    wrapper_config[f.name]
                    if f.name in wrapper_config
                    else getattr(training_config, f.name)
                )
                for f in fields(policy_training_wrapper_class)
                if f.name not in ["params_from_dataset", "policy_config"]
            },
        ),
    )

    return training_config, policy_training_wrapper_class_config


def extract_relevant_generator_configs(
    training_config: ILTrainingConfig, policy_training_wrapper: PolicyTrainingWrapper
) -> MultiOutputGeneratorClassConfig:
    """
    Extract relevant generator configs from dataset.
    """
    # get generator configs from dataset meta
    dataset_path = training_config.data_curator_config.config.dataset_paths[0].path
    with open(os.path.join(dataset_path, "meta.json")) as fp:
        dataset_meta = json.load(fp)
    features_labels_generator_config = MultiOutputGeneratorClassConfig(
        **dataset_meta["info"]["features_labels_generator_config"]
    )

    required_features_labels_names = (
        policy_training_wrapper.get_required_features_labels_names()
    )
    features_labels_generator_config.config.generators = [
        gen
        for gen in features_labels_generator_config.config.generators
        if gen.config.output_name in required_features_labels_names
    ]

    return features_labels_generator_config


def train_from_config(config: dict) -> None:
    if "random_seed" in config and config["random_seed"]:
        init_global_seeds(config["random_seed"])
    if (
        "use_deterministic_algorithms" in config
        and config["use_deterministic_algorithms"]
    ):
        torch.use_deterministic_algorithms(True)

    # TODO: update configs for diff-sim and prediction, then delete handle_training_configs
    if config["requires_legacy_config_handling"]:
        training_config, policy_training_wrapper_class_config = handle_training_configs(
            config
        )
    else:
        cfg = config["training_config"]
        training_config = ILTrainingConfig(
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            learning_rate_scheduler=cfg["learning_rate_scheduler"],
            weight_decay=cfg["weight_decay"],
            use_gpu=cfg["use_gpu"],
            clip_grad_norm_to=cfg["clip_grad_norm_to"],
            loss_config=cfg["loss_config"],
            additional_dev_loss_config=cfg["additional_dev_loss_config"],
            additional_train_loss_config=cfg["additional_train_loss_config"],
            evaluate_dev_loss_every_nth_epoch=cfg["evaluate_dev_loss_every_nth_epoch"],
            policy_training_wrapper_config=cfg["policy_training_wrapper_config"],
            data_curator_config=cfg["data_curator_config"],
            experiment_name=cfg["experiment_name"],
            mlflow_directory=cfg["mlflow_directory"],
            policy_directory=cfg["policy_directory"],
            mlflow_run_name=cfg["mlflow_run_name"],
        )
        policy_training_wrapper_class_config = PolicyTrainingWrapperClassConfig(
            **training_config.policy_training_wrapper_config
        )

    policy_training_wrapper = policy_training_wrapper_from_config(
        config=policy_training_wrapper_class_config,
    )
    policy_config = policy_training_wrapper.policy_config

    training_features_labels_generator_config = extract_relevant_generator_configs(
        training_config, policy_training_wrapper
    )
    inference_features_generator_config = (
        policy_training_wrapper.get_inference_features_generator_config(
            training_features_labels_generator_config
        )
    )

    os.makedirs(training_config.policy_directory, exist_ok=True)
    with open(os.path.join(training_config.policy_directory, "config.json"), "w") as fh:
        policy_config.path_to_state_dict = os.path.join(
            training_config.policy_directory, f"policy.{STATE_DICT}"
        )
        json.dump(
            {
                "training_config": asdict(training_config),
                "policy_config": asdict(policy_config),
                "training_features_labels_generator_config": asdict(
                    training_features_labels_generator_config
                ),
                "inference_features_generator_config": asdict(
                    inference_features_generator_config
                ),
                "meta_info": {
                    "date": current_time(),
                    "commit": git_revision_hash(),
                },
            },
            fh,
            indent=2,
        )

    train_with_mlflow_setup(
        config=training_config,
        policy_training_wrapper=policy_training_wrapper,
        path_to_state_dict=policy_config.path_to_state_dict,
    )


def train_with_mlflow_setup(
    config: ILTrainingConfig,
    policy_training_wrapper: PolicyTrainingWrapper,
    path_to_state_dict: Optional[str] = None,
) -> dict:
    # prepare output directories
    os.makedirs(config.policy_directory, exist_ok=True)
    os.makedirs(config.mlflow_directory, exist_ok=True)

    # set up mlflow
    mlflow.set_tracking_uri(os.path.join(config.mlflow_directory, "mlruns"))
    experiment_name = (
        config.experiment_name if config.experiment_name else "ImitationLearning"
    )
    mlflow.set_experiment(experiment_name)
    mlflow.end_run()  # End any active runs

    basic_config = from_dict(data_class=ILTrainingConfigBase, data=asdict(config))
    with mlflow.start_run(run_name=config.mlflow_run_name):
        startup_command = "python " + " ".join(sys.argv)
        mlflow.set_tag("command", startup_command[:ML_FLOW_MAX_TAG_LENGTH])
        with tempfile.TemporaryDirectory() as tmp_dir:
            git_diff_file = os.path.join(tmp_dir, "git_diff.txt")
            with open(git_diff_file, "w") as diff_file:
                diff_file.write(git_diff())
            mlflow.log_artifact(git_diff_file)
        mlflow.log_param(key="policy_directory", value=config.policy_directory)
        result = train(
            config=basic_config,
            policy_training_wrapper=policy_training_wrapper,
            path_to_state_dict=path_to_state_dict,
        )
    return result


def train(
    config: ILTrainingConfigBase,
    policy_training_wrapper: PolicyTrainingWrapper,
    path_to_state_dict: Optional[str] = None,
) -> dict:
    policy = policy_training_wrapper.policy

    mlflow.log_dict(asdict(config), "il_training_config.json")

    mlflow.log_param(key="policy_class", value=policy.__class__.__name__)
    mlflow.log_dict(asdict(policy_training_wrapper.policy_config), "policy_config.json")
    mlflow.log_param(
        key="policy_num_parameters",
        value=sum(p.numel() for p in policy.parameters() if p.requires_grad),
    )
    mlflow.log_dict(asdict(config), "training_config.json")

    policy.train()
    if config.use_gpu:
        policy = policy.cuda()

    mlflow.log_param(
        key="len_train_set",
        value=len(policy_training_wrapper.data_curator.train_set.dataset),
    )
    mlflow.log_param(
        key="len_dev_set",
        value=len(policy_training_wrapper.data_curator.dev_set.dataset),
    )

    dev_loss_ema = ExponentialMovingAverage(discount=0.8)

    optim = torch.optim.AdamW(
        policy.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler, scheduler_step = scheduler_from_config(
        config.learning_rate_scheduler,
        optim=optim,
        train_data_loader=policy_training_wrapper.data_curator.train_data_loader,
    )

    for epoch in tqdm(range(config.epochs), desc="Training epoch", position=0):
        # Training
        policy.train()
        grad_norm_list = []
        for batch in tqdm(
            policy_training_wrapper.data_curator.train_data_loader,
            desc="Training batch",
            position=1,
        ):
            grad_norm = _train_on_batch(
                policy_training_wrapper=policy_training_wrapper,
                batch=batch,
                optim=optim,
                clip_grad_norm_to=config.clip_grad_norm_to,
            )
            grad_norm_list.append(grad_norm)
            if scheduler_step == LearningRateSchedulerConfig.StepType.ITERATION:
                scheduler.step()

        (
            train_loss_value,
            train_loss_components,
        ) = policy_training_wrapper.train_loss.get_average_loss()
        mlflow.log_metric("train loss", train_loss_value, step=epoch)
        for loss_component_name, value in train_loss_components.items():
            mlflow.log_metric(
                "train loss component " + loss_component_name, value, step=epoch
            )
        if config.clip_grad_norm_to is not None:
            mlflow.log_metric("grad norm mean", np.mean(grad_norm_list), step=epoch)
            mlflow.log_metric("grad norm max", np.max(grad_norm_list), step=epoch)

        # Evaluation
        if (
            (epoch + 1) % config.evaluate_dev_loss_every_nth_epoch == 0
            or epoch == config.epochs - 1
        ):
            policy.eval()
            for batch in policy_training_wrapper.data_curator.dev_data_loader:
                policy_training_wrapper.validation_step(batch)

            (
                dev_loss_value,
                dev_loss_components,
            ) = policy_training_wrapper.dev_loss.get_average_loss()
            mlflow.log_metric("dev loss", dev_loss_value, step=epoch)
            for loss_component_name, value in dev_loss_components.items():
                mlflow.log_metric(
                    "dev loss component " + loss_component_name, value, step=epoch
                )

            current_dev_loss_ema = dev_loss_ema.update(dev_loss_value)
            mlflow.log_metric("dev loss EMA", current_dev_loss_ema, step=epoch)

        policy_training_wrapper.on_train_epoch_end(epoch=epoch)

        # Scheduler
        mlflow.log_metric(
            "lr", optim.param_groups[0]["lr"], step=epoch
        )  # get_last_lr is not available for all schedulers
        if scheduler_step == LearningRateSchedulerConfig.StepType.EPOCH:
            scheduler.step()
        elif scheduler_step == LearningRateSchedulerConfig.StepType.PLATEAU:
            scheduler.step(dev_loss_value)
        if path_to_state_dict is not None:
            policy.save_state_dict(path_to_state_dict)
            if config.store_state_dict_after_each_epoch:
                _save_model_state(
                    policy=policy,
                    path_to_state_dict=path_to_state_dict,
                    epoch=epoch,
                    extension=STATE_DICT,
                )

    return {
        "final_train_loss": train_loss_value,
        "final_dev_loss": dev_loss_value,
        "final_dev_loss_ema": current_dev_loss_ema,
    }


def _train_on_batch(
    policy_training_wrapper: PolicyTrainingWrapper,
    batch: torch.utils.data.Dataset,
    optim: torch.optim.Optimizer,
    clip_grad_norm_to: Optional[float] = None,
) -> None:
    loss_val = policy_training_wrapper.training_step(batch)

    optim.zero_grad()
    loss_val.backward()

    grad_norm = None
    if clip_grad_norm_to is not None:
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                policy_training_wrapper.policy.parameters(), max_norm=clip_grad_norm_to
            )
            .cpu()
            .numpy()
        )
    optim.step()

    return grad_norm


def _save_model_state(
    policy: Policy, path_to_state_dict: str, epoch: int, extension: str = STATE_DICT
) -> None:
    """
    Saves the state of a given policy model with a filename that includes the epoch number.
    """
    # Ensure the path doesn't already end with the file extension
    # Reason for that check is that there is noo garantee that path_to_state_dict
    # has the same extension.
    if path_to_state_dict.endswith(f".{extension}"):
        base_path = path_to_state_dict[: -len(f".{extension}")]
    else:
        base_path = path_to_state_dict

    # Save the state dictionary with the epoch number in the filename
    policy.save_state_dict(f"{base_path}_{epoch}.{extension}")
