# Conditional Unscented Autoencoder

This is the companion code for the paper [Conditional Unscented Autoencoders for Trajectory Prediction](https://arxiv.org/abs/2310.19944) by Faris Janjoš, Marcel Hallgarten, Anthony Knittel, Maxim Dolgov, Andreas Zell, J. Marius Zöllner. The code allows the users to reproduce and extend the results reported in the paper. Please cite the above paper when reporting, reproducing or extending the results. Recommended citation:
```
@article{janjovs2023conditional,
  title={Conditional Unscented Autoencoders for Trajectory Prediction},
  author={Janjo{\v{s}}, Faris and Hallgarten, Marcel and Knittel, Anthony and Dolgov, Maxim and Zell, Andreas and Z{\"o}llner, J Marius},
  journal={arXiv preprint arXiv:2310.19944},
  year={2023}
```

## Technical details

#### Installation

Create and activate the conda environment:
```
$ conda create -n cuae_env python=3.10
$ conda activate cuae_env
```

Set up the repository and install necessary packages:
```
$ git clone https://github.com/boschresearch/cuae-prediction
$ cd cuae-prediction
$ pip install -r requirements.txt
$ pip install -e .
```

#### INTERACTION data

Obtain the [INTERACTION dataset](http://interaction-dataset.com/) (make sure to use the v1.2 version) and store to a folder of your choosing: `path_to_interaction_raw_data`.

Generate the INTERACTION map polylines by:
```
$ python scripts/extract_maps.py --in_path path_to_interaction_raw_data/maps --out_path path_to_interaction_raw_data
```

#### Rollouts

Generate the INTERACTION training and validation rollouts.

Create an empty directory at a desired path `path_to_interaction_rollouts`.
Run the following:
```
$ python scripts/extract_interaction_rollouts.py --interaction path_to_interaction_raw_data/train --parsed_map path_to_interaction_dataset/parsed_map/parsed_interaction_map.json --out path_to_interaction_rollouts/train --num_workers 12
$ python scripts/extract_interaction_rollouts.py --interaction path_to_interaction_raw_data/val --parsed_map path_to_interaction_dataset/parsed_map/parsed_interaction_map.json --out path_to_interaction_rollouts/val --num_workers 12
```
Adjust the number of workers if desired.

#### Model features
Generate the pre-processed features for all the models.

Run the following while using the existing `path_to_interaction_rollouts` as the rollout location and setting `path_to_interaction_features` to the desired pre-processed features destination:
```
$ python scripts/dataset/hydra_generate_dataset_from_config.py --config-name prediction_dataset_gen_interaction_starnet_cvae dataset_config.config.rollouts_path_in=path_to_interaction_rollouts dataset_config.config.dataset_path_out=path_to_interaction_features dataset_config.config.num_workers=12 dataset_config.config.num_splits=12
```
Adjust the number of workers if desired.

#### Models

The INTERACTION models currently included in the repository are:
- CVAE
- CUAE
- CVAE+clusters
- CUAE+clusters
- CXP-CVAE+clusters
- CXP-CUAE+clusters
- GMM-CVAE
- GMM-CUAE

See Table I in the paper for an overview.

Note: the models trained on the CelebA dataset will be published in the [repository](https://github.com/boschresearch/unscented-autoencoder) of the [UAE predecessor model](https://arxiv.org/abs/2306.05256).

#### Training:

Use the following paths:
- `path_to_interaction_features`: location of pre-processed model features
- `path_to_mlflow`: desired location of MLFlow-stored experiment results
- `path_to_policy_output`: desired location of stored model parameters and weights

Run the following commands to train different models from Table I in the paper.

- CVAE (K=M=6):
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_cvae training_config.mlflow_run_name=cvae_k6m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cvae_k6m6 training_config.mlflow_directory=path_to_mlflow
```

- CVAE (K=M=65):
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_cvae training_config.mlflow_run_name=cvae_k65m65 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cvae_k65m65 training_config.mlflow_directory=path_to_mlflow policy_config.config.num_z_samples=65
```

- CUAE (K=M=6):
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_utcvae training_config.mlflow_run_name=cuae_k6m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cuae_k6m6 training_config.mlflow_directory=path_to_mlflow policy_config.config.output_samples_handling.method=averaging policy_config.config.num_z_samples=6 policy_config.config.latent_sampling.heuristic=random_pairs training_config.loss_config.utvae_loss.loss_function.config.reconstruction_loss_func_config.config.normal_dist_std=1.0
```

- CUAE (K=M=65):
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_utcvae training_config.mlflow_run_name=cuae_k65m65 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cuae_k65m65 training_config.mlflow_directory=path_to_mlflow policy_config.config.output_samples_handling.method=averaging  training_config.loss_config.utvae_loss.loss_function.config.reconstruction_loss_func_config.config.normal_dist_std=1.0
```

- CVAE+clusters (K=65, M=6)
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_cvae training_config.mlflow_run_name=cvae_clusters_k65m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cvae_clusters_k65m6 training_config.mlflow_directory=path_to_mlflow policy_config.config.output_samples_handling.method=averaging_with_clustering policy_config.config.output_samples_handling.path_to_clustering_func=learning.clustering_utils.kmeans_clustering policy_config.config.output_samples_handling.num_clusters=6 policy_config.config.num_z_samples=65 training_config.loss_config.vae_loss.loss_function.config.reconstruction_loss_func_config.config.normal_dist_std=regressed_std
```

- CUAE+clusters (K=65, M=6)
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_utcvae training_config.mlflow_run_name=cuae_clusters_k65m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cuae_clusters_k65m6 training_config.mlflow_directory=path_to_mlflow
```

- CXP-CVAE+clusters (K=65, M=6)
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_utcvae_expost training_config.mlflow_run_name=cxp_cvae_clusters_k65m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cxp_cvae_clusters_k65m6 training_config.mlflow_directory=path_to_mlflow policy_config.config.latent_sampling.method=random training_config.loss_config.utvae_loss.loss_function.config.reduce_sample_multimodality=mean
```

- CXP-CUAE+clusters (K=65, M=6)
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_utcvae_expost training_config.mlflow_run_name=cxp_cuae_clusters_k65m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/cxp_cuae_clusters_k65m6 training_config.mlflow_directory=path_to_mlflow
```

- GMM-CVAE (K=65, M=6)
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_cvae_gmm_prior training_config.mlflow_run_name=gmm_cvae_k65m6 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/gmm_cvae_k65m6 training_config.mlflow_directory=path_to_mlflow
```

- GMM-CUAE (K=65, M=6)
```
$ python scripts/training/hydra_train_from_config.py --config-name prediction_training_starnet_cvae_gmm_prior training_config.mlflow_run_name=gmm_cuae_k65m65 training_config.data_curator_config.config.dataset_paths="[{path: path_to_interaction_features}]" training_config.policy_directory=path_to_policy_output/gmm_cuae_k65m65 training_config.mlflow_directory=path_to_mlflow policy_config.config.latent_sampling.method=unscented policy_config.config.latent_sampling.heuristic=mean_random_pairs
```

## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication cited above. It will neither be maintained nor monitored in any way.

## License
Conditional Unscented Autoencoder is open-sourced under the AGPL-3.0 license. See the [LICENSE](LICENSE) file for details.
