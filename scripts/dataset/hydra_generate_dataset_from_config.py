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

"""
This script can be used to generate a dataset consisting of input features and labels from rollouts in a directory.
"""

import hydra
from omegaconf import OmegaConf

from lib.data_handling import DatasetGenerationConfig, generate_dataset_from_rollouts
from lib.hydra_utils import PATH_TO_HYDRA_CONFIGS, instantiate_dataclass_from_omegaconf


@hydra.main(
    version_base=None,
    config_path=PATH_TO_HYDRA_CONFIGS,
    config_name="default_dataset_generation",
)
def hydra_generate_dataset_from_config(cfg) -> None:
    OmegaConf.resolve(cfg)
    config = instantiate_dataclass_from_omegaconf(cfg=cfg, cls=DatasetGenerationConfig)
    generate_dataset_from_rollouts(
        config=config,
    )


if __name__ == "__main__":
    hydra_generate_dataset_from_config()
