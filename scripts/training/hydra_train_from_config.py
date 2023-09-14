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

import hydra
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

from learning.imitation_learning import train_from_config
from lib.hydra_utils import PATH_TO_HYDRA_CONFIGS, load_dataset_info


@hydra.main(
    version_base=None,
    config_path=PATH_TO_HYDRA_CONFIGS,
    config_name="default_training",
)
def hydra_train_from_config(cfg) -> None:
    cfg = load_dataset_info(cfg)

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("list_conf_eval", lambda a: ListConfig(eval(a)))
    OmegaConf.resolve(cfg)
    train_from_config(OmegaConf.to_container(cfg, throw_on_missing=True))


if __name__ == "__main__":
    hydra_train_from_config()
