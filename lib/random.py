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

import random

import numpy as np
import torch

# Helper functions to guarantee reproducibility (https://pytorch.org/docs/stable/notes/randomness.html).
# NOTE: all top-level files including torch additionally need to set torch.use_deterministic_algorithms(True),
# which requires export CUBLAS_WORKSPACE_CONFIG=:4096:8
# TODO: this was tested and verified for training a policy, there might be other top-level
# scripts where reproducibility is not yet guaranteed (e.g. open-loop eval).


def init_global_seeds(seed: int) -> None:
    """
    Set seeds for all sources of randomness.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id: int) -> None:
    """
    Torch correctly sets (different, yet fixed) seeds per
    worker, but does not initialize external libraries ->
    do this manually.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
