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

import numpy as np
import torch

# maximum absolute accelerations observed in a dataset
DEFAULT_MAX_ABS_ACC = 5.0
IND_MAX_ABS_ACC = 6.53624
INTERACTION_MAX_ACCELERATION = 18.45537813386247

# maximum assumed steering angle for all datasets
DEFAULT_MAX_STEER = np.pi / 4


class ActionConstraints:
    def __init__(self, dataset: str = "default"):
        self.acc_constraints = {
            "default": DEFAULT_MAX_ABS_ACC,
            "ind": IND_MAX_ABS_ACC,
            "interaction": INTERACTION_MAX_ACCELERATION,
        }
        self.steer_constraints = {
            "default": DEFAULT_MAX_STEER,
            "ind": DEFAULT_MAX_STEER,
            "interaction": DEFAULT_MAX_STEER,
        }

        self.max_abs_acc = self.acc_constraints[dataset]
        self.max_abs_steer = self.steer_constraints[dataset]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multiplies the values in x with action constraints
        :param x: action features with values in [-1, 1] range, shape [batch_size, num_actions, 2]
        :return x: scaled actions, shape [batch_size, num_actions, 2]
        """
        # ensure in range [-1,1]
        constr = torch.ones_like(x)
        assert torch.all(torch.ge(x, -constr)) and torch.all(torch.le(x, constr))

        constr[:, :, 0] = self.max_abs_acc
        constr[:, :, 1] = self.max_abs_steer

        return x * constr  # elem-wise
