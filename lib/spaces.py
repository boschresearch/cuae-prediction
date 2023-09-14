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
from gym import spaces


class Box(spaces.Box):
    """
    A class that extends spaces.Box so that dimensions can be prepended, i.e.
        b = Box(shape=(M, N))
        b.prepend_dims((K, L))
        b.shape == (K, L, M, N)
    """

    def __init__(self, low, high, shape=None, dtype=np.float32, seed=None):
        assert seed is None, "This wrapper does not support a seed that is not None"
        super().__init__(low, high, shape, dtype, seed)

    def prepend_dims(self, shape: tuple):
        shape = tuple(shape)
        new_shape = (*shape, *self.shape)
        new_high = np.broadcast_to(self.high, new_shape)
        new_low = np.broadcast_to(self.low, new_shape)
        super().__init__(new_low, new_high, new_shape, self.dtype)
