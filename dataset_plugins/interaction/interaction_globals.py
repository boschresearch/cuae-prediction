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
Example of dataset folder structure:

    |_ parsed_map
        |_ parsed_interaction_map.json
    |_ test_single-agent
        |_ DR_CHN_Merging_ZS0.csv
           ...
    |_ test_conditional-single-agent
        |_ DR_CHN_Merging_ZS0.csv
        ...
    |_ train
        |_ DR_CHN_Merging_ZS0.csv
           ...
    |_ val
        |_ DR_CHN_Merging_ZS0.csv
           ...

"""

PEDESTRIAN_CYCLIST_LENGTH_WIDTH = (1, 1)
LIST_OF_LOCATIONS = [
    "DR_CHN_Merging_ZS0",
    "DR_CHN_Merging_ZS2",
    "DR_CHN_Roundabout_LN",
    "DR_DEU_Merging_MT",
    "DR_DEU_Roundabout_OF",
    "DR_Intersection_CM",
    "DR_LaneChange_ET0",
    "DR_LaneChange_ET1",
    "DR_Merging_TR0",
    "DR_Merging_TR1",
    "DR_Roundabout_RW",
    "DR_USA_Intersection_EP0",
    "DR_USA_Intersection_EP1",
    "DR_USA_Intersection_GL",
    "DR_USA_Intersection_MA",
    "DR_USA_Roundabout_EP",
    "DR_USA_Roundabout_FT",
    "DR_USA_Roundabout_SR",
]
FRAME_RATE = 10
