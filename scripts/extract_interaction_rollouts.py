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
import os

from dataset_plugins.interaction.interaction_converter import (
    generate_rollouts_from_interaction,
)
from dataset_plugins.interaction.interaction_globals import LIST_OF_LOCATIONS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract rollouts from interaction data. Every rollout is cut around a dedicated pivot (ego) vehicle."
    )
    parser.add_argument(
        "--interaction",
        type=str,
        help="path to the interaction dataset",
        required=True,
    )
    parser.add_argument(
        "--parsed_map",
        type=str,
        help="path to the parsed map .json",
        required=True,
    )
    parser.add_argument(
        "--locations",
        type=str,
        nargs="+",
        help="list of locations separated by spaces; if not specified all tracks in given interaction folder are taken",
        required=False,
    )
    parser.add_argument(
        "--out",
        type=str,
        help="path to the directory where the rollouts will be stored",
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        help="number of workers for parallelized rollout generation",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--only_tracks_to_predict",
        type=bool,
        help="Generate rollouts only for objects with 'track_to_predict' in .csv set to 1",
        required=False,
        default=False,
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if len(os.listdir(args.out)):
        raise RuntimeError(f"out must be empty: {args.out}")

    v12_path = "interaction-dataset_v1.2"
    assert (
        v12_path in args.interaction
    ), f"Path {args.interaction} does not contain {v12_path}; INTERACTION version 1.2 must be used"

    if args.locations is None:
        locations = []
        loc_tracks_files = os.listdir(args.interaction)
        for track_file in loc_tracks_files:
            locations.append(track_file.replace(".csv", ""))
    else:
        locations = args.locations

    assert set(locations).issubset(
        set(LIST_OF_LOCATIONS)
    ), f"Locations: {locations} are not all among allowed locations: {LIST_OF_LOCATIONS}"

    for location_name in locations:
        # extract N rollouts from a single location
        rollouts = generate_rollouts_from_interaction(
            path_to_interaction=args.interaction,
            path_to_parsed_map=args.parsed_map,
            location_name=location_name,
            num_workers=args.num_workers,
        )

        # store rollouts to disk
        num_stored_rollouts = len(os.listdir(args.out))
        rollouts = [rollout for rollout in rollouts if len(rollout) > 0]
        for i, rollout in enumerate(rollouts):
            rollout.save(
                os.path.join(
                    args.out,
                    f"rollout_{i+num_stored_rollouts}.json",  # rollouts indexed from 0, i+num is ok
                )
            )
