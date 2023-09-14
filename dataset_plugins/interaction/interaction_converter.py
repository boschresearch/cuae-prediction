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

import os
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

from dataset_plugins.interaction.interaction_globals import (
    FRAME_RATE,
    PEDESTRIAN_CYCLIST_LENGTH_WIDTH,
)
from dataset_plugins.interaction.interaction_map import InteractionMap
from lib.frame import DynamicState, Frame, Meta, Obj, ObjectClasses
from lib.rollout import Rollout, StaticRolloutInfo, cut_rollout
from lib.utils import map_classes_to_rollouts_classes

interaction_to_rollout_classes_map = {
    ObjectClasses.passenger_car: {"car"},
    ObjectClasses.large_vehicle: {"bus", "truck"},
    ObjectClasses.bicycle: {"bicycle", "pedestrian/bicycle"},
    ObjectClasses.motorbike: {"motorcycle"},
    ObjectClasses.pedestrian: {"pedestrian"},
}

from lib.objectives import Objectives


def _create_rollout_from_pandas_frame(
    df: pd.DataFrame, map_info: InteractionMap, location_name: str
) -> Rollout:
    frame_ixs = np.sort(df["frame"].unique().astype(int))

    frames = [generate_frame(frame_ix, df) for frame_ix in frame_ixs]

    # get scene_id from to store in meta
    scene_id = df["scene_id"].unique().astype(int)
    assert len(scene_id) == 1

    static_info = StaticRolloutInfo(
        data_frequency=FRAME_RATE,
        map_info=map_info.to_config(),
        ego_id=None,  # setting ego_id does not make sense since agent ids in a .csv are non-unique; different scenes are distinguished by scene_id
        objective=Objectives.PREDICTION,
        meta={"location_name": location_name, "scene_id": scene_id[0]},
    )
    rollout = Rollout(
        static_info=static_info,
        frames=frames,
    )

    return rollout


def generate_rollouts_from_interaction(
    path_to_interaction: str,
    path_to_parsed_map: str,
    location_name: str,
    num_workers: int = 1,
    only_tracks_to_predict: bool = False,
) -> List[Rollout]:
    """
    Generates INTERACTION rollouts with ego information.
    """
    # extract N rollouts from a single location
    location_rollouts = generate_scene_rollouts_from_interaction(
        path_to_interaction=path_to_interaction,
        path_to_parsed_map=path_to_parsed_map,
        location_name=location_name,
        num_workers=num_workers,
    )

    # parallelize extraction of ego-centric rollouts from list of location-wise, 'global' rollouts
    lambda_make_ego_centric_rollouts = lambda x: make_ego_centric_rollouts(
        x, only_tracks_to_predict
    )

    with Pool(num_workers) as pool:
        loc_all_ego_centric_rollouts = list(
            tqdm(
                pool.imap(lambda_make_ego_centric_rollouts, location_rollouts),
                total=len(location_rollouts),
                desc="Extracting ego-centric rollouts...",
            )
        )

    # flatten list of lists of rollouts
    loc_all_ego_centric_rollouts = list(chain(*loc_all_ego_centric_rollouts))

    return loc_all_ego_centric_rollouts


def generate_scene_rollouts_from_interaction(
    path_to_interaction: str,
    path_to_parsed_map: str,
    location_name: str,
    num_workers: int = 1,
) -> List[Rollout]:
    """
    :return rollouts: a list of rollouts for the given location; a single .csv file contains disjoint scenes (on the same location) denoted by the same time-step values and agent id's, therefore, multiple rollouts without defining an ego are extracted.
    """
    map_info = InteractionMap(path_to_interaction, path_to_parsed_map, location_name)
    traj_dfs = read_trajectory_data(
        path_to_interaction, location_name
    )  # returns a list of scene data frames

    lambda_create_rollout_from_pandas_frame = (
        lambda x: _create_rollout_from_pandas_frame(x, map_info, location_name)
    )
    with Pool(num_workers) as pool:
        rollouts = list(
            tqdm(
                pool.imap(lambda_create_rollout_from_pandas_frame, traj_dfs),
                total=len(traj_dfs),
                desc="Extracting rollouts without a defined ego...",
            )
        )

    return rollouts


def generate_frame(frame_ix: int, traj_df: pd.DataFrame) -> Frame:
    df = get_rows_by_frame(traj_df, frame_ix)
    return Frame(
        timestamp=frame_ix / FRAME_RATE,
        objects=[
            Obj(
                dynamic_state=DynamicState(
                    x=df.loc[obj_loc]["x"],
                    y=df.loc[obj_loc]["y"],
                    yaw=df.loc[obj_loc]["yaw"],
                    v_x=df.loc[obj_loc]["v_x"],
                    v_y=df.loc[obj_loc]["v_y"],
                    a_x=df.loc[obj_loc]["a_x"],
                    a_y=df.loc[obj_loc]["a_y"],
                ),
                meta=Meta(
                    obj_class=map_classes_to_rollouts_classes(
                        interaction_to_rollout_classes_map, df.loc[obj_loc]["class"]
                    ),
                    obj_valid=True
                    if ("track_to_predict", 1) in df.loc[obj_loc].items()
                    else False,
                    obj_id=df.loc[obj_loc]["agent_id"],
                    obj_dimensions=[
                        df.loc[obj_loc]["length"],
                        df.loc[obj_loc]["width"],
                    ],
                    info={"scene_id": df.loc[obj_loc]["scene_id"].astype(int)},
                ),
            )
            for obj_loc in df.index
        ],
    )


def read_trajectory_data(
    path_to_interaction: str, location_name: str
) -> List[pd.DataFrame]:
    """
    :param path_to_interaction: path to the interaction data set data folder.
    :param location_name: name of location in the track file, e.g. DR_CHN_Merging_ZS0, see interaction_globals.py.
    :return scenes_df: list of dataframes of different traffic scenes for the given location
    """
    track_file = os.path.join(path_to_interaction, location_name + ".csv")
    tracks = pd.read_csv(track_file)
    out = tracks.copy()
    out.rename(
        columns={
            "frame_id": "frame",
            "psi_rad": "yaw",
            "vx": "v_x",
            "vy": "v_y",
            "track_id": "agent_id",
            "case_id": "scene_id",
            "agent_type": "class",
        },
        inplace=True,
    )

    # a_t = (v_t+1 -v_t)/dt  , 1/dt=FRAME_RATE
    out["a_x"] = (out["v_x"].diff() * FRAME_RATE).round(3)
    out["a_y"] = (out["v_y"].diff() * FRAME_RATE).round(3)

    # .diff() puts difference in row with larger index (a_t-1 is placed in row of v_t) -> shift one row up
    out["a_x"] = out["a_x"].shift(-1)
    out["a_y"] = out["a_y"].shift(-1)

    # replace empty pedestrian/bicycle values
    out.loc[out["class"] == "pedestrian/bicycle", "yaw"] = None
    out.loc[
        out["class"] == "pedestrian/bicycle", "length"
    ] = PEDESTRIAN_CYCLIST_LENGTH_WIDTH[0]
    out.loc[
        out["class"] == "pedestrian/bicycle", "width"
    ] = PEDESTRIAN_CYCLIST_LENGTH_WIDTH[1]

    # replace NaN values
    out[["a_x", "a_y", "yaw"]] = out[["a_x", "a_y", "yaw"]].fillna(value=0.0)

    # split different scenes in track_file since they share same timesteps
    scenes_df = []
    for i in out["scene_id"].unique():
        scenes_df.append(out[out["scene_id"] == i])

    return scenes_df


def get_rows_by_frame(pandas_frame: pd.DataFrame, frame: int) -> pd.DataFrame:
    return pandas_frame[pandas_frame["frame"] == frame]


def make_ego_centric_rollouts(
    loc_rollout: Rollout, only_tracks_to_predict: bool = False
) -> List[Rollout]:
    """
    Helper to extract list of ego-centric rollouts from a single 'global' rollout.
    """
    ego_obj_ids = loc_rollout.get_unique_object_ids(
        exclude_classes=[ObjectClasses.pedestrian, ObjectClasses.bicycle],
        exclude_invalid_obj=only_tracks_to_predict,
    )
    loc_ego_centric_rollouts = []
    for ego_id in ego_obj_ids:
        rollout = cut_rollout(loc_rollout, ego_id)
        map_info = InteractionMap.from_config(rollout.static_info.map_info)
        rollout.static_info.map_info = map_info.to_config()
        loc_ego_centric_rollouts.append(rollout)

    return loc_ego_centric_rollouts
