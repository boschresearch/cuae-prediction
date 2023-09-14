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
import json
import math
import os
import xml.etree.ElementTree as xml

import numpy as np
import pyproj
from tqdm import tqdm


class LL2XYProjector:
    def __init__(self, lat_origin=0.0, lon_origin=0.0):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = (
            math.floor((lon_origin + 180.0) / 6) + 1
        )  # works for most tiles, and for all in the dataset
        # self.zone = 19
        self.p = pyproj.Proj(proj="utm", ellps="WGS84", zone=self.zone, datum="WGS84")
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]


def _get_way_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            suptype = tag.get("v")
            if suptype not in ["line_thin", "line_thick"]:
                return suptype
            else:
                for tag in element.findall("tag"):
                    if tag.get("k") == "subtype":
                        return suptype + " " + tag.get("v")
    return None


def _parse_type(type):
    other = "solid"
    virtual = "solid"
    dict_of_types = {
        "line_thin dashed": "broken",
        "line_thick dashed": "broken",
        "curbstone": other,
        "line_thin solid": "solid",
        "line_thin solid_solid": "solid",
        "line_thick solid": "solid",
        "line_thick solid_solid": "solid",
        "bike_marking": other,
        "stop_line": other,
        "road_border": other,
        "guard_rail": other,
        "virtual": virtual,
        "pedestrian_marking": other,
        "traffic_sign": other,
    }
    return dict_of_types[type]


def _parse_nodes_to_dict(nodes) -> dict:
    projector = LL2XYProjector()
    dict_of_nodes = {}
    for node in nodes:
        x, y = projector.latlon2xy(float(node.get("lat")), float(node.get("lon")))
        dict_of_nodes.update(
            {
                node.get("id"): {
                    "x": x,
                    "y": y,
                }
            }
        )
    return dict_of_nodes


def _parse_boundaries_to_dict(boundaries, dict_of_nodes) -> dict:
    dict_of_boundaries = {}
    for boundary in boundaries:
        nodes = boundary.findall("nd")
        type = _parse_type(_get_way_type(boundary))
        dict_of_boundaries.update(
            {
                boundary.get("id"): {
                    "boundary_points": [
                        [
                            dict_of_nodes[node.get("ref")]["x"],
                            dict_of_nodes[node.get("ref")]["y"],
                        ]
                        for node in nodes
                    ],
                    "type": type,
                }
            }
        )
    return dict_of_boundaries


def _parse_speedlimits_to_dict(relations):
    dict_of_speed_limits = {}
    for relation in relations:
        tags = relation.findall("tag")
        tag_dict = {tag.get("k"): tag.get("v") for tag in tags}
        if "speed_limit" in tag_dict.values():
            speed_limit = tag_dict["sign_type"]
            if "kmh" in speed_limit:
                speed_limit = float(speed_limit.replace("kmh", "")) / 3.6
            elif "mph" in speed_limit:
                speed_limit = float(speed_limit.replace("mph", "")) / 2.24
            else:
                raise ValueError("unknown speed unit")

            dict_of_speed_limits.update({relation.get("id"): speed_limit})
    return dict_of_speed_limits


def _parse_lanes_to_dict(relations, dict_of_speed_limits, dict_of_boundaries):
    lanes = {}
    for relation in relations:
        tag_values = [tag.get("v") for tag in relation.findall("tag")]
        if "lanelet" not in tag_values:
            continue
        lane_id = relation.get("id")
        lanes.update(
            {
                lane_id: {
                    "speed_limit": None,
                    "boundaries": {},
                }
            }
        )
        members = relation.findall("member")
        for member in members:
            if member.get("type") == "relation":
                rel_id = member.get("ref")
                if rel_id in dict_of_speed_limits.keys():
                    lanes[lane_id]["speed_limit"] = dict_of_speed_limits[rel_id]
            elif member.get("type") == "way":
                boundary_id = member.get("ref")
                boundary = dict_of_boundaries[boundary_id].copy()
                boundary.update({"side": member.get("role")})
                lanes[lane_id]["boundaries"].update({boundary_id: boundary})
    return lanes


def _order_lane_boundary_points(lanes):
    for lane_id, lane in lanes.items():
        boundary_ids = list(lane["boundaries"].keys())
        # print(boundary_ids)
        first_boundary = lane["boundaries"][boundary_ids[0]]["boundary_points"]
        start1, end1 = np.array(first_boundary[0]), np.array(first_boundary[-1])

        second_boundary = lane["boundaries"][boundary_ids[1]]["boundary_points"]
        start2, end2 = np.array(second_boundary[0]), np.array(second_boundary[-1])

        dist_keep_orientation = np.linalg.norm((start1 - start2)) + np.linalg.norm(
            (end1 - end2)
        )
        dist_invert_orientation = np.linalg.norm((start1 - end2)) + np.linalg.norm(
            (end1 - start2)
        )

        if dist_invert_orientation < dist_keep_orientation:
            lanes[lane_id]["boundaries"][boundary_ids[1]]["boundary_points"] = lanes[
                lane_id
            ]["boundaries"][boundary_ids[1]]["boundary_points"][::-1]


def _load_interaction_map(path_to_map):
    osm_map = xml.parse(path_to_map).getroot()

    dict_of_nodes = _parse_nodes_to_dict(osm_map.findall("node"))
    dict_of_boundaries = _parse_boundaries_to_dict(
        osm_map.findall("way"), dict_of_nodes
    )
    dict_of_speed_limits = _parse_speedlimits_to_dict(osm_map.findall("relation"))

    lanes = _parse_lanes_to_dict(
        osm_map.findall("relation"), dict_of_speed_limits, dict_of_boundaries
    )
    _order_lane_boundary_points(lanes)
    return lanes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""
        Loads the INTERACTION map, extracts the polylines and stores them to a suitable .json format.
    """
    )
    parser.add_argument(
        "--in_path",
        type=str,
        help="path to the INTERACTION map directory",
        required=True,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="path where the generated .json file is stored",
        required=True,
    )
    args = parser.parse_args()

    in_dir = args.in_path
    map_files = [
        path
        for path in os.listdir(in_dir)
        if "_xy" not in path and "TestScenario" not in path and ".osm" in path
    ]
    interaction_map_data = {}

    for map_file in tqdm(map_files):
        map = _load_interaction_map(os.path.join(in_dir, map_file))
        interaction_map_data.update({map_file.replace(".osm", ""): map})
    poly_out_path = args.out_path + "/parsed_map/"
    os.mkdir(poly_out_path)
    with open(poly_out_path + "parsed_interaction_map.json", "w") as fh:
        json.dump(interaction_map_data, fh)

    print(list(interaction_map_data.keys()))
