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

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from learning.policies import Policy, PolicyPin
from learning.policies_base import (
    FlexibleGAT,
    MultiHeadAttentionBlock,
    MultiHeadAttentionBlockConfig,
)
from lib.graph_utils import GraphAggregator, StarGraphEdgeGenerator
from lib.utils import class_from_path, ensure_init_type


@dataclass
class MLPEncoderConfig:
    feat_dim: int = 128
    num_input_channels: int = 2


class MLPEncoder(nn.Module):
    """
    Dummy network which encodes a sequence of num_input_channels elements to a feat_dim-dimensional feature vector.
    """

    def __init__(self, feat_dim, num_input_channels=2):
        super(MLPEncoder, self).__init__()

        self.linear = nn.Linear(num_input_channels, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input features [batch_size, num_timesteps, dim]
        Returns:
            output features [batch_size, feat_dim]
        """
        return self.linear(x[:, -1, :])

    @classmethod
    def from_config(cls, cf: dict):
        config = MLPEncoderConfig(**cf)
        return cls(
            feat_dim=config.feat_dim, num_input_channels=config.num_input_channels
        )


class BatchedStarNetAttentionBlock(torch.nn.Module):
    """
    Applies multiple Multi-Head Attention blocks onto the input node features.
        Note:
            Implementation works with varying num_nodes per sample
            - Attention mask is used to replace the required padding mask
    Examples:
        batched_mha_blocks_config example:
        [
            {
                lin_layers: [
                    {
                      "dim": 128,
                      "activation": "torch.nn.Tanh",
                      "norm": "torch.nn.BatchNorm1d"
                    },
                    {
                      "dim": 64,
                      "activation": "torch.nn.Tanh",
                      "norm": "torch.nn.BatchNorm1d"
                    }
                ]
                embed_dim: 64
                num_heads: 8
                norm_layer: "torch.nn.LayerNorm"
            },
            {
                lin_layers: []
                embed_dim: 64
                num_heads: 8
                norm_layer: "torch.nn.LayerNorm"
            },
        ]

    """

    def __init__(
        self,
        batched_mha_blocks_config: List[Union[Dict, MultiHeadAttentionBlockConfig]],
    ):
        super(BatchedStarNetAttentionBlock, self).__init__()

        self.mha_blocks = nn.ModuleList([])

        for mha_block_config in batched_mha_blocks_config:
            config = ensure_init_type(mha_block_config, MultiHeadAttentionBlockConfig)
            self.mha_blocks.append(MultiHeadAttentionBlock.from_config(config))

    def get_attention_mask(self, ordering: torch.Tensor) -> torch.Tensor:
        """
        Computes the required attention mask for a flattened batch with a variable number of feature vectors per batch element.
        The sequence ordering indicates how many elements of a flattened batch correspond to which batch element.

        E.g. batch_size=3, ordering=[0, 1, 1, 2]  => 1 feature in vector in batch element 0, 2 in 1, 1 in 2
            => attention_mask = [[False, True,  True,  True],
                                 [ True, False, False, True],
                                 [ True, False, False, True],
                                 [ True, True,  True,  False]])
            True: not allowed to attend
            False: allowed to attend

        Args:
            ordering: shape [total_num_nodes], indexing to determine which feature corresponds to which batch.

        Returns:
            mask tensor, shape [total_num_nodes, total_num_nodes]
        """
        idxs, idx_count = ordering.unique_consecutive(dim=0, return_counts=True)
        # ensure each batch idx is represented
        assert torch.equal(
            idxs, torch.arange(idxs[-1] + 1, device=idxs.device)
        ), f"Invalid ordering tensor: {ordering}, either not consecutive or not all batch element indices represented."

        offset = 0
        attn_mask = []
        for count in idx_count:
            block_matrix = torch.zeros(
                (count, count), dtype=torch.bool, device=ordering.device
            )

            attn_mask.append(
                torch.cat(
                    [
                        torch.ones(
                            (count, offset), dtype=torch.bool, device=ordering.device
                        ),
                        block_matrix,
                        torch.ones(
                            (count, sum(idx_count) - count - offset),
                            dtype=torch.bool,
                            device=ordering.device,
                        ),
                    ],
                    dim=1,
                )
            )
            offset += count

        return torch.cat(attn_mask)

    def forward(self, feat: torch.Tensor, ordering: torch.Tensor) -> torch.Tensor:
        """
        Applies the operations of multiple MHA blocks onto a sequence of tokens. No batch dimension since multiple batch elements can be combined into the same sequence.

        Args:
            feat: shape [total_num_nodes, feat_dim], a sequence of total_num_nodes tokens where each token is of feat_dim dimension.
            ordering: shape [total_num_nodes], indexing to determine which token corresponds to which batch element.

        Returns:
            x: shape [total_num_nodes, embed_dim]
        """
        assert (
            len(feat.shape) == 2
        ), "Features must contain a flattened batch of shape [total_num_nodes, feat_dim]; use ordering to handle batching."

        # compute attention mask
        attn_mask = self.get_attention_mask(ordering)

        x = feat.unsqueeze(0)  # [1, total_num_nodes, feat_dim] # send as batch size 1
        for block in self.mha_blocks:
            x = block(x, attn_mask)
            # x [1, total_num_nodes, feat_dim]
        return x.squeeze(0)


@dataclass
class StarGraphContextEncoderConfig:
    inputs: List[PolicyPin]
    outputs: List[PolicyPin]
    waypoints_encoder_config: dict
    graph_edge_config: dict
    gat_layers_config: List[dict]
    gat_aggregator_config: dict
    mha_aggregator_config: dict
    output_dim: int


class StarGraphContextEncoder(Policy):
    """
    Extracts features from a tensor of all points in a map and an auxiliary ordering tensor.
    Features are start, end vectors of x-y positions and 2-dim one-hot encoding, and are grouped by the polyline the features belong to.
    Vector features are encoded with a linear layer and returned together with the track encoding of the corresponding ego vehicle.
    """

    def __init__(
        self,
        inputs: List[PolicyPin],
        outputs: List[PolicyPin],
        waypoints_encoder_config: dict,
        graph_edge_config: dict,
        gat_layers_config: List[dict],
        gat_aggregator_config: dict,
        mha_aggregator_config: dict,
        output_dim: int,
    ):
        self.polyline_features_input: PolicyPin = None
        self.ego_waypoints_input: PolicyPin = None
        self.traffic_poses_input: PolicyPin = None
        self.context_features_output: PolicyPin = None

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            output_dim=output_dim,
        )

        assert 3 == len(inputs), f"Exactly 3 inputs are allowed, inputs = {inputs}"
        assert 1 == len(outputs), f"Exactly 1 output is allowed, outputs = {outputs}"

        # vector features shape with poly index removed (last element), [x_0, y_0, x_1, y_1, 3-dim type one-hot]
        self.vector_features_len = self.polyline_features_input.shape[1] - 1
        self.feat_dim = output_dim
        self.vector_features_layer = nn.Linear(
            in_features=self.vector_features_len, out_features=self.feat_dim, bias=True
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=(self.feat_dim))

        # waypoints encoder used for both ego and traffic
        self.waypoints_encoder = self.prepare_waypoints_encoder(
            waypoints_encoder_config
        )

        # star graph encoder: graph, GNN, graph aggregation, multi-head attention aggregation
        # TODO: generalize star graph models init via path_to_class and config
        self.star_graph = StarGraphEdgeGenerator.from_config(graph_edge_config)
        self.star_graph_gnn = FlexibleGAT(gat_layers_config=gat_layers_config)
        self.star_graph_aggregator = GraphAggregator.from_config(gat_aggregator_config)
        self.mha_aggregator = BatchedStarNetAttentionBlock(
            batched_mha_blocks_config=mha_aggregator_config
        )

    def prepare_waypoints_encoder(
        self, waypoints_encoder_config: dict
    ) -> torch.nn.Module:
        model_cf = waypoints_encoder_config["config"]
        return class_from_path(waypoints_encoder_config["path_to_class"]).from_config(
            model_cf
        )

    @staticmethod
    def depad_vector_features(poly_padded_features: torch.Tensor) -> torch.Tensor:
        """
        Remove padding from vector features.
        """
        # shape [batch_size*N, 9]
        poly_padded_features = poly_padded_features.reshape(
            -1, poly_padded_features.shape[2]
        )
        # find rows with any NaN
        poly_features_mask = torch.any(torch.isnan(poly_padded_features), dim=1)
        # remove rows with NaNs, shape [batch_size*N<<, 9]
        poly_features = poly_padded_features[~poly_features_mask, :]

        return poly_features

    @staticmethod
    def compute_batch_poly_stats(
        poly_idxs: torch.Tensor, batch_idxs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the number of polylines in each batch element and vectors in each polyline using vector polyline indices and vector batch indices.
        Example 1:
            poly_idxs: torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
            batch_idxs: torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
            num_polylines_per_batch_el: torch.tensor([2, 1])
            num_vectors_per_poly: torch.tensor([2, 3, 5]))
        Example 2:
            poly_idxs: torch.tensor([0, 0, 0, 0, 0, 0, 0])
            batch_idxs: torch.tensor([0, 0, 0, 1, 1, 1, 1])
            num_polylines_per_batch_el: torch.tensor([1, 1])
            num_vectors_per_poly: torch.tensor([3, 4]))
        """
        # find counts of each batch id
        _, batch_ids_counts = torch.unique_consecutive(batch_idxs, return_counts=True)
        # split poly ids according to batch they belong to
        # returns list of batch-grouped poly ids of vectors, example output: [tensor([p1, p1, p1, p2, p2]), tensor([p3, p3, p3, p3, p3]), ...]
        batch_split_poly_ids = torch.split(
            poly_idxs,
            split_size_or_sections=batch_ids_counts.tolist(),
            dim=0,
        )

        # iterate to find number of vectors in each poly and polylines in each batch element
        # returns list of tuples [poly_ids, poly_ids_counts] per batch element, example output: [(tensor([p1, p2]), tensor([3, 2])), (tensor([p3]), tensor([5]))...]
        batch_split_stats = [
            torch.unique_consecutive(batch_split, return_counts=True)
            for batch_split in batch_split_poly_ids
        ]
        # unzip list of tuples, example output: [tensor([p1, p2]), tensor([p3]), ...], [tensor([3, 2]), tensor([5]), ...]
        poly_ids_per_batch_el, poly_ids_count_per_batch_el = zip(*batch_split_stats)
        # find number of polylines per batch el, example output and shape: tensor([2, 1, ...]), [batch_size]
        num_polylines_per_batch_el = torch.tensor(
            [el.shape[0] for el in poly_ids_per_batch_el], device=poly_idxs.device
        )
        # find number of vectors per poly, example output and shape: [3, 2, 5, ...], [sum(num_polylines_per_batch_el)]
        num_vectors_per_poly = torch.cat(poly_ids_count_per_batch_el)

        return num_polylines_per_batch_el, num_vectors_per_poly

    def group_outer_node_features_by_polyline(
        self, vector_features: torch.Tensor, num_vectors_per_poly: torch.tensor
    ) -> List[torch.Tensor]:
        """
        Split features according to num of vectors in each poly.
        """
        # list of polyline-grouped vector embeds [tensor([v1, v2, v3]_p1), tensor([v1, v2]_p2), ...]
        poly_split_vector_features = torch.split(
            vector_features,
            split_size_or_sections=num_vectors_per_poly,
            dim=0,
        )

        return poly_split_vector_features

    def prepare_center_node_features_for_each_polyline(
        self,
        ego_waypoints_features: torch.Tensor,
        num_polylines_per_batch_el: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Prepare waypoint embeddings for each polyline while considering which batch element polyline/waypoints belong to.
        """
        # repeat waypoint features for each polyline, shape: [len(poly_split_vector_embeds), feat_dim]
        repeated_ego_waypoints_features = ego_waypoints_features.repeat_interleave(
            num_polylines_per_batch_el, dim=0
        )
        # split into list of len(poly_split_vector_embeds) tensors, shape: [1, feat_dim]
        repeated_ego_waypoints_features = torch.split(
            repeated_ego_waypoints_features, split_size_or_sections=1, dim=0
        )

        return repeated_ego_waypoints_features

    def create_star_graphs(
        self,
        num_outer_nodes_per_graph: List[int],
        center_node_features: List[torch.Tensor],
        outer_node_features: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create star graphs using waypoints and vectors features. Multiple graphs are joined into a single disjoint adjacency matrix graph covering the entire batch.
        """
        # prepend waypoint embeds to vector embeds by interleaving two lists
        num_graphs = len(num_outer_nodes_per_graph)
        star_graph_nodes = [None] * (2 * num_graphs)
        star_graph_nodes[
            ::2
        ] = center_node_features  # waypoint embeds are first, star graph origin
        star_graph_nodes[1::2] = outer_node_features  # vector embeds outside
        star_graph_nodes = torch.cat(star_graph_nodes)

        # compute star graph edges, +1 because of waypoints center node
        star_graph_edges = torch.cat(
            [
                self.star_graph(
                    single_graph_num_outer_nodes + 1, device=star_graph_nodes.device
                )
                for single_graph_num_outer_nodes in num_outer_nodes_per_graph
            ],
            dim=1,
        )

        return star_graph_nodes, star_graph_edges

    def generate_star_graph_features(
        self, ego_waypoints_features: torch.Tensor, padded_polylines: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate star graph features using embedded ego waypoints and polyline vectors.
        First, the polyline inputs are depadded and vectors are embedded with the vector_features_layer to generate vector embeddings.
        Then, the waypoint and vector embeddings are combined into star graphs (waypoints in the center) and aggregated via a StarGraph GNN.
        Finally, graph-level embeddings are computed via the star_graph_aggregator
        Args:
            padded_polylines: shape [batch_size, N, 8], 8 is 4-dim xy position of vector points + 3-dim one-hot enc. + 1-dim poly id. N is maximum number of vectors in a map (padding dim).
            ego_waypoints_features: shape [batch_size, feat_dim], embedded ego track
        Returns:
            poly_graph_features: shape [num_polylines, feat_dim], graph-level embeddings for each polyline in the batch
            poly_batch_ordering: shape [num_polylines], ordering of polylines per batch index
        """
        # extract poly shapes
        batch_size, num_vectors, vector_dim = padded_polylines.shape

        # append batch element indices to poly features [batch_size, N, 8] -> [batch_size, N, 9]
        batch_ids = torch.arange(batch_size, device=ego_waypoints_features.device)
        batch_ids = batch_ids.unsqueeze(-1).repeat(1, num_vectors)
        padded_polylines = torch.cat((padded_polylines, batch_ids.unsqueeze(-1)), dim=2)

        # remove padding
        poly_features = self.depad_vector_features(padded_polylines)

        # embed depadded vector features without polyline and batch ids
        vector_embeds = torch.relu(
            self.layer_norm(
                self.vector_features_layer(poly_features[:, : self.vector_features_len])
            )
        )

        # compute batch polyline statistics: number of polylines per batch element (scene) and number of vectors per polyline
        (
            num_polylines_per_batch_el,
            num_vectors_per_poly,
        ) = self.compute_batch_poly_stats(
            poly_idxs=poly_features[:, -2], batch_idxs=poly_features[:, -1]
        )
        num_polylines_in_batch = torch.sum(num_polylines_per_batch_el)
        num_vectors_per_poly = num_vectors_per_poly.tolist()

        # prepare vector embeds for each polyline
        poly_split_vector_embeds = self.group_outer_node_features_by_polyline(
            vector_embeds, num_vectors_per_poly
        )
        assert len(poly_split_vector_embeds) == num_polylines_in_batch

        # prepare waypoint embeds for each polyline
        repeated_ego_waypoints_features = (
            self.prepare_center_node_features_for_each_polyline(
                ego_waypoints_features, num_polylines_per_batch_el
            )
        )

        # create star graphs, where each polyline has a corresponding graph
        star_graph_nodes, star_graph_edges = self.create_star_graphs(
            num_outer_nodes_per_graph=num_vectors_per_poly,
            center_node_features=repeated_ego_waypoints_features,
            outer_node_features=poly_split_vector_embeds,
        )

        # process star graphs through GAT, shape [num_vectors_in_batch + num_polylines_in_batch, feat_dim]
        star_graph_nodes = self.star_graph_gnn(star_graph_nodes, star_graph_edges)
        assert (
            star_graph_nodes.shape[0]
            == sum(num_vectors_per_poly) + num_polylines_in_batch
        )

        # aggregate polyline-level graphs (waypoints+vectors), use num_vectors_per_poly
        poly_wise_idx = torch.arange(
            num_polylines_in_batch, device=ego_waypoints_features.device
        ).repeat_interleave(
            torch.tensor(num_vectors_per_poly, device=ego_waypoints_features.device) + 1
        )
        poly_graph_features = self.star_graph_aggregator(
            star_graph_nodes, poly_wise_idx
        )

        # generate ordering of polylines per batch element
        poly_batch_ordering = torch.arange(
            batch_size, device=poly_graph_features.device
        ).repeat_interleave(num_polylines_per_batch_el)

        return poly_graph_features, poly_batch_ordering

    def generate_traffic_waypoints_features(
        self, padded_traffic_poses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate traffic waypoints features by removing the padded_traffic_poses tensor padding, extracting waypoints from poses, and embedding them via the waypoints encoder, while keeping track which traffic object belongs to which batch element.
        Args:
            padded_traffic_poses: shape [batch_size, num_timesteps, M, 4], Traffic object poses, M is the maximum number of traffic objects in a scene (padding dim), 4 is existence at time-step flag and 3-dim pose
        Returns:
            traffic_waypoints_features: shape [num_traffic_objects_in_batch, feat_dim] Embedded waypoint features of every traffic object.
            traffic_batch_el_ordering: shape [num_traffic_objects_in_batch] Batch-el.-wise ordering of traffic objects.
        """
        # append batch element indices to traffic poses, [batch_size, num_timesteps, M, 4] -> [batch_size, num_timesteps, M, 5]
        batch_size, num_timesteps, num_objects, _ = padded_traffic_poses.shape
        batch_ids = torch.arange(batch_size, device=padded_traffic_poses.device)
        batch_ids = (
            batch_ids.unsqueeze(-1).unsqueeze(-1).repeat(1, num_timesteps, num_objects)
        )
        padded_traffic_poses = torch.cat(
            (padded_traffic_poses, batch_ids.unsqueeze(-1)), dim=3
        )

        # [batch_size, num_timesteps, M, 5] -> [batch_size, M, num_timesteps, 5] -> [batch_size*M, num_timesteps, 5]
        padded_traffic_poses = padded_traffic_poses.swapaxes(1, 2).reshape(
            batch_size * num_objects, num_timesteps, -1
        )

        # find non-padded objects
        traffic_object_mask = torch.sum(padded_traffic_poses[:, :, 0], dim=1) > 1.0

        # extract padded object poses, shape [batch_size*M<<, num_timesteps, 5]
        depadded_traffic_poses = padded_traffic_poses[traffic_object_mask, :, :]

        # extract batch-el. ordering, [batch_size*M<<]
        traffic_batch_el_ordering = depadded_traffic_poses[
            :, 0, -1
        ].long()  # 0 irrelevant

        # embed traffic waypoints + existence flag, [batch_size*M, feat_dim]
        traffic_waypoints_features = self.waypoints_encoder.forward(
            depadded_traffic_poses[:, :, :3]
        )

        return traffic_waypoints_features, traffic_batch_el_ordering

    def generate_ego_waypoints_features(
        self, ego_waypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ego_waypoints [batch_size, num_timesteps, 2]
        Returns:
            waypoints features [batch_size, feat_dim]
        """
        # append existence flag to ego waypoints
        waypoints_encoder_input_features = torch.cat(
            (
                torch.ones(
                    ego_waypoints.shape[:2], device=ego_waypoints.device
                ).unsqueeze(-1),
                ego_waypoints,
            ),
            dim=2,
        )

        return self.waypoints_encoder.forward(waypoints_encoder_input_features)

    def combine_ego_poly_traffic_features(
        self,
        ego_features: torch.Tensor,
        poly_features: torch.Tensor,
        poly_ordering: torch.Tensor,
        traffic_features: torch.Tensor,
        traffic_ordering: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combines ego, poly, and traffic embedded features into a single context feature vector as an output of a scene encoder and input to a trajectory decoder.
        All features tensors are flattened over an entire batch, hence ordering tensors keep track of which tensor element belongs to which batch element.
        Ego features have a single feature vector per batch element.
        Example ordering (batch_size 4):
        ego_ordering: [0, 1, 2, 3] (not an argument)
        poly_ordering: [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]  => 4, 3, 2, 1 polylines per batch el
        traffic_ordering: [0, 0, 2, 3, 3, 3, 3]  => 2, 0, 1, 4 traffic objects per batch el
        Args:
            ego_features [batch_size, feat_dim]
            poly_features [num_polylines_in_batch, feat_dim]
            poly_ordering [num_polylines_in_batch]
            traffic_features [num_traffic_objects_in_batch, feat_dim]
            traffic_ordering [num_traffic_objects_in_batch]
        Returns:
            context_features [batch_size + num_polylines_in_batch + num_traffic_objects_in_batch, feat_dim]
            context_features_ordering [batch_size + num_polylines_in_batch + num_traffic_objects_in_batch]
        """
        batch_size, feat_dim = ego_features.shape

        # count how many feature vectors are in each batch element for ego, poly, and traffic
        # zero feature vectors for poly or traffic for a batch element idx are possible
        # ego features have a single vector per batch element
        ego_features_batch_el_idx = torch.arange(batch_size, device=ego_features.device)
        (
            poly_features_batch_el_idx,
            poly_features_batch_el_count,
        ) = torch.unique_consecutive(poly_ordering, return_counts=True)
        (
            traffic_features_batch_el_idx,
            traffic_features_batch_el_count,
        ) = torch.unique_consecutive(traffic_ordering, return_counts=True)

        # split feature vectors acc. to batch element into a list of variable size tensors
        split_ego_features = torch.split(ego_features, split_size_or_sections=1, dim=0)
        split_poly_features = torch.split(
            poly_features,
            split_size_or_sections=poly_features_batch_el_count.tolist(),
            dim=0,
        )
        split_traffic_features = torch.split(
            traffic_features,
            split_size_or_sections=traffic_features_batch_el_count.tolist(),
            dim=0,
        )

        # make dicts of batch el idx: feature tensor
        ego_batch_el_idx_features_mapping = dict(
            zip(ego_features_batch_el_idx.tolist(), split_ego_features)
        )
        poly_batch_el_idx_features_mapping = dict(
            zip(poly_features_batch_el_idx.tolist(), split_poly_features)
        )
        traffic_batch_el_idx_features_mapping = dict(
            zip(traffic_features_batch_el_idx.tolist(), split_traffic_features)
        )

        # join the ego/poly/traffic features by batch el idx considering potentially non-existent (poly or traffic) features
        context_features = []
        context_features_ordering = []
        for idx in range(batch_size):
            # collect features for each batch el idx
            context_features_batch_el_idx = [ego_batch_el_idx_features_mapping[idx]]
            if idx in poly_batch_el_idx_features_mapping:
                context_features_batch_el_idx.append(
                    poly_batch_el_idx_features_mapping[idx]
                )
            if idx in traffic_batch_el_idx_features_mapping:
                context_features_batch_el_idx.append(
                    traffic_batch_el_idx_features_mapping[idx]
                )
            context_features_batch_el_idx = torch.cat(
                context_features_batch_el_idx, dim=0
            )

            # collect features in entire batch
            context_features.append(context_features_batch_el_idx)
            context_features_ordering.append(
                torch.full(
                    (context_features_batch_el_idx.shape[0],),
                    fill_value=idx,
                    device=context_features_batch_el_idx.device,
                )
            )

        context_features = torch.cat(context_features, dim=0)
        context_features_ordering = torch.cat(context_features_ordering, dim=0).long()

        return context_features, context_features_ordering

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            polyline_features_input: shape [batch_size, N, 8], 8 is 4-dim xy position of vector points + 3-dim one-hot enc. + 1-dim poly id. N is maximum number of vectors in a map (padding dim).
            ego_waypoints_features_input: shape [batch_size, num_timesteps, 2], Ego waypoints track
            traffic_waypoints_features_input: shape [batch_size, num_timesteps, M, 4], Traffic object poses, M is the maximum number of traffic objects in a scene (padding dim), 4 is existence at time-step flag and 3-dim pose
        Returns:
            all_features_agg: shape [batch_size, feat_dim] feature vectors aggregating polyline and traffic object features in a dummy manner
        """
        ego_waypoints = x[self.ego_waypoints_input.key]
        padded_polylines = x[self.polyline_features_input.key]
        padded_traffic_poses = x[self.traffic_poses_input.key]

        # encode ego waypoints, [batch_size, feat_dim]
        ego_waypoints_features = self.generate_ego_waypoints_features(ego_waypoints)

        # get star-graph-level features and num graphs per batch el., [num_polylines_in_batch, feat_dim], [batch_size]
        (
            star_graph_features,
            poly_batch_el_ordering,
        ) = self.generate_star_graph_features(ego_waypoints_features, padded_polylines)

        # get traffic object features
        (
            traffic_waypoints_features,
            traffic_batch_el_ordering,
        ) = self.generate_traffic_waypoints_features(padded_traffic_poses)

        # combine the ego, poly, and traffic features, considering batch element idxs
        (
            context_features,
            context_features_ordering,
        ) = self.combine_ego_poly_traffic_features(
            ego_features=ego_waypoints_features,
            poly_features=star_graph_features,
            poly_ordering=poly_batch_el_ordering,
            traffic_features=traffic_waypoints_features,
            traffic_ordering=traffic_batch_el_ordering,
        )

        # aggregate context features per batch element
        agg_context_features = self.mha_aggregator(
            context_features, context_features_ordering
        )

        # extract first feature vector corresponding to ego token
        (
            _,
            context_features_ordering_counts,
        ) = context_features_ordering.unique_consecutive(dim=0, return_counts=True)
        split_agg_context_features = torch.split(
            agg_context_features,
            split_size_or_sections=context_features_ordering_counts.tolist(),
            dim=0,
        )
        agg_ego_features = [
            agg_ego_feature[0, :] for agg_ego_feature in split_agg_context_features
        ]

        return torch.stack(agg_ego_features, dim=0)

    @classmethod
    def from_config(cls, config: Union[dict, StarGraphContextEncoderConfig]):
        if isinstance(config, dict):
            config = StarGraphContextEncoderConfig(**config)
        return cls(
            inputs=config.inputs,
            outputs=config.outputs,
            waypoints_encoder_config=config.waypoints_encoder_config,
            graph_edge_config=config.graph_edge_config,
            gat_layers_config=config.gat_layers_config,
            gat_aggregator_config=config.gat_aggregator_config,
            mha_aggregator_config=config.mha_aggregator_config,
            output_dim=config.output_dim,
        )


class Res1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        1D Residual block used in the temporal CNN encoder.
        """
        super(Res1d, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.group_norm = nn.GroupNorm(1, out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                self.group_norm,
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.conv1(x)
        out = self.group_norm(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.group_norm(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@dataclass
class TempCNNEncoderConfig:
    feat_dim: int = 64
    num_input_channels: int = 3
    scales: Tuple[float, float] = (2, 2)


class TempCNNEncoder(nn.Module):
    """
    Temporal (1D) CNN Encoder reproduced from the ActorNet model in LaneGCN: https://arxiv.org/abs/2007.13732
    """

    def __init__(
        self,
        feat_dim: int = 128,
        num_input_channels: int = 2,
        scales: Tuple[float, float] = (2.0, 2.0),
    ):
        """
        Args:
            feat_dim: dimension of output featurs
            num_input_channels: dimension of input features at each timestep, e.g. x, y, padding
            scales: used for interpolating a sequence of num_timesteps input features, must be tuned depending on num_timesteps. For example, (2, 2) implies that num_timesteps is divisible by 2*2=4 so that an integer number of elements in the sequence is obtained after interpolation.
        """
        super(TempCNNEncoder, self).__init__()

        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Res1d(in_channels=num_input_channels, out_channels=feat_dim // 4),
                    Res1d(in_channels=feat_dim // 4, out_channels=feat_dim // 4),
                ),
                nn.Sequential(
                    Res1d(
                        in_channels=feat_dim // 4, out_channels=feat_dim // 2, stride=2
                    ),
                    Res1d(in_channels=feat_dim // 2, out_channels=feat_dim // 2),
                ),
                nn.Sequential(
                    Res1d(in_channels=feat_dim // 2, out_channels=feat_dim, stride=2),
                    Res1d(in_channels=feat_dim, out_channels=feat_dim),
                ),
            ]
        )

        self.temp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=feat_dim // 4,
                        out_channels=feat_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, feat_dim),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=feat_dim // 2,
                        out_channels=feat_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, feat_dim),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=feat_dim,
                        out_channels=feat_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, feat_dim),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        self.scales = scales

        self.output = Res1d(in_channels=feat_dim, out_channels=feat_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input features [batch_size, num_timesteps, num_input_channels]

        Returns:
            out: output features [batch_size, feat_dim]
        """
        x = x.transpose(1, 2)  # switch order of num_timesteps and num_input_channels

        res_block_out = []
        for res_block in self.res_blocks:
            x = res_block(x)
            res_block_out.append(x)

        out = self.temp_blocks[-1](res_block_out[-1])

        out = F.interpolate(
            out, scale_factor=self.scales[1], mode="linear", align_corners=False
        )
        out += self.temp_blocks[1](res_block_out[1])

        out = F.interpolate(
            out, scale_factor=self.scales[0], mode="linear", align_corners=False
        )
        out += self.temp_blocks[0](res_block_out[0])

        out = self.output(out)[:, :, -1]

        return out

    @classmethod
    def from_config(cls, cf: dict):
        config = TempCNNEncoderConfig(**cf)
        return cls(
            feat_dim=config.feat_dim,
            num_input_channels=config.num_input_channels,
            scales=config.scales,
        )
