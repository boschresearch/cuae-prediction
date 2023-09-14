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
from enum import Enum
from typing import Dict, Tuple, Union

import torch

from lib.utils import class_from_path, ensure_init_type


@dataclass
class GraphEdgeGeneratorConfig:
    self_loop: bool
    bidirect: bool


class GraphEdgeGenerator:
    def __init__(self, self_loop: bool = False, bidirect: bool = True):
        self.self_loops = self_loop
        self.bidirect = bidirect

        # For efficiency purposes, stores pre-computed connectivity tensors for node cardinality of graphs encountered throughout class lifetime.
        self._precomputed_edges = {}

    def query_precomputed_edges_with_num_nodes(self, num_nodes: int) -> bool:
        """
        Returns True if num_nodes exists in self._precomputed_edges, i.e. if a graph for num_nodes nodes is already stored in the class.
        """
        if num_nodes in self._precomputed_edges:
            return True
        else:
            return False

    def update_precomputed_edges(
        self, src: torch.Tensor, dst: torch.Tensor, num_nodes: int
    ):
        edges = torch.stack((src, dst), dim=0)
        self._precomputed_edges[num_nodes] = edges

    def add_self_loops(
        self, src: torch.Tensor, dst: torch.Tensor, num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extends edges with self-loops.
        """
        if self.self_loops:
            all_nodes_idxs = torch.arange(
                start=0, end=num_nodes, device=src.device, dtype=torch.long
            )
            src = torch.cat((src, all_nodes_idxs))
            dst = torch.cat((dst, all_nodes_idxs))

        return src, dst

    def __call__(self, num_nodes: int, device: torch.device):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Union[Dict, GraphEdgeGeneratorConfig]):
        config = ensure_init_type(config, GraphEdgeGeneratorConfig)
        return cls(self_loop=config.self_loop, bidirect=config.bidirect)


class StarGraphEdgeGenerator(GraphEdgeGenerator):
    def __init__(self, self_loop: bool = False, bidirect: bool = True):
        super().__init__(self_loop=self_loop, bidirect=bidirect)

    def __call__(self, num_nodes: int, device=torch.device("cpu")) -> torch.Tensor:
        """
        Computes edge tensor of source and destination indices for a star graph with node 0 in the center.
        :param num_nodes: Number of nodes in the graph
        :return edges: shape [2, num_edges], two rows of source and destination nodes given a number of nodes
        example source and destination nodes for n=4 (bidirect==True, self_loops==False): [0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]
        """
        if self.query_precomputed_edges_with_num_nodes(num_nodes):
            return self._precomputed_edges[num_nodes]
        else:
            center_node = torch.zeros(num_nodes - 1, device=device, dtype=torch.long)
            outer_nodes = torch.arange(
                start=1, end=num_nodes, device=device, dtype=torch.long
            )

            if self.bidirect:
                src = torch.cat((center_node, outer_nodes), dim=0)
                dst = torch.cat((outer_nodes, center_node), dim=0)
            else:
                src = outer_nodes
                dst = center_node

            src, dst = self.add_self_loops(src, dst, num_nodes)

            self.update_precomputed_edges(src, dst, num_nodes)

            return self._precomputed_edges[num_nodes]

    @classmethod
    def from_config(cls, config: Union[Dict, GraphEdgeGeneratorConfig]):
        return super().from_config(config)


class GraphAggregatorOperator(Enum):
    scatter_max = "torch_scatter.scatter_max"


@dataclass
class GraphAggregatorConfig:
    operator: str


class GraphAggregator(torch.nn.Module):
    def __init__(self, operator: str):
        """
        Aggregates subgraphs of a given graph nodes using a permutation-invariant pooling function like max.
        """
        self.allowed_operators = {el.name: el.value for el in GraphAggregatorOperator}
        assert (
            operator in self.allowed_operators.keys()
        ), f"Invalid graph aggregator: {operator}, not in {self.allowed_operators}"
        self.operator = operator

        super().__init__()

    def forward(self, feat: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """
        :param feat: [num_nodes, feat_dim], example [[1. 2.], [0., 10.], [5., 5.]]
        :param order: [num_nodes], controls which nodes establish subgraphs to be aggregated, example: [0, 0, 1]
        :param result: [num_subgraphs, feat_dim], num_subgraphs is number of aggregated subgraphs determined by order / the number of unique elements in order, example with scatter_max: [[1., 10.], [5., 5.]]
        """
        agg_function = class_from_path(self.allowed_operators[self.operator])
        output_feat, _ = agg_function(feat, order, dim=0)
        return output_feat

    @classmethod
    def from_config(cls, config: Union[Dict, GraphAggregatorConfig]):
        config = ensure_init_type(config, GraphAggregatorConfig)
        return cls(config.operator)
