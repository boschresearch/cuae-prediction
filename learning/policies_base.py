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
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv

from lib.utils import class_from_path, ensure_init_type


class FlexibleMLP(torch.nn.Module):
    """
    Flexible multi-layer perceptron with tunable activation and normalization after every hidden layer.

    Starts with an input dim, adds a variable number of linear layers with activation and normalization, ends with an optional output dim:
    input_dim -> [lin1 -> act1 -> norm1 -> lin2 -> ... -> actN -> normN -> linN] { -> output_dim }
    """

    def __init__(
        self,
        input_dim: int,
        lin_layers: List[Dict],
        output_dim: Union[int, None] = None,
    ) -> None:
        super().__init__()
        self.layers = []
        if not lin_layers:
            assert (
                output_dim is not None
            ), "output_dim must be defined if no linear layers are provided"
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            dim = input_dim
            for layer in lin_layers:
                self.layers.append(nn.Linear(dim, layer["dim"], bias=False))
                if layer["norm"] is not None:
                    self.layers.append(class_from_path(layer["norm"])(layer["dim"]))
                self.layers.append(class_from_path(layer["activation"])())
                dim = layer["dim"]

            if output_dim is not None:
                self.layers.append(nn.Linear(dim, output_dim))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LoopingGRU(nn.Module):
    """
    Wrapper for a GRU that takes as input a sequence of feature vectors and iterates over it num_loops times.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        num_loops,
    ):
        super(LoopingGRU, self).__init__()

        self.num_loops = num_loops
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor):
        """
        Shapes of intermediate features:
        hidden: [n_layers, batch_size, hidden_dim]
        out: [batch_size, seq_length, hidden_dim]

        :param x: [batch_size, seq_length, in_dim]
        :return x: [batch_size, seq_length, in_dim]
        """
        # ensure model parameters are contiguous in memory
        self.gru.flatten_parameters()

        hidden = None
        for i in range(self.num_loops):
            out, hidden = self.gru(x, hidden)
            x = self.fc(out)
        # detach hidden state from history to prevent backpropagating through entire history
        hidden = hidden.data

        return x


@dataclass
class GATLayerConfig:
    gat_conv_path: str
    in_channels: int
    out_channels: int
    heads: int
    concatenate: Optional[bool] = False
    norm: Optional[str] = None
    activation: Optional[str] = None


class GATLayer(torch.nn.Module):
    """
    Single GAT layer with tunable activation and normalization.
    """

    def __init__(
        self,
        gat_conv_path: str,
        in_channels: int,
        out_channels: int,
        heads: int,
        concatenate: Optional[bool],
        norm: Optional[Union[None, str]] = None,
        activation: Optional[Union[None, str]] = None,
    ):
        """
        :param gat_conv_path path to GAT conv block from torch_geometric
        :param in_channels, number of input channels
        :param out_channels, number of output channels
        :param heads, number of attention heads
        :param concatenate, tunes whether to concatenate or average the output of different attention heads
        :param norm, normalization layer after the GAT conv block
        :param activation, activation layer after the GAT conv block
        """
        assert gat_conv_path in [
            "torch_geometric.nn.GATConv",
            "torch_geometric.nn.GATv2Conv",
        ], "Invalid GATConv path"

        super().__init__()
        self.gat_layer = nn.ModuleList([])
        self.gat_layer.append(
            class_from_path(gat_conv_path)(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concatenate,
            )
        )

        norm_in_channels = heads * out_channels if concatenate else out_channels

        if norm:
            self.gat_layer.append(class_from_path(norm)(norm_in_channels))

        if activation:
            self.gat_layer.append(class_from_path(activation)())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param x: node features, [num_nodes, feat_dim]
        :param edge_index: node connectivity, [2, num_edges]
        """
        for layer in self.gat_layer:
            if isinstance(layer, GATConv) or isinstance(layer, GATv2Conv):
                x = layer(x, edge_index=edge_index)
            else:
                x = layer(x)
        return x

    @classmethod
    def from_config(cls, config: Union[Dict, GATLayerConfig]):
        config = ensure_init_type(config, GATLayerConfig)
        return cls(
            config.gat_conv_path,
            config.in_channels,
            config.out_channels,
            config.heads,
            config.concatenate,
            config.norm,
            config.activation,
        )


class FlexibleGAT(torch.nn.Module):
    """
    Flexible multi-layer GAT network.

        gat_layers_config example:
        [
            {
                "gat_conv_path": "torch_geometric.nn.GATConv",
                "in_channels": 128
                "out_channels": 64,
                "heads": 5,
                "concatenate": True,
                "norm": "torch.nn.LayerNorm",
                "activation": "torch.nn.ReLU",
            },
            {
                "gat_conv_path": "torch_geometric.nn.GATConv",
                "in_channels": 64
                "out_channels": 32,
                "heads": 5,
                "concatenate": True,
                "norm": "torch.nn.LayerNorm",
                "activation": "torch.nn.ReLU",
            }
        ]

        num_out_channels of last layer:
            if concatenate == True:
                num_out_channels = out_channels
            else:
                num_out_channels = heads * out_channels
    """

    def __init__(self, gat_layers_config: List[Union[Dict, GATLayerConfig]]) -> None:
        super().__init__()

        assert len(gat_layers_config) > 0, "No GAT layers are specified"

        self.layers = nn.ModuleList([])

        for layer_config in gat_layers_config:
            config = ensure_init_type(layer_config, GATLayerConfig)
            self.layers.append(GATLayer.from_config(config))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        :param x: node features, [num_nodes, feat_dim]
        :param edge_index: node connectivity, [2, num_edges]
        """
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


@dataclass
class MultiHeadAttentionBlockConfig:
    feat_dim: int
    num_heads: int
    norm: str
    lin_layers: List[Dict]


class MultiHeadAttentionBlock(nn.Module):
    """
    Building block of the Transformer, Multi-Head Attention (MHA) followed by linear layer(s), implemented by FlexibleMLP. The ordering of the norm/MHA/lin operations is slightly adapted from the original paper to match the Vision Transformer implementation.
    """

    def __init__(
        self,
        feat_dim: int,
        num_heads: int,
        norm: str,
        lin_layers: List[Dict],
    ):
        """
        Args:
            feat_dim: feature dimension at the input and output
            heads, number of attention heads
            norm, normalization layer after the MHA and linear layer(s)
            lin_layers, linear layer(s) after the MHA
        """
        super().__init__()
        self.norm = class_from_path(norm)(feat_dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp = FlexibleMLP(
            input_dim=feat_dim, lin_layers=lin_layers, output_dim=feat_dim
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
             x: input features [batch_size, sequence_len, feat_dim]
             mask: attention mask [sequence_len, sequence_len]
        """
        assert (
            len(x.shape) == 3
        ), "Input features must consist of [batch_size, sequence_len, feat_dim]"

        norm_x = self.norm(x)

        # compute self-attention
        attn_output, _ = self.mha(
            query=norm_x, key=norm_x, value=norm_x, attn_mask=mask
        )

        x = x + attn_output
        x = x + self.mlp(self.norm(x))
        return x

    @classmethod
    def from_config(cls, config: Union[Dict, MultiHeadAttentionBlockConfig]):
        config = ensure_init_type(config, MultiHeadAttentionBlockConfig)
        return cls(
            feat_dim=config.feat_dim,
            num_heads=config.num_heads,
            norm=config.norm,
            lin_layers=config.lin_layers,
        )
