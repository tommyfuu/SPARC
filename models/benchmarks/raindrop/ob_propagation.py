"""Raindrop_v2 observation-propagation layer, ported from mims-harvard/Raindrop code/Ob_propagation.py.

Keeps only the executed path (`message`, use_beta=False); the unused message_selfattention and beta pruning are dropped.
"""
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.typing import PairTensor, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

try:
    # avoids depending on torch_scatter
    from torch_geometric.utils import scatter as pyg_scatter
except ImportError:  # pragma: no cover
    from torch_scatter import scatter as pyg_scatter


class Observation_progation(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int,
                 n_nodes: int, ob_dim: int, heads: int = 1, concat: bool = True,
                 dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.n_nodes = n_nodes
        self.ob_dim = ob_dim

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # message() only uses lin_value
        self.lin_value = Linear(in_channels[0], heads * out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_value.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index, edge_weights=None,
                edge_attr: OptTensor = None, return_attention_weights: Optional[bool] = None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the edge
                weights used for each edge (fed to the next layer as `edge_weights`).
        """
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_weights=edge_weights,
                              edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if return_attention_weights:
            return out, (edge_index, alpha)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_weights: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # the incoming edge weight is the attention value: fixed adjacency in layer 1,
        # layer 1's alpha in layer 2
        if edge_weights is not None:
            gamma = edge_weights.unsqueeze(-1)
        else:
            gamma = torch.ones(x_i.size(0), 1, device=x_i.device)

        self._alpha = gamma
        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        # uses x_i, not x_j, as in the official code
        out = F.relu(self.lin_value(x_i)).view(-1, self.heads, self.out_channels)
        out = out * gamma.view(-1, self.heads, 1)
        return out

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        reduce = 'sum' if self.aggr == 'add' else self.aggr
        return pyg_scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                            reduce=reduce)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.heads)
