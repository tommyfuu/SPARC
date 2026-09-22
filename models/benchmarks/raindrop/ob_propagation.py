"""
Ported from the official Raindrop_v2 implementation:
mims-harvard/Raindrop, code/Ob_propagation.py

This is the message-passing layer that makes a model "Raindrop_v2" rather
than the simpler "Raindrop" (v1) baseline: Raindrop_v2 calls this layer
twice per sample, feeding layer 1's output edge weights in as layer 2's
edge weights, so the sensor graph is refined between the two hops. That
chaining happens in the caller's forward(), not in this file.

NOTE: the official file defines two message-computing methods -- a content
based `message_selfattention` (query/key dot-product) that is never called,
and the `message` method actually invoked by PyG's MessagePassing.propagate().
The real Raindrop_v2 forward() also always calls with use_beta=False, which
skips the beta/edge-pruning logic entirely. This port keeps only the code
path that is actually executed, so this is behaviorally identical to the
official Raindrop_v2, not a re-derivation of it. If you want byte-for-byte
source parity (including the dead code), the original file is at the path
above.
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
    # Present in modern torch_geometric; avoids a hard dependency on the
    # separate torch_scatter package.
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

        # Only lin_value is needed: the actually-invoked `message()` below
        # does not use lin_key/lin_query (those back the unused
        # message_selfattention path in the original file).
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
                weights that were used for each edge (which is what the
                caller feeds in as `edge_weights` for the *next* layer).
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
        # Matches the official Raindrop_v2 code path exactly: the edge
        # weight passed in *is* the attention value (no learned query/key
        # score here) -- for layer 1 that's the fixed/complete adjacency,
        # for layer 2 it's layer 1's own (softmax-normalized) alpha, which
        # is the adaptive-structure mechanism that defines Raindrop_v2.
        if edge_weights is not None:
            gamma = edge_weights.unsqueeze(-1)
        else:
            gamma = torch.ones(x_i.size(0), 1, device=x_i.device)

        self._alpha = gamma
        gamma = softmax(gamma, index, ptr, size_i)
        gamma = F.dropout(gamma, p=self.dropout, training=self.training)

        # NOTE: uses x_i, not x_j -- this matches the official file as-is.
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
