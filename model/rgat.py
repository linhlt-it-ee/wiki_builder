from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GATConv


class RGAT(nn.Module):
    def __init__(
        self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, 
        rel_names: List[str], aggregate: str = "sum", 
        num_heads: int = 2, multihead_aggregate: str = "concat",
        dropout: float = 0.2,
        ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.multihead_agg_fn = self.get_multihead_aggregate_fn(multihead_aggregate)

        last_feat = in_feat
        for _ in range(n_layers - 1):
            next_feat = hidden_feat // num_heads if multihead_aggregate == "concat" else hidden_feat
            self.convs.append(HeteroGraphConv(
                {rel: GATConv(last_feat, next_feat, num_heads, feat_drop=0.3) for rel in rel_names},
                aggregate=aggregate
            ))
            last_feat = hidden_feat
        
        self.clf = HeteroGraphConv({rel: GATConv(hidden_feat, n_classes, num_heads) for rel in rel_names})

    def get_multihead_aggregate_fn(self, multihead_aggregate):
        if multihead_aggregate == "concat":
            fn = lambda att, activation : torch.flatten(activation(att) if activation is not None else att, start_dim=1)
        elif multihead_aggregate == "mean":
            fn = lambda att, activation : torch.mean(activation(att) if activation is not None else att, dim=1)
        else:
            raise NotImplementedError
        return fn

    def forward(self, graph: DGLGraph, inputs: Dict, target_node: str, edge_weight: Dict = None, return_features: bool = False):
        h = {k: self.dropout(v) for k, v in inputs.items()}
        for i, conv in enumerate(self.convs):
            h = conv(graph, h)
            h = {k: self.multihead_agg_fn(v, F.relu) for k, v in h.items()}
        features = h[target_node]
        h = self.clf(graph, h)
        logits = self.get_multihead_aggregate_fn("mean")(h[target_node], None)
        return (features, logits) if return_features else logits
