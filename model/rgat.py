from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GATConv
from .gatconv import WeightedGATConv

class RGAT(nn.Module):
    def __init__(
        self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, 
        rel_names: List[str], aggregate: str = "sum", 
        num_heads: int = 2, multihead_aggregate: str = "concat",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.multihead_agg_fn = self.get_multihead_aggregate_fn(multihead_aggregate)

        last_feat = in_feat
        for i in range(n_layers - 1):
            hidden_head_feat = hidden_feat // num_heads if multihead_aggregate == "concat" else hidden_feat
            self.convs.append(HeteroGraphConv(
                {rel: WeightedGATConv(last_feat, hidden_head_feat, num_heads, feat_drop=dropout, attn_drop=dropout) for rel in rel_names},
                aggregate=aggregate,
            ))
            self.skips.append(nn.Linear(last_feat, hidden_feat))
            last_feat = hidden_feat

        self.clf = nn.Sequential(
            # nn.Linear(hidden_feat, hidden_feat),
            nn.BatchNorm1d(hidden_feat),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_feat, n_classes)
        )

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
        mod_kwargs = {rel: {"edge_weight": weight} for rel, weight in edge_weight.items()}
        for i, (skip, conv) in enumerate(zip(self.skips, self.convs)):
            h1 = {k: skip(v) for k, v in h.items()}
            h2 = conv(graph, h, mod_kwargs=mod_kwargs)
            h2 = {k: self.multihead_agg_fn(v, F.relu) for k, v in h2.items()}
            h = {k: h1[k] + h2[k] for k in h.keys()}
        features = h[target_node]
        logits = self.clf(features)
        return (features, logits) if return_features else logits
