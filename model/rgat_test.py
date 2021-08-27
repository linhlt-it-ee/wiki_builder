from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GATConv, GraphConv


class RGATtest(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, aggregate: str = "sum", multihead_aggregate: str = "concat", rel_names: List[str] = None, node_names: List[str] = None, num_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.norms = nn.ModuleList()

        last_feat = in_feat
        for _ in range(n_layers - 1):
            next_feat = hidden_feat // num_heads if multihead_aggregate == "concat" else hidden_feat
            self.convs.append(HeteroGraphConv(
                {rel: GATConv(last_feat, next_feat, num_heads) for rel in rel_names},
                aggregate=aggregate
            ))
            self.skips.append(nn.Linear(last_feat, hidden_feat))
            self.norms.append(nn.BatchNorm1d(hidden_feat))
            last_feat = hidden_feat

        self.clf = nn.Sequential(
            nn.Linear(last_feat, last_feat),
            nn.BatchNorm1d(last_feat),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(last_feat, n_classes)    
        )
        self.multihead_agg_fn = self.get_multihead_aggregate_fn(multihead_aggregate)

    def get_multihead_aggregate_fn(self, multihead_aggregate):
        if multihead_aggregate == "concat":
            fn = lambda att, activation : torch.reshape(activation(att), (att.shape[0], -1))
        elif multihead_aggregate == "mean":
            fn = lambda att, activation : torch.mean(activation(att), dim=1)
        else:
            raise NotImplementedError
        return fn

    def forward(self, graph: DGLGraph, target_node: str):
        h = graph.ndata["feat"]
        for i, (skip, conv, norm) in enumerate(zip(self.skips, self.convs, self.norms)):
            h1 = {k: skip(v) for k, v in h.items()}
            h2 = conv(graph, h)
            h2 = {k: self.multihead_agg_fn(v, F.relu) for k, v in h2.items()}
            h = {k: norm(h1[k] + h2[k]) for k in h.keys()}
        return self.clf(h[target_node])
