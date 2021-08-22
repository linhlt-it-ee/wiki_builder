from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GATConv, GraphConv


class RGAT(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, aggregate: str = "sum", multihead_aggregate: str = "concat", rel_names: List[str] = None, num_heads: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.multihead_agg_fn = self.get_multihead_aggregate_fn(multihead_aggregate)

        last_feat = in_feat
        for _ in range(n_layers - 1):
            next_feat = hidden_feat // num_heads if multihead_aggregate == "concat" else hidden_feat
            self.layers.append(HeteroGraphConv(
                {rel: GATConv(last_feat, next_feat, num_heads) for rel in rel_names},
                aggregate=aggregate
            ))
            last_feat = hidden_feat

        self.clf = nn.Sequential(
            nn.BatchNorm1d(last_feat),
            nn.Linear(last_feat, n_classes),
        )

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
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            h = {k: self.multihead_agg_fn(v, F.relu) for k, v in h.items()}
        return self.clf(h[target_node])
