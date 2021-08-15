from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GATConv


class MultiHeadGATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        x = torch.mean(x, dim=1)
        return x

class RGAT(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, rel_names: List[str], num_heads: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        last_feat = in_feat
        for _ in range(n_layers - 1):
            self.layers.append(HeteroGraphConv(
                {rel: MultiHeadGATConv(last_feat, hidden_feat, num_heads, activation=F.relu) for rel in rel_names},
                aggregate="mean"
            ))
            last_feat = hidden_feat
        self.layers.append(HeteroGraphConv(
            {rel: MultiHeadGATConv(hidden_feat, n_classes, num_heads) for rel in rel_names}, 
            aggregate="mean"
        ))

    def forward(self, graph: DGLGraph, target_node: str):
        h = graph.ndata["feat"]
        for layer in self.layers:
            h = layer(graph, h)
        return h[target_node]
