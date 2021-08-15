from typing import List

import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GraphConv


class RGCN(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, rel_names: List[str]):
        super().__init__()
        self.layers = nn.ModuleList()
        last_feat = in_feat
        for _ in range(n_layers - 1):
            self.layers.append(HeteroGraphConv(
                {rel: GraphConv(last_feat, hidden_feat, activation=F.relu) for rel in rel_names},
                aggregate="mean"
            ))
            last_feat = hidden_feat
        self.layers.append(HeteroGraphConv(
            {rel: GraphConv(hidden_feat, n_classes) for rel in rel_names}, 
            aggregate="mean"
        ))

    def forward(self, graph: DGLGraph, target_node: str):
        h = graph.ndata["feat"]
        for layer in self.layers:
            h = layer(graph, h)
        return h[target_node]
