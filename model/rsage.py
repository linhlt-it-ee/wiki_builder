from typing import List, Dict

import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, SAGEConv


class RSAGE(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, rel_names: List[str], aggregate: str = "sum"):
        super().__init__()
        self.layers = nn.ModuleList()

        last_feat = in_feat
        for _ in range(n_layers - 1):
            self.layers.append(HeteroGraphConv(
                {rel: SAGEConv(last_feat, hidden_feat, aggregator_type="mean") for rel in rel_names},
                aggregate=aggregate
            ))
            last_feat = hidden_feat

        self.layers.append(HeteroGraphConv(
            {rel: SAGEConv(last_feat, n_classes, aggregator_type="mean") for rel in rel_names}, 
            aggregate=aggregate
        ))

    def forward(self, graph: DGLGraph, inputs: Dict, target_node: str, edge_weight: Dict = None):
        h = inputs
        mod_kwargs = {rel: {"edge_weight": weight} for rel, weight in edge_weight.items()}
        for i, layer in enumerate(self.layers):
            h = layer(graph, h, mod_kwargs=mod_kwargs)
            if i != len(self.layers) - 1:
                h = {k: F.relu(v) for k, v in h.items()}
        return h[target_node]
