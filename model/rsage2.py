from typing import List, Dict

import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, SAGEConv


class RSAGE2(nn.Module):
    def __init__(
        self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, 
        rel_names: List[str], aggregate: str = "sum", 
        dropout: float = 0.2
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.skips = nn.ModuleList()

        last_feat = in_feat
        for _ in range(n_layers - 1):
            self.convs.append(HeteroGraphConv(
                {rel: SAGEConv(last_feat, hidden_feat, aggregator_type="mean") for rel in rel_names},
                aggregate=aggregate
            ))
            self.skips.append(nn.Linear(last_feat, hidden_feat))
            last_feat = hidden_feat

        self.clf = nn.Sequential(
            nn.Linear(hidden_feat, hidden_feat),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_feat, n_classes)
        )

    def forward(self, graph: DGLGraph, inputs: Dict, target_node: str, edge_weight: Dict = None, return_features: bool = False):
        h = inputs
        mod_kwargs = {rel: {"edge_weight": weight} for rel, weight in edge_weight.items()}
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            h1 = {k: skip(v) for k, v in h.items()}
            h2 = conv(graph, h, mod_kwargs=mod_kwargs)
            h = {k: h1[k] + F.relu(h2[k]) for k in h.keys()}
        features = h[target_node]
        logits = self.clf(features)
        return (features, logits) if return_features else logits
