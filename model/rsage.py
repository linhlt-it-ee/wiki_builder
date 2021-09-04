from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, SAGEConv


class RSAGE(nn.Module):
    def __init__(
        self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, 
        rel_names: List[str], aggregate: str = "sum",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        last_feat = in_feat
        for _ in range(n_layers - 1):
            self.convs.append(HeteroGraphConv(
                {rel: SAGEConv(last_feat, hidden_feat, aggregator_type="mean", feat_drop=0.5) for rel in rel_names},
                aggregate=aggregate,
            ))
            last_feat = hidden_feat

        self.clf = HeteroGraphConv(
            {rel: SAGEConv(hidden_feat, n_classes, aggregator_type="pool") for rel in rel_names},
            aggregate=aggregate,
        )

    def forward(
        self, graph: DGLGraph, inputs: Dict, 
        target_node: str, edge_weight: Dict = None, 
        return_features: bool = False
    ):
        h = {k: self.dropout(v) for k, v in inputs.items()}
        mod_kwargs = {rel: {"edge_weight": weight} for rel, weight in edge_weight.items()}
        for i, conv in enumerate(self.convs):
            h = conv(graph, h, mod_kwargs=mod_kwargs)
            h = {k: F.relu(v) for k, v in h.items()}
        features = h[target_node]
        h = self.clf(graph, h, mod_kwargs=mod_kwargs)
        logits = h[target_node]
        return (features, logits) if return_features else logits
