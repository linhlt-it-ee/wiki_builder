import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GraphConv


class RGCN(nn.Module):
    def __init__(self, graph: DGLGraph, hidden_feat: int, n_classes: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroGraphConv(
            {rel: GraphConv(graph.nodes[src].data["feat"].shape[1], hidden_feat, activation=F.relu) for src, rel, dst in graph.canonical_etypes}, 
            aggregate="mean"
        ))
        for _ in range(n_layers - 1):
            self.layers.append(HeteroGraphConv(
                {rel: GraphConv(hidden_feat, hidden_feat, activation=F.relu) for rel in graph.etypes}, 
                aggregate="mean"
            ))
        self.layers.append(HeteroGraphConv(
            {rel: GraphConv(hidden_feat, n_classes) for rel in graph.etypes}, 
            aggregate="mean"
        ))

    def forward(self, graph: DGLGraph, target_node: str):
        h = graph.ndata["feat"]
        for layer in self.layers:
            h = layer(graph, h)
        return h[target_node]
