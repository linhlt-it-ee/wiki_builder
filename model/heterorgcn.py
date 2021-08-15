from typing import List
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as dfn
from dgl import DGLGraph


class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, rel_names: List[str]):
        super().__init__()
        self.linears = nn.ModuleDict({rel: nn.Linear(in_feat, out_feat) for rel in rel_names})

    def forward(self, graph: DGLGraph, feat):
        msg_passing_fn = {}
        for src, rel, dst in graph.canonical_etypes:
            Wh = self.linears[rel](feat[src])
            graph.nodes[src].data["Wh_%s" % rel] = Wh
            msg_fn = dfn.copy_u("Wh_%s" % rel, "m")
            reduce_fn = dfn.mean("m", "h")
            msg_passing_fn[rel] = (msg_fn, reduce_fn)
        graph.multi_update_all(msg_passing_fn, "sum")
        return graph.ndata["h"]

class HeteroRGCN(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, n_layers: int, rel_names: List[str]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HeteroRGCNLayer(in_feat, hidden_feat, rel_names))
        for _ in range(n_layers - 1):
            self.layers.append(HeteroRGCNLayer(hidden_feat, hidden_feat, rel_names))
        self.layers.append(HeteroRGCNLayer(hidden_feat, n_classes, rel_names))

    def forward(self, graph: DGLGraph, target_node: str):
        h = graph.ndata["feat"]
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            if i != len(self.layers) - 1:
                h = {k : F.relu(v) for k, v in h.items()}
        return h[target_node]
