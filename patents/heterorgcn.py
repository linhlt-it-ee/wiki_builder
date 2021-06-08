import dgl.function as dfn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, List


class HeteroRGCNLayer(nn.Module):
    def __init__(
        self, in_dims: Union[Dict[str, int], int], out_dim: int, etypes: List[str]
    ) -> None:
        super().__init__()
        if isinstance(in_dims, dict):
            self.linears = nn.ModuleDict(
                {x: nn.Linear(in_dims[x], out_dim) for x in etypes}
            )
        else:
            self.linears = nn.ModuleDict(
                {x: nn.Linear(in_dims, out_dim) for x in etypes}
            )

    def forward(
        self, graph: dgl.DGLHeteroGraph, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        msg_passing_fns = {}
        for stype, etype, dtype in graph.canonical_etypes:
            Wh = self.linears[etype](inputs[stype])
            graph.nodes[stype].data["Wh_%s" % etype] = Wh
            msg_fn = dfn.copy_u("Wh_%s" % etype, "m")
            reduce_fn = dfn.mean("m", "h")
            msg_passing_fns[etype] = (msg_fn, reduce_fn)

        graph.multi_update_all(msg_passing_fns, "mean")
        return graph.ndata["h"]


class HeteroRGCN(nn.Module):
    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        in_dims: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.layer1 = HeteroRGCNLayer(in_dims, hidden_dim, graph.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_dim, out_dim, graph.etypes)

    def forward(self, graph: dgl.DGLHeteroGraph, out_type: str) -> torch.Tensor:
        inputs = {ntype: graph.ndata["feature"][ntype] for ntype in graph.ntypes}
        x = self.layer1(graph, inputs)
        x = {ntype: F.relu(x[ntype]) for ntype in graph.ntypes}
        x = self.layer2(graph, x)
        return x[out_type]
