import dgl.function as dfn
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HeteroRGCNLayer(nn.Module):
    def __init__(self, graph, in_dims, out_dims):
        super(HeteroRGCNLayer, self).__init__()
        self.linears = nn.ModuleDict(
            {
                ntype: nn.Linear(in_dims[ntype], out_dims[ntype])
                for ntype in graph.ntypes
            }
        ).to(device)

    def forward(self, graph, inputs):
        msg_passing_fns = {}
        for stype, etype, dtype in graph.canonical_etypes:
            Wh = self.linears[stype](inputs[stype])
            graph.nodes[stype].data["Wh_%s" % etype] = Wh
            msg_fn = dfn.copy_u("Wh_%s" % etype, "m")
            # msg_fn = dfn.u_mul_e("Wh_%s" % etype, "weight", "m")
            reduce_fn = dfn.mean("m", "h")
            msg_passing_fns[etype] = (msg_fn, reduce_fn)

        graph.multi_update_all(msg_passing_fns, "mean")
        return graph.ndata["h"]


class HeteroRGCN(nn.Module):
    def __init__(self, graph, in_dims, hid_dims, out_dims):
        super(HeteroRGCN, self).__init__()
        self.layer1 = HeteroRGCNLayer(graph, in_dims, hid_dims)
        self.layer2 = HeteroRGCNLayer(graph, hid_dims, out_dims)

    def forward(self, graph, out_type):
        inputs = {ntype: graph.ndata["feat"][ntype] for ntype in graph.ntypes}
        x = self.layer1(graph, inputs)
        x = {ntype: F.relu(x[ntype]) for ntype in graph.ntypes}
        x = self.layer2(graph, x)
        return x[out_type]
