from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import HeteroGraphConv, GraphConv


def get_att_aggregate_fn(cat_weight, alpha: float = 0.5):
    def fn(src_feat, self_feat):
        stack = torch.stack(src_feat, dim=1)
        cat = torch.cat((stack, self_feat.expand_as(stack.transpose(0, 1)).transpose(0 ,1)), dim=2)
        e = torch.matmul(cat, cat_weight)
        score = F.softmax(F.leaky_relu(e), dim=1)
        agg_dst = torch.sum(score * stack, dim=1)
        return agg_dst if self_feat is None else (1 - alpha) * agg_dst + alpha * self_feat
    return fn

class HeteroGraphAttentionConv(HeteroGraphConv):
    def __init__(self, *args, att_in_feat: int, att_out_feat: int, alpha: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.att_in_feat = att_in_feat
        self.att_out_feat = att_out_feat
        self.att_cat_weight = nn.Parameter(torch.Tensor(self.att_out_feat * 2, 1))
        nn.init.xavier_uniform_(self.att_cat_weight)
        self.agg_fn = get_att_aggregate_fn(self.att_cat_weight, self.alpha)

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        mod_args = mod_args if mod_args is not None else {}
        mod_kwargs = mod_kwargs if mod_kwargs is not None else {}
        outputs = {nty: {} for nty in g.dsttypes}
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            if stype not in inputs:
                continue
            dstdata = self.mods[etype](
                rel_graph,
                (inputs[stype], inputs[dtype]),
                *mod_args.get(etype, ()),
                **mod_kwargs.get(etype, {}),
            )
            outputs[dtype][etype] = dstdata

        rsts = {}
        for nty, adict in outputs.items():
            if len(adict) != 0:
                new_self_feature = adict.get("self_connection", None)
                alist = [v for k, v in adict.items() if k != "self_connection"]
                rsts[nty] = self.agg_fn(alist, new_self_feature)

        return rsts

class RGAT(nn.Module):
    def __init__(self, in_feat: int, hidden_feat: int, n_classes: int, att_feat: int, n_layers: int, rel_names: List[str], aggregate: str = "sum"):
        super().__init__()
        self.layers = nn.ModuleList()
        last_feat = in_feat
        for _ in range(n_layers - 1):
            self.layers.append(HeteroGraphAttentionConv(
                {rel: GraphConv(last_feat, hidden_feat) for rel in rel_names},
                aggregate=aggregate,
                att_in_feat=n_classes,
                att_out_feat=att_feat,
            ))
            last_feat = hidden_feat
        self.layers.append(HeteroGraphAttentionConv(
            {rel: GraphConv(hidden_feat, n_classes) for rel in rel_names}, 
            aggregate=aggregate,
            att_in_feat=n_classes,
            att_out_feat=att_feat,
        ))

    def forward(self, graph: DGLGraph, target_node: str):
        h = graph.ndata["feat"]
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            if i != len(self.layers) - 1:
                h = {k: F.relu(v) for k, v in h.items()}
        return h[target_node]
