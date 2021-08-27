import torch
import torch.nn as nn
from dgl import DGLGraph
from .rgcn import *
from .rgcn2 import *
from .rsage import *
from .rsage2 import *
from .rgat import *
from .rgat2 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_model(
    model_name: str, graph: DGLGraph, n_classes: int, 
    hidden_feat: int = 64, n_layers: int = 2, aggregate: str = "sum", 
    num_heads: int = 2, multihead_aggregate: str = "concat", dropout: float = 0.2,
) -> nn.Module:
    in_feat = 768
    if model_name == "rgcn":
        model = RGCN(
            in_feat, hidden_feat, n_classes, n_layers, 
            rel_names=graph.etypes, aggregate=aggregate,
        )
    elif model_name == "rgcn2":
        model = RGCN2(
            in_feat, hidden_feat, n_classes, n_layers, 
            rel_names=graph.etypes, aggregate=aggregate,
            dropout=dropout,
        )
    elif model_name == "rgat":
        model = RGAT(
            in_feat, hidden_feat, n_classes, n_layers, 
            rel_names=graph.etypes, aggregate=aggregate,
            num_heads=num_heads, multihead_aggregate=multihead_aggregate,
        )
    elif model_name == "rgat2":
        model = RGAT2(
            in_feat, hidden_feat, n_classes, n_layers, 
            rel_names=graph.etypes, aggregate=aggregate,
            num_heads=num_heads, multihead_aggregate=multihead_aggregate,
            dropout=dropout,
        )
    elif model_name == "rsage":
        model = RSAGE(
            in_feat, hidden_feat, n_classes, n_layers, 
            rel_names=graph.etypes, aggregate=aggregate,
        )
    elif model_name == "rsage2":
        model = RSAGE2(
            in_feat, hidden_feat, n_classes, n_layers, 
            rel_names=graph.etypes, aggregate=aggregate,
            dropout=dropout,
        )
    else:
        raise NotImplementedError

    return model.to(device)
