import torch
import torch.nn as nn
from dgl import DGLGraph
from .rgcn import *
from .rgat import *
from .heterorgcn import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_model(model_name: str, graph: DGLGraph, n_classes: int, hidden_feat: int, n_layers: int) -> nn.Module:
    if model_name == "rgcn":
        model = RGCN(768, hidden_feat, n_classes, n_layers, rel_names=graph.etypes)
    elif model_name == "rgat":
        model = RGAT(768, hidden_feat, n_classes, n_layers, rel_names=graph.etypes)
    elif model_name == "heterorgcn":
        model = HeteroRGCN(768, hidden_feat, n_classes, n_layers, rel_names=graph.etypes)
    else:
        raise NotImplementedError

    return model.to(device)
