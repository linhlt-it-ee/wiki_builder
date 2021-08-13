import torch
import torch.nn as nn
from dgl import DGLGraph
from .gcn import *
from .rgcn import *
from .gat import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_model(model_name: str, graph: DGLGraph, n_classes: int, hidden_feat: int, n_layers: int) -> nn.Module:
    if model_name == "rgcn":
        model = RGCN(graph, hidden_feat, n_classes, n_layers)
    elif model_name == "gcn":
        model = GCN(768, hidden_feat, n_classes, n_layers)
    else:
        raise NotImplementedError

    return model.to(device)
