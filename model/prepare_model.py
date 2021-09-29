import torch
import torch.nn as nn
from dgl import DGLGraph
from .rgcn import *
from .rsage import *
from .rgat import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dict = {"rgcn": RGCN, "rsage": RSAGE, "rgat": RGAT}

def prepare_model(
    model_name: str, graph: DGLGraph, n_classes: int, 
    hidden_feat: int = 128, n_layers: int = 2, aggregate: str = "sum", 
    num_heads: int = 2, multihead_aggregate: str = "concat", dropout: float = 0.2,
) -> nn.Module:
    params = {
        "in_feat": 768,
        "hidden_feat": hidden_feat,
        "n_classes": n_classes,
        "n_layers": n_layers,
        "rel_names": graph.etypes,
        "aggregate": aggregate,
        "dropout": dropout,
    }
    if model_name == "rgat":
        params.update({
        "num_heads": num_heads,
        "multihead_aggregate": multihead_aggregate,
    })
    model = model_dict[model_name](**params)
    return model.to(device)
