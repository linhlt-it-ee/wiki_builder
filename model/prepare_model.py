import torch
import torch.nn as nn
from dgl import DGLGraph

from .rgat import *
from .rgcn import *
from .rsage import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dict = {"rgcn": RGCN, "rsage": RSAGE, "rgat": RGAT}


def prepare_model(
    model_name: str,
    graph: DGLGraph,
    n_classes: int,
    hidden_feat: int = 128,
    n_layers: int = 2,
    aggregate: str = "sum",
    dropout: float = 0.2,
    num_heads: int = 2,
    lang: str = "en",
) -> nn.Module:
    params = {
        "in_feat": 512 if lang == "ja" else 768,
        "hidden_feat": hidden_feat,
        "n_classes": n_classes,
        "n_layers": n_layers,
        "rel_names": graph.etypes,
        "aggregate": aggregate,
        "dropout": dropout,
    }
    if model_name == "rgat":
        params.update(
            {
                "num_heads": num_heads,
            }
        )
    model = model_dict[model_name](**params)
    return model.to(device)
