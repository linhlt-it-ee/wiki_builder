from collections import defaultdict
from typing import Dict, Iterable, Tuple

import torch
import dgl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatentClassificationDataset:
    def __init__(self, predict_category: str):
        self.predict_category = predict_category
        self.node_encoder = {}
        self.num_nodes_dict = {}
        self.data_dict = {}
        self.nodes, self.edges = defaultdict(lambda: {}), defaultdict(lambda: {})

    def add_labels(self, labels, encoder: Dict) -> None:
        self.label_encoder = encoder
        self.nodes[self.predict_category]["label"] = labels
        self.num_classes = labels.shape[1]

    def add_nodes(self, ntype: str, encoder: Dict[str, int], feat, **kwargs) -> None:
        self.num_nodes_dict[ntype] = len(encoder)
        self.node_encoder[ntype] = encoder
        self.nodes[ntype]["feat"] = feat
        for key, value in kwargs.items():
            self.nodes[ntype][key] = value

    def add_edges(self, etype: Tuple[str, str, str], edges: Tuple[Iterable[int], Iterable[int]], weight = None) -> None:
        assert len(edges[0]) == len(weight), f"Edge connections {len(edges[0])} != weight {len(weight)}"
        src, etype, dst = etype
        self.data_dict[(src, etype, dst)] = edges
        self.edges[etype]["weight"] = weight
        if src != dst:
            self.data_dict[(dst, "rev-" + etype, src)] = edges[::-1]
            self.edges["rev-" + etype]["weight"] = weight

    def get_graph(self) -> dgl.DGLGraph:
        for ntype, num_nodes in self.num_nodes_dict.items():
            self.data_dict[(ntype, "loop-" + ntype, ntype)] = (range(num_nodes), range(num_nodes))
        graph = dgl.heterograph(data_dict=self.data_dict, num_nodes_dict=self.num_nodes_dict)
        for ntype in self.nodes:
            for dtype in self.nodes[ntype]:
                graph.nodes[ntype].data[dtype] = _convert2tensor(self.nodes[ntype][dtype])
        for etype in self.edges:
            for dtype in self.edges[etype]:
                if self.edges[etype][dtype] is None:
                    continue
                graph.edges[etype].data[dtype] = _convert2tensor(self.edges[etype][dtype])
        return graph.to(device)

def _convert2tensor(array):
    return torch.tensor(array)
