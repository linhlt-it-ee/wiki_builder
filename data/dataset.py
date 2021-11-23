from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import dgl
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PatentClassificationDataset:
    def __init__(self, predict_category: str):
        self.predict_category = predict_category
        self.nodes, self.edges = defaultdict(lambda: {}), defaultdict(lambda: {})
        self.node_classes = {}
        self.data_dict = {}

    def get_num_classes(self):
        return len(self.classes)

    def get_num_nodes_dict(self):
        num_nodes_dict = {}
        for ntype in self.nodes:
            num_nodes_dict[ntype] = len(self.nodes[ntype]["feat"])
        return num_nodes_dict

    def get_graph(self) -> dgl.DGLGraph:
        num_nodes_dict = self.get_num_nodes_dict()
        for ntype, num_nodes in num_nodes_dict.items():
            self.data_dict[(ntype, "loop-" + ntype, ntype)] = (range(num_nodes), range(num_nodes))
        graph = dgl.heterograph(data_dict=self.data_dict, num_nodes_dict=num_nodes_dict)
        for ntype in self.nodes:
            for dtype in self.nodes[ntype]:
                graph.nodes[ntype].data[dtype] = _convert2tensor(self.nodes[ntype][dtype])
        for etype in self.edges:
            for dtype in self.edges[etype]:
                if self.edges[etype][dtype] is None:
                    continue
                graph.edges[etype].data[dtype] = _convert2tensor(self.edges[etype][dtype])
        return graph.to(device)

    def add_labels(self, classes: List, labels) -> None:
        self.classes = classes
        self.nodes[self.predict_category]["label"] = labels

    def add_nodes(self, ntype: str, classes: List, feat, **kwargs) -> None:
        self.node_classes[ntype] = classes
        self.nodes[ntype]["feat"] = feat
        for key, value in kwargs.items():
            self.nodes[ntype][key] = value

    def add_edges(
        self, etype: Tuple[str, str, str], edges: Tuple[Iterable[int], Iterable[int]], weight=None
    ) -> None:
        assert len(edges[0]) == len(
            weight
        ), f"Edge connections {len(edges[0])} != weight {len(weight)}"
        src, etype, dst = etype
        self.data_dict[(src, etype, dst)] = edges
        self.edges[etype]["weight"] = weight

    def add_rev_edges(
        self, etype: Tuple[str, str, str], edges: Tuple[Iterable[int], Iterable[int]], weight=None
    ) -> None:
        assert len(edges[0]) == len(
            weight
        ), f"Edge connections {len(edges[0])} != weight {len(weight)}"
        src, etype, dst = etype
        self.data_dict[(dst, "rev-" + etype, src)] = edges[::-1]
        self.edges["rev-" + etype]["weight"] = weight

def _convert2tensor(array):
    return torch.tensor(array)
