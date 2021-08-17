import os
import logging
import sys
sys.path.append("../")
from typing import Tuple, Iterable, List, Dict
from tqdm import tqdm
from collections import defaultdict

import dgl
import networkx as nx
import torch
from dgl import DGLGraph
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def prepare_graph(data_dir: str, par_num: List[int]) -> Tuple[DGLGraph, Dict, Dict, int]:
    doc_path = os.path.join(data_dir, "data.ndjson")
    doc_label_path = os.path.join(data_dir, "doc_label_encoder.json")
    cache_dir = os.path.join(data_dir, "cache")
    doc_mention_path = os.path.join(cache_dir, "doc_mention.json")
    mention_concept_path = os.path.join(cache_dir, "concept_links.json")
    concept_path = os.path.join(cache_dir, "concept_labels.json")
    # saved path
    doc_feat_path = os.path.join(cache_dir, "doc_feat.pck")
    concept_feat_path = os.path.join(cache_dir, "concept_feat.pck")

    # broadcast relation between doc - mention - concept, then prune by `par_num`
    D, C, DvsC, CvsC = _broadcast(doc_mention_path, mention_concept_path, par_num)
    if not os.path.exists(doc_feat_path):
        D_info, C_info = _embed_node(doc_path, doc_label_path, concept_path, D, C)
        utils.dump(D_info, doc_feat_path)
        utils.dump(C_info, concept_feat_path)
    else:
        D_info = utils.load(doc_feat_path)
        C_info = utils.load(concept_feat_path)

    # create heterogenous graph
    num_nodes_dict = {"doc": len(D), "concept": len(C)}
    graph = dgl.heterograph(
        data_dict={
            ("doc", "contain", "concept"): DvsC,
            ("concept", "in", "doc"): (DvsC[1], DvsC[0]),
            ("concept", "belong", "concept"): CvsC,
            # ("concept", "elaborate", "concept"): (CvsC[1], CvsC[0]),
        },
        num_nodes_dict=num_nodes_dict
    )
    graph.nodes["concept"].data["feat"] = torch.tensor(C_info[0], dtype=torch.float32)
    graph.nodes["doc"].data["feat"] = torch.tensor(D_info[0], dtype=torch.float32)
    graph.nodes["doc"].data["label"] = torch.tensor(D_info[1], dtype=torch.long)
    graph.nodes["doc"].data["train_mask"] = torch.tensor(D_info[2]["train_mask"], dtype=torch.bool)
    graph.nodes["doc"].data["val_mask"] = torch.tensor(D_info[2]["val_mask"], dtype=torch.bool)
    graph.nodes["doc"].data["test_mask"] = torch.tensor(D_info[2]["test_mask"], dtype=torch.bool)
    logging.info(graph)
    num_classes = len(utils.load_json(doc_label_path))
    graph = dgl.add_self_loop(graph, etype="belong")

    return graph.to(device), D, C, num_classes

def _broadcast(doc_mention_path: str, mention_concept_path: str, par_num: List[int]):
    doc_mention = utils.load_json(doc_mention_path)
    mention_concept = utils.load_json(mention_concept_path)
    mention_ids = defaultdict(set)
    # mapping between name_mentions (text) and name_mentions IDs
    for mid, mention_info in mention_concept.items():
        for label in mention_info["name_mention"]:
            mention_ids[label].add(mid)

    # create concept graph
    CvsC_graph = nx.DiGraph()
    for x in mention_concept.values():
        for parent in x["parents"]:
            path = parent["path"].split(" >> ")
            nx.add_path(CvsC_graph, path)

    C = set()
    children = mention_concept.keys()
    for level, parlevel_num in enumerate(par_num, start=1):
        cnt = defaultdict(lambda : 0)
        for child in children:
            if child not in CvsC_graph.nodes:  # ignore parent nodes of discarded node
                continue
            for node, parents in nx.bfs_successors(CvsC_graph, source=child, depth_limit=1):
                for par in parents:
                    if par not in C:    # handle duplicate nodes
                        cnt[par] += 1
        children = sorted(cnt, key=cnt.get, reverse=True)[:parlevel_num]
        logging.info(f"Extracting {parlevel_num} parents level {level} with most children, got {len(children)}")
        C.update(children)
    CvsC_graph = nx.DiGraph(CvsC_graph.subgraph(C))
    logging.info(f"CvsC graph: {CvsC_graph}")
    logging.info(f"C size: {len(C)}")
    
    # create document graph
    DvsC_graph = nx.DiGraph()
    D = doc_mention.keys()
    for did, mentions in doc_mention.items():
        DvsC_graph.add_node(did)
        # traverse meanings of a name_mention in a document
        for label in mentions:
            for mid in mention_ids[label]:
                # consider concept level 1 of each meaning
                related_concepts = mention_concept[mid]["parents"]
                for concept in mention_concept[mid]["parents"]:
                    if concept["level"] == 1 and concept["id"] in CvsC_graph.nodes:
                        DvsC_graph.add_edge(did, concept["id"])
    logging.info(f"DvsC graph: {DvsC_graph}")
    logging.info(f"D size: {len(D)}")

    # flatten
    D = {did: id for id, did in enumerate(sorted(D))}
    C = {cid: id for id, cid in enumerate(sorted(C))}
    DvsC = ([], [])
    CvsC = ([], [])
    for u, v in DvsC_graph.edges:
        DvsC[0].append(D[u])
        DvsC[1].append(C[v])
    for u, v in CvsC_graph.edges:
        CvsC[0].append(C[u])
        CvsC[1].append(C[v])
    return D, C, DvsC, CvsC

def _embed_node(doc_path: str, doc_label_path: str, concept_path: str, D: Dict[str, int], C: Dict[str, int], pretrained_node_encoder: str = "distilbert-base-uncased"):
    D_feat = [None] * len(D)
    D_label = [None] * len(D)
    D_mask = {data_type: [None] * len(D) for data_type in ("train_mask", "val_mask", "test_mask")}
    C_feat = [None] * len(C)
    doc_labels = utils.load_json(doc_label_path)
    encoder = utils.get_encoder(pretrained_node_encoder)
    print(D)
    for doc in utils.load_ndjson(doc_path, pname="Encode documents"):
        id = D.get(doc["id"], None)
        print(id)
        exit()
        if id is not None:
            D_feat[id] = utils.get_text_embedding(encoder, doc["title"]).detach().cpu().numpy()
            D_label[id] = utils.get_onehot(doc["labels"], doc_labels)
            D_mask["train_mask"][id] = doc["is_train"]
            D_mask["val_mask"][id] = doc["is_dev"]
            D_mask["test_mask"][id] = doc["is_test"]

    concepts = utils.load_json(concept_path)
    for cid, label in tqdm(concepts.items(), "Encode concepts"):
        id = C.get(cid, None)
        if id is not None:
            C_feat[id] = utils.get_text_embedding(encoder, label).detach().cpu().numpy()

    return (D_feat, D_label, D_mask), (C_feat,)
