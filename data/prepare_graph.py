import os
import logging
import sys
sys.path.append("../")
from typing import Tuple, List, Dict, Callable
from tqdm import tqdm
from collections import defaultdict

import dgl
import numpy as np
import networkx as nx
import torch
from dgl import DGLGraph
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_graph(data_dir: str, feature_type: str, par_num: List[int] = None) -> Tuple[DGLGraph, Dict, Dict, int]:
    doc_path = os.path.join(data_dir, "data.ndjson")
    doc_label_path = os.path.join(data_dir, "doc_label_encoder.json")
    cache_dir = os.path.join(data_dir, "cache")
    doc_mention_path = os.path.join(cache_dir, "doc_mention.json")
    mention_concept_path = os.path.join(cache_dir, "concept_links.json")
    concept_path = os.path.join(cache_dir, "concept_labels.json")
    # saved path
    word2word_path = os.path.join(cache_dir, "word2word.pck")
    doc_info_path = os.path.join(cache_dir, "doc_info.pck")
    doc_concept_info_path = os.path.join(cache_dir, "doc_concept_info.pck")

    # create heterogeneous graph
    num_nodes_dict, data_dict = {}, {}
    nodes, edges = defaultdict(lambda : {}), defaultdict(lambda : {})
    
    D, D_feat, D_label, D_mask, doc_content = _cache_to_path(doc_info_path, get_document, doc_path, doc_label_path)
    num_nodes_dict["doc"] = len(D)
    nodes["doc"]["train_mask"] = torch.tensor(D_mask["train_mask"], dtype=torch.bool)
    nodes["doc"]["val_mask"] = torch.tensor(D_mask["val_mask"], dtype=torch.bool)
    nodes["doc"]["test_mask"] = torch.tensor(D_mask["test_mask"], dtype=torch.bool)
    nodes["doc"]["label"] = torch.tensor(D_label, dtype=torch.long)
    nodes["doc"]["feat"] = torch.tensor(D_feat, dtype=torch.float32)
    if feature_type in ("ours", "mixed"):
        C, C_feat, DvsC, CvsC = _cache_to_path(doc_concept_info_path, get_document_concept, D, concept_path, doc_mention_path, mention_concept_path, par_num)
        num_nodes_dict["concept"] = len(C)
        data_dict.update({
            ("doc", "include-concept", "concept"): DvsC,
            ("concept", "concept-in", "doc"): (DvsC[1], DvsC[0]),
            ("concept", "belong", "concept"): CvsC,
        })
        nodes["concept"]["feat"] = torch.tensor(C_feat, dtype=torch.float32)
    if feature_type in ("textgcn", "mixed"):
        W, W_feat, DvsW, WvsW, DvsW_weight, WvsW_weight = get_document_word(doc_content, word2word_path)
        WvsD_weight = DvsW_weight.transpose()
        WvsD = (DvsW[1], DvsW[0])
        num_nodes_dict["word"] = len(W)
        data_dict.update({
            ("doc", "include-word", "word"): DvsW,
            ("word", "word-in", "doc"): WvsD,
            ("word", "relate-to", "word"): WvsW,
        })
        nodes["word"]["feat"] = torch.tensor(W_feat, dtype=torch.float32)
        edges["include-word"]["weight"] = torch.tensor(np.asarray(DvsW_weight[DvsW]).squeeze(), dtype=torch.float32)
        edges["word-in"]["weight"] = torch.tensor(np.asarray(WvsD_weight[WvsD]).squeeze(), dtype=torch.float32)
        edges["relate-to"]["weight"] = torch.tensor(np.asarray(WvsW_weight[WvsW]).squeeze(), dtype=torch.float32)

    graph = dgl.heterograph(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
    num_classes = len(utils.load_json(doc_label_path))
    for ntype in nodes:
        for dtype in nodes[ntype]:
            graph.nodes[ntype].data[dtype] = nodes[ntype][dtype]
    for etype in edges:
        for dtype in edges[etype]:
            graph.edges[etype].data[dtype] = edges[etype][dtype]

    logging.info(graph)
    return graph.to(device), D, num_classes

def get_document(doc_path: str, doc_label_path: str):
    doc_labels = utils.load_json(doc_label_path)
    docs = [doc for doc in utils.load_ndjson(doc_path)]
    D = {did: id for id, did in enumerate(sorted(doc["id"] for doc in docs))}
    doc_content = [None] * len(D)
    D_feat = [None] * len(D)
    D_label = [None] * len(D)
    D_feat = [None] * len(D)
    D_mask = {data_type: [None] * len(D) for data_type in ("train_mask", "val_mask", "test_mask")}

    for doc in tqdm(docs, desc="Encoding documents"):
        id = D.get(doc["id"], None)
        if id is not None:
            doc_content[id] = doc["content"]
            D_feat[id] = doc["title"]
            D_label[id] = utils.get_onehot(doc["labels"], doc_labels)
            D_mask["train_mask"][id] = doc["is_train"]
            D_mask["val_mask"][id] = doc["is_dev"]
            D_mask["test_mask"][id] = doc["is_test"]
    D_feat = _encode_text(D_feat)

    return D, D_feat, D_label, D_mask, doc_content

def get_document_concept(D: Dict, concept_path: str, doc_mention_path: str, mention_concept_path: str, par_num: List[int], return_graph: bool = False):
    doc_mention = utils.load_json(doc_mention_path)
    valid_mentions = set()
    for x in doc_mention.values():
        valid_mentions.update(x)

    # create concept graph
    mention_concept = utils.load_json(mention_concept_path)
    CvsC_graph = nx.DiGraph()
    for x in mention_concept.values():
        if x["label"] in valid_mentions:
            for parent in x["parents"]:
                path = parent["path"].split(" >> ")
                nx.add_path(CvsC_graph, path)

    C = set()
    children = mention_concept.keys()
    C1 = None
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
        if level == 1:
            C1 = children
    CvsC_graph = nx.DiGraph(CvsC_graph.subgraph(C))
    logging.info(f"CvsC graph: {CvsC_graph}")
    logging.info(f"C size: {len(C)}")
    if return_graph:
        return CvsC_graph, C1
    
    # mapping name_mentions and their IDs to connect a document and its mentioned concepts
    mention_ids = defaultdict(set)
    for mid, mention_info in mention_concept.items():
        for label in set(x.lower() for x in mention_info["name_mention"]):  # match normalized nouns
            mention_ids[label].add(mid)

    DvsC_graph = nx.DiGraph()
    for did, mentions in doc_mention.items():
        DvsC_graph.add_node(did)
        # traverse meanings of a name_mention in a document
        for label in mentions:
            for mid in mention_ids[label]:
                # consider concept level 1 of each meaning
                for concept in mention_concept[mid]["parents"]:
                    if concept["level"] == 1 and concept["id"] in CvsC_graph.nodes:
                        DvsC_graph.add_edge(did, concept["id"])
    logging.info(f"DvsC graph: {DvsC_graph}")
    logging.info(f"D size: {len(D)}")

    C = {cid: id for id, cid in enumerate(sorted(C))}
    concepts = utils.load_json(concept_path)
    C_feat = [None] * len(C)
    for cid, label in tqdm(concepts.items(), "Encoding concepts"):
        id = C.get(cid, None)
        if id is not None:
            C_feat[id] = label
    C_feat = _encode_text(C_feat)

    # flatten
    DvsC = ([], [])
    CvsC = ([], [])
    for u, v in DvsC_graph.edges:
        DvsC[0].append(D[u])
        DvsC[1].append(C[v])
    for u, v in CvsC_graph.edges:
        CvsC[0].append(C[u])
        CvsC[1].append(C[v])
    return C, C_feat, DvsC, CvsC

def get_document_word(doc_content: List[str], word2word_path: str, vocab_size: int = 5000):
    if os.path.exists(word2word_path):
        W, DvsW_weight, WvsW_weight = utils.load(word2word_path)
    else:
        doc_content = utils.stem_text(doc_content)
        DvsW_weight, W = utils.get_tfidf_score(doc_content)
        WvsW_weight = utils.get_pmi(doc_content, vocab=list(W.keys()))
        utils.dump((W, DvsW_weight, WvsW_weight), word2word_path)

    W = {k: W[k] for k in list(W.keys())[:vocab_size]}
    DvsW_weight = DvsW_weight[:, :vocab_size]
    DvsW = DvsW_weight.nonzero()
    WvsW_weight = WvsW_weight[:vocab_size, :vocab_size]
    WvsW = WvsW_weight.nonzero()
    W_feat = _encode_text(list(W.keys()))

    return W, W_feat, DvsW, WvsW, DvsW_weight, WvsW_weight

def _encode_text(text: List[str], text_encoder: str = "distilbert-base-uncased"):
    encoder = utils.get_encoder(pretrained_model_name=text_encoder)
    return utils.get_bert_features(encoder, text)

def _cache_to_path(path: str, fn: Callable, *args, **kwargs):
    if os.path.exists(path):
        res = utils.load(path)
    else:
        res = fn(*args, **kwargs)
        utils.dump(res, path)
    return res
