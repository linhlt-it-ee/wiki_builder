import os
import logging
import sys
sys.path.append("../")
from typing import Tuple, List, Dict
from tqdm import tqdm

import torch.nn.functional as F
import numpy as np
import dgl
import torch
from dgl import DGLGraph
import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_graph_textgcn(data_dir: str) -> Tuple[DGLGraph, Dict, Dict, int]:
    doc_path = os.path.join(data_dir, "data.ndjson")
    doc_label_path = os.path.join(data_dir, "doc_label_encoder.json")
    cache_dir = os.path.join(data_dir, "cache_textgcn")
    os.makedirs(cache_dir, exist_ok=True)
    # saved path
    word2word_path = os.path.join(cache_dir, "word2word.pck")

    # doc processing
    D = [doc["id"] for doc in utils.load_ndjson(doc_path)]
    D = {did: id for id, did in enumerate(sorted(D))}
    doc_labels = utils.load_json(doc_label_path)
    
    D_content = [None] * len(D)
    D_label = [None] * len(D)
    D_feat = [None] * len(D)
    D_title = [None] * len(D)
    D_mask = {data_type: [None] * len(D) for data_type in ("train_mask", "val_mask", "test_mask")}
    for doc in utils.load_ndjson(doc_path, pname="Encode documents"):
        id = D.get(doc["id"], None)
        if id is not None:
            D_title[id] = doc["title"]
            D_content[id] = doc["content"].lower()
            D_label[id] = utils.get_onehot(doc["labels"], doc_labels)
            D_mask["train_mask"][id] = doc["is_train"]
            D_mask["val_mask"][id] = doc["is_dev"]
            D_mask["test_mask"][id] = doc["is_test"]
            
    # edge processing
    D_content = utils.stem_text(D_content)
    DvsW_weight, W = utils.get_tfidf_score(D_content)
    WvsD_weight = DvsW_weight.transpose()
    if os.path.exists(word2word_path):
        WvsW_weight = utils.load(word2word_path)
    else:
        WvsW_weight = utils.get_pmi(D_content, list(W.keys()), cache_dir)
        utils.dump(WvsW_weight, word2word_path)
        
    # create heterogeneous graph
    DvsW = DvsW_weight.nonzero()
    WvsD = WvsD_weight.nonzero()
    WvsW = WvsW_weight.nonzero()
    num_nodes_dict = {"doc": len(D), "word": len(W)}
    graph = dgl.heterograph(
        data_dict={
            ("doc", "contain", "word"): DvsW,
            ("word", "in", "doc"): WvsD,
            ("word", "relate-to", "word"): WvsW,
        },
        num_nodes_dict=num_nodes_dict
    )
    graph.edges["in"].data["weight"] = torch.tensor(np.asarray(DvsW_weight[DvsW]).squeeze(), dtype=torch.float32)
    graph.edges["contain"].data["weight"] = torch.tensor(np.asarray(WvsD_weight[WvsD]).squeeze(), dtype=torch.float32)
    graph.edges["relate-to"].data["weight"] = torch.tensor(np.asarray(WvsW_weight[WvsW]).squeeze(), dtype=torch.float32)
    encoder = utils.get_encoder()
    graph.nodes["word"].data["feat"] = torch.tensor(utils.get_bert_features(encoder, W.keys()), dtype=torch.float32)
    graph.nodes["doc"].data["feat"] = torch.tensor(utils.get_bert_features(encoder, D_title), dtype=torch.float32)
    graph.nodes["doc"].data["label"] = torch.tensor(D_label, dtype=torch.long)
    graph.nodes["doc"].data["train_mask"] = torch.tensor(D_mask["train_mask"], dtype=torch.bool)
    graph.nodes["doc"].data["val_mask"] = torch.tensor(D_mask["val_mask"], dtype=torch.bool)
    graph.nodes["doc"].data["test_mask"] = torch.tensor(D_mask["test_mask"], dtype=torch.bool)
    graph = dgl.add_self_loop(graph, etype="relate-to")
    logging.info(graph)
    num_classes = len(utils.load_json(doc_label_path))

    return graph.to(device), D, W, num_classes
