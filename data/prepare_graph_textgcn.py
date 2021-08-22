import os
import logging
import sys
sys.path.append("../")
from typing import Tuple, List, Dict
from tqdm import tqdm

import torch.nn.functional as F
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
    D = D[:10]
    D = {did: id for id, did in enumerate(sorted(D))}
    doc_labels = utils.load_json(doc_label_path)
    
    D_content = [None] * len(D)
    D_label = [None] * len(D)
    D_feat = [None] * len(D)
    D_title = [None] * len(D)
    D_mask = {data_type: [None] * len(D) for data_type in ("train_mask", "val_mask", "test_mask")}
    count = 0
    for doc in utils.load_ndjson(doc_path, pname="Encode documents"):
        id = D.get(doc["id"], None)
        if id is not None:
            D_title[id] = doc["title"]
            D_content[id] = doc["content"].lower()
            D_label[id] = utils.get_onehot(doc["labels"], doc_labels)
            D_mask["train_mask"][id] = doc["is_train"]
            D_mask["val_mask"][id] = doc["is_dev"]
            D_mask["test_mask"][id] = doc["is_test"]
            count += 1
            if count == 10:
                break
            
    # edge processing
    D_content = utils.stem_text(D_content)
    DvsW, W = utils.get_tfidf_score(D_content)
    if os.path.exists(word2word_path):
        WvsW = utils.load(word2word_path)
    else:
        WvsW = utils.get_pmi(D_content, list(W.keys()), cache_dir)
        utils.dump(WvsW, word2word_path)
        
    # create heterogeneous graph
    num_nodes_dict = {"doc": len(D), "word": len(W)}
    graph = dgl.heterograph(
        data_dict={
            ("doc", "contain", "word"): DvsW.nonzero(),
            ("word", "in", "doc"): DvsW.transpose().nonzero(),
            ("word", "relate-to", "word"): WvsW.nonzero(),
        },
        num_nodes_dict=num_nodes_dict
    )
    encoder = utils.get_encoder()
    graph.nodes["word"].data["feat"] = torch.tensor(utils.get_bert_features(encoder, W.keys()), dtype=torch.float32)
    graph.nodes["doc"].data["feat"] = torch.tensor(utils.get_bert_features(encoder, D_title), dtype=torch.float32)
    graph.nodes["doc"].data["label"] = torch.tensor(D_label, dtype=torch.long)
    graph.nodes["doc"].data["train_mask"] = torch.tensor(D_mask["train_mask"], dtype=torch.bool)
    graph.nodes["doc"].data["val_mask"] = torch.tensor(D_mask["val_mask"], dtype=torch.bool)
    graph.nodes["doc"].data["test_mask"] = torch.tensor(D_mask["test_mask"], dtype=torch.bool)
    graph.edges["in"].data["weight"] = torch.from_numpy(DvsW[DvsW.nonzero()].transpose())
    graph.edges["contain"].data["weight"] = torch.from_numpy(DvsW[DvsW.nonzero()].transpose())
    graph.edges["relate-to"].data["weight"] = torch.from_numpy(WvsW[WvsW.nonzero()])
    graph = dgl.add_self_loop(graph, etype="relate-to")
    logging.info(graph)
    num_classes = len(utils.load_json(doc_label_path))

    return graph.to(device), D, W, num_classes
