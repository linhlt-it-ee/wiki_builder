import os
import json
import dgl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("../")
from argparse import ArgumentParser
from tqdm import tqdm, trange
from typing import Dict, Union, Tuple, List
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoTokenizer
from utils import text_util, file_util, model_util
from heterorgcn import HeteroRGCN
from metrics import compute_metrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prepare_graph(data_dir: str, pretrained_model_name: str = None, **kwargs) -> Union[dgl.DGLGraph, Dict[str, int]]:
    document_ids = file_util.load_json(os.path.join(data_dir, "document_ids.json"))
    entities_ids = file_util.load_json(os.path.join(data_dir, "entities_ids.json"))
    document_label_ids = file_util.load_json(os.path.join(data_dir, "document_label_ids.json"))
    graph = dgl.data.utils.load_graphs(os.path.join(data_dir, "graph.bin"))[0][0]
    num_documents, num_entities = len(document_ids), len(entities_ids)
    assert graph.num_nodes("document") == num_documents
    assert graph.num_nodes("concept") == num_entities

    model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    document_labels = [None] * num_documents
    document_masks = [None] * num_documents
    document_feats = [None] * num_documents
    with open(os.path.join(data_dir, "data.ndjson"), "r") as f:
        for line in tqdm(f, desc="Featurizing documents"):
            doc = json.loads(line)
            id = str(doc["doc_id"])
            if id in document_ids:
                doc_id = document_ids[id] 
                document_feats[doc_id] = model_util.get_text_embedding(model, tokenizer, doc["title"], max_length=64, device=device)
                document_labels[doc_id] = model_util.get_onehot_encoding(doc["labels"], document_label_ids)
                document_masks[doc_id] = doc["is_train"]
    graph.nodes["document"].data["feature"] = torch.tensor(document_feats, dtype=torch.float)
    graph.nodes["document"].data["label"] = torch.tensor(document_labels, dtype=torch.long)
    graph.nodes["document"].data["train_mask"] = torch.tensor(document_masks, dtype=torch.bool)
    del document_feats, document_labels, document_masks
    
    entities_labels = file_util.load_json(os.path.join(data_dir, "entities_labels.json"))
    entities_feats = [None] * num_entities
    for id in tqdm(entities_labels, desc="Featurizing entities"):
        if id in entities_ids:
            eid = entities_ids[id]
            title = entities_labels[id]
            entities_feats[eid] = model_util.get_text_embedding(model, tokenizer, title, device=device)
    graph.nodes["concept"].data["feature"] = torch.tensor(entities_feats, dtype=torch.float)
    del entities_feats

    return graph.to(device), document_label_ids


def prepare_model(graph: dgl.DGLGraph, model_name: str, out_dim: int, hidden_dim: int = 512, **kwargs) -> nn.Module:
    if model_name == "hrgcn":
        model_dims = {"in_dims": {}, "hidden_dim": hidden_dim, "out_dim": out_dim}
        for stype, etype, _ in graph.canonical_etypes:
            model_dims["in_dims"][etype] = graph.ndata["feature"][stype].shape[1]
        model = HeteroRGCN(graph, **model_dims)
    else:
        raise NotImplementedError(f"Canot initialize model {model_name}")
    return model.to(device)


def finetune(graph: dgl.DGLGraph, model: nn.Module, logger: SummaryWriter, threshold: float = 0.5, num_train_epochs: int = 500, lr: float = 1e-2, **kwargs) -> None:
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    labels = graph.nodes["document"].data["label"]
    train_mask = graph.nodes["document"].data["train_mask"]
    test_mask = ~train_mask

    pbar = trange(num_train_epochs, desc="Training")
    for epoch in pbar:
        logits = model(graph, "document")
        optimizer.zero_grad()
        loss = criterion(logits[train_mask], labels[train_mask].type_as(logits))
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

        logger.add_scalar("train/loss", loss.item(), epoch + 1)
        if (epoch + 1) % 100 == 0:
            test_logits, test_labels = logits[test_mask], labels[test_mask]
            eval_loss = criterion(test_logits, test_labels.type_as(test_logits)) 
            y_true = test_labels.detach().cpu().numpy()
            y_prob = torch.sigmoid(test_logits).detach().cpu().numpy()
            metrics = compute_metrics(y_true, y_prob, threshold=threshold)
            logger.add_scalar("eval/loss", loss.item(), epoch + 1)
            for k, v in metrics.items():
                logger.add_scalar(f"eval/{k}", v, epoch + 1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/", help="Data directory")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of HeteroRGCN")
    parser.add_argument("--num_train_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for classification")
    parser.add_argument("--model_name", type=str, default="hrgcn", help="Name of model, accept <hrgcn, han>")
    parser.add_argument("--pretrained_model_name", type=str, default="distilbert-base-uncased", help="Pretrained model used to get node features")
    parser.add_argument("--log_path", type=str, default="runs/log", help="Folder for Tensorboard logging")
    args = vars(parser.parse_args())
    print(args)

    graph, label_ids = prepare_graph(**args)
    print("****** Graph Information ******")
    print(graph)
    model = prepare_model(graph, out_dim=len(label_ids), **args)
    logger = SummaryWriter(args["log_path"])
    finetune(graph, model, logger=logger, **args)
