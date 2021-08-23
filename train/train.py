import os
from tqdm import trange

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dgl import DGLGraph
from .strategy import *
from .metrics import compute_metrics


def train(model: nn.Module, graph: DGLGraph, target_node: str, lr: float, epochs: int, threshold: float = 0.5, strategy_name: str = "lc", writer: SummaryWriter = None, exp_name: str = "test"):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : lr * 0.9 ** (epoch - 1))
    train_mask = graph.nodes[target_node].data["train_mask"]
    val_mask = graph.nodes[target_node].data["val_mask"]
    test_mask = graph.nodes[target_node].data["test_mask"]
    labels = graph.nodes[target_node].data["label"]

    pbar = trange(epochs, desc="Training")
    strategy = prepare_strategy(strategy_name, train_mask)
    query_train_mask = strategy.init_random_mask()
    for epoch in pbar:
        # train step
        optimizer.zero_grad()
        logits = model(graph, graph.ndata["feat"], target_node, edge_weight=graph.edata["weight"])
        loss = criterion(logits[query_train_mask], labels[query_train_mask].type_as(logits))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item(), lr="{:.1e}".format(lr_scheduler.get_last_lr()[0]))
        writer.add_scalar("loss/train", loss.item(), epoch)
        # choose next samples
        probs = torch.sigmoid(logits.detach())
        query_train_mask = strategy.query(probs)
        if (epoch + 1) % 100 == 0:
            val_scores = eval(labels, logits, val_mask, threshold=threshold)
            print(val_scores)
            for k, v in val_scores.items():
                writer.add_scalar(f"{k}/val", v, epoch)
            lr_scheduler.step()

    print("**** TEST ****")
    test_scores = eval(labels, logits, test_mask, threshold=threshold)
    print(test_scores)
    for k, v in test_scores.items():
        writer.add_scalar(f"{k}/test", v, epoch)

    os.makedirs("results", exist_ok=True)
    report = pd.Series(test_scores).sort_index() * 100
    report.to_csv(os.path.join("results", f"{exp_name}.csv"), float_format="%.2f")

def eval(labels, logits, mask: torch.BoolTensor, threshold: float = 0.5):
    y_true = labels[mask].detach().cpu()
    y_prob = torch.sigmoid(logits[mask]).detach().cpu()
    return compute_metrics(y_true, y_prob, threshold=threshold)
