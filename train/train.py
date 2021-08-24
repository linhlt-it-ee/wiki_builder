import os
import logging
from tqdm import trange

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dgl import DGLGraph

from .strategy import *
from .metrics import compute_metrics

def run(
        model: nn.Module, graph: DGLGraph, target_node: str, 
        lr: float, epochs: int, threshold: float = 0.5, strategy_name: str = "lc", 
        writer: SummaryWriter = None, exp_name: str = "test"
    ):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it : lr * 0.9 ** (it - 1))
    train_mask = graph.nodes[target_node].data["train_mask"]
    val_mask = graph.nodes[target_node].data["val_mask"]
    test_mask = graph.nodes[target_node].data["test_mask"]

    use_active_learning = True if strategy_name is not None else False
    if use_active_learning:
        n_rounds = 100
        round_epoch = epochs // n_rounds
        strategy = prepare_strategy(strategy_name, train_mask, n_rounds)
        query_train_mask = strategy.random_mask()
        update_freq = min(10, round_epoch)
    else:
        n_rounds = 1
        round_epoch = epochs
        query_train_mask = train_mask
        update_freq = 100

    iteration = 0
    for round in range(n_rounds):
        logging.info(f"START ROUND {round + 1}")
        pbar = trange(round_epoch, desc="Training")
        for epoch in pbar:
            iteration += 1
            logits, loss = train(model, graph, target_node, query_train_mask, criterion, optimizer)
            pbar.set_postfix(loss=loss.item(), lr="{:.1e}".format(lr_scheduler.get_last_lr()[0]))
            writer.add_scalar("loss/train", loss.item(), epoch)
            # validation
            if epoch == 0 or (epoch + 1) % update_freq == 0:
                train_scores = predict(model, graph, target_node, train_mask, threshold=threshold)
                for k, v in train_scores.items():
                    writer.add_scalar(f"{k}/train", v, iteration)
                val_scores = predict(model, graph, target_node, val_mask, threshold=threshold)
                print(val_scores)
                for k, v in val_scores.items():
                    writer.add_scalar(f"{k}/val", v, iteration)
            # update lr every 100 iterations
            if iteration % 100 == 0:
                lr_scheduler.step()

        # choose next samples for the next round (except the last)
        if use_active_learning and round != (n_rounds - 1):
            logits = logits.detach().cpu()
            probs = torch.sigmoid(logits)
            query_train_mask = strategy.query(probs, logits)

    # inference at the last round
    print("**** TEST ****")
    test_scores = predict(model, graph, target_node, test_mask, threshold=threshold)
    print(test_scores)
    for k, v in test_scores.items():
        writer.add_scalar(f"{k}/test", v, 0)
    # save classification report
    os.makedirs("results", exist_ok=True)
    report = pd.Series(test_scores).sort_index() * 100
    report.to_csv(os.path.join("results", f"{exp_name}.csv"), float_format="%.2f")

def train(model, graph, target_node, mask, criterion, optimizer):
    labels = graph.nodes[target_node].data["label"]
    model.train()
    optimizer.zero_grad()
    logits = model(graph, graph.ndata["feat"], target_node, graph.edata["weight"])
    loss = criterion(logits[mask], labels[mask].type_as(logits))
    loss.backward()
    optimizer.step()
    return logits, loss

def predict(model, graph, target_node, mask, threshold: float = 0.5):
    labels = graph.nodes[target_node].data["label"]
    model.eval()
    with torch.no_grad():
        logits = model(graph, graph.ndata["feat"], target_node, graph.edata["weight"])
        scores = eval(labels, logits, mask, threshold=threshold)
    return scores

def eval(labels, logits, mask: torch.BoolTensor, threshold: float = 0.5):
    y_true = labels[mask].cpu()
    y_prob = torch.sigmoid(logits[mask]).cpu()
    return compute_metrics(y_true, y_prob, threshold=threshold)
