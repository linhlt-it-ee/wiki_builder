import os
from collections import defaultdict
from tqdm import trange
from typing import Dict

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from dgl import DGLGraph

from .strategy import *
from .metrics import compute_metrics


def train(
        model: nn.Module, graph, dataset,
        strategy_name: str = None, 
        lr: float = 0.05, epochs: int = 500, threshold: float = 0.5, 
        run = None,
    ):
    target_node = dataset.predict_category
    graph = dataset.get_graph()

    train_mask = graph.nodes[target_node].data["train_mask"]
    val_mask = graph.nodes[target_node].data["val_mask"]
    test_mask = graph.nodes[target_node].data["test_mask"]
    labels = graph.nodes[target_node].data["label"]
    inputs = graph.ndata["feat"]
    edge_weight = {} if not "weight" in graph.edata else {k[1]: v for k, v in graph.edata["weight"].items()}

    # settings
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it : lr * 0.9 ** (it - 1))
    use_active_learning = True if strategy_name is not None else False
    if use_active_learning:
        n_rounds = 100
        round_epoch = epochs // n_rounds
        strategy = prepare_strategy(strategy_name, train_mask.cpu(), n_rounds)
        query_train_mask = strategy.random_mask()
        update_freq = round_epoch
    else:
        n_rounds = 1
        round_epoch = epochs
        query_train_mask = train_mask
        update_freq = 100
    
    # validation at the first round
    iteration = 0
    train_scores = predict(
        model, graph, target_node, inputs, edge_weight, labels, train_mask, threshold=threshold
    )
    val_scores = predict(
        model, graph, target_node, inputs, edge_weight, labels, val_mask, threshold=threshold
    )
    log(run, train_scores, type="train")
    log(run, val_scores, type="val")

    # training
    init_state = model.state_dict()
    for rnd in range(n_rounds):
        num_samples = query_train_mask.sum()
        print(f"===== START ROUND {rnd + 1}: {num_samples} samples =====")
        model.load_state_dict(init_state)
        pbar = trange(round_epoch, desc="Training")
        for epoch in pbar:
            iteration += 1
            model.train()
            optimizer.zero_grad()
            features, logits = model(
                graph, inputs, target_node, edge_weight=edge_weight, return_features=True
            )
            loss = nn.BCEWithLogitsLoss()(
                logits[query_train_mask], labels[query_train_mask].type_as(logits)
            )

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr="{:.1e}".format(lr_scheduler.get_last_lr()[0]))
            log(run, {"loss": loss.item()}, type="train")

            # validation
            if (epoch + 1) % update_freq == 0:
                train_scores = predict(
                    model, graph, target_node, inputs, edge_weight, labels, train_mask, threshold=threshold
                )
                val_scores = predict(
                    model, graph, target_node, inputs, edge_weight, labels, val_mask, threshold=threshold
                )
                log(run, train_scores, type="train")
                log(run, val_scores, type="val")
                print(val_scores)

            if iteration % 100 == 0:
                lr_scheduler.step()

        # choose next samples for the next round (except the last)
        if use_active_learning and rnd != (n_rounds - 1):
            probs = torch.sigmoid(logits)
            query_train_mask = strategy.query(probs.detach().cpu(), labels.cpu(), features=features.detach().cpu())

    # inference at the last round
    print("**** TEST ****")
    test_scores = predict(
        model, graph, target_node, inputs, edge_weight, labels, test_mask, threshold=threshold
    )
    log(run, test_scores, type="test")
    print(test_scores)
    os.makedirs("results", exist_ok=True)
    report = pd.Series(test_scores).sort_index() * 100
    report.to_csv("results.csv", float_format="%.2f")
    
    return model

def predict(
        model: nn.Module, graph: DGLGraph, target_node: str,
        inputs: torch.Tensor, edge_weight: torch.Tensor,
        labels: torch.Tensor, mask: torch.BoolTensor,
        threshold: float = 0.5
    ):
    model.eval()
    with torch.no_grad():
        logits = model(graph, inputs, target_node, edge_weight=edge_weight)
        y_true = labels[mask].cpu()
        y_prob = torch.sigmoid(logits[mask]).cpu()
        scores = compute_metrics(y_true, y_prob, threshold=threshold)
    return scores

def log(run, scores: Dict, type: str = "train"):
    for k, v in scores.items():
        run[f"{type}/{k}"].log(v)
