import os
import logging
from collections import defaultdict
from tqdm import trange
from typing import Dict

import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dgl import DGLGraph

from .strategy import *
from .metrics import compute_metrics

def run(
        model: nn.Module, graph: DGLGraph, dataset, 
        lr: float, epochs: int = 500, threshold: float = 0.5, strategy_name: str = None, 
        writer: SummaryWriter = None, exp_name: str = "test"
    ):
    target_node = dataset.predict_category
    train_mask = graph.nodes[target_node].data["train_mask"]
    val_mask = graph.nodes[target_node].data["val_mask"]
    test_mask = graph.nodes[target_node].data["test_mask"]
    labels = graph.nodes[target_node].data["label"]
    inputs = graph.ndata["feat"]

    edge_weight = {} if not "weight" in graph.edata else {k[1]: v for k, v in graph.edata["weight"].items()}
    hier_indices = []
    for i in range(4):
        hier_dict = defaultdict(list)
        for k, v in dataset.label_encoder.items():
            hier_dict[k[:i+1]].append(v)
        hier_indices.append(hier_dict)

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
    log(writer, train_scores, iteration, type="train")
    log(writer, val_scores, iteration, type="val")

    # training
    init_state = model.state_dict()
    excel_writer = pd.ExcelWriter("probs.xlsx")
    alpha = 0.5
    for rnd in range(n_rounds):
        num_samples = query_train_mask.sum()
        logging.info(f"START ROUND {rnd + 1}: {num_samples} samples")
        model.load_state_dict(init_state)
        pbar = trange(round_epoch, desc="Training")
        for epoch in pbar:
            iteration += 1
            model.train()
            optimizer.zero_grad()
            features, logits = model(
                graph, inputs, target_node, edge_weight=edge_weight, return_features=True
            )
            probs = torch.sigmoid(logits)
            main_loss = nn.BCEWithLogitsLoss()(
                logits[query_train_mask], labels[query_train_mask].type_as(logits)
            )
            hier_loss = None
            for aux_idx in hier_indices:
                aux_logits, aux_labels = [], []
                for k, v in aux_idx.items():
                    aux_logits.append(torch.max(logits[:, v], dim=1).values.reshape((-1, 1)))
                    aux_labels.append(torch.max(labels[:, v], dim=1).values.reshape((-1, 1)))
                aux_logits = torch.cat(aux_logits, dim=1)
                aux_labels = torch.cat(aux_labels, dim=1)
                aux_loss = nn.BCEWithLogitsLoss()(
                    aux_logits[query_train_mask], aux_labels[query_train_mask].type_as(aux_logits)
                )
                hier_loss = aux_loss if hier_loss is None else hier_loss + aux_loss

            loss = alpha * main_loss + (1 - alpha) * hier_loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item(), lr="{:.1e}".format(lr_scheduler.get_last_lr()[0]))
            writer.add_scalar("loss/train", loss.item(), epoch)

            # validation
            if (epoch + 1) % update_freq == 0:
                log_iteration = (rnd + 1) if use_active_learning else iteration
                train_scores = predict(
                    model, graph, target_node, inputs, edge_weight, labels, train_mask, threshold=threshold
                )
                val_scores = predict(
                    model, graph, target_node, inputs, edge_weight, labels, val_mask, threshold=threshold
                )
                log(writer, train_scores, log_iteration, type="train")
                log(writer, val_scores, log_iteration, type="val")
                print(val_scores)

            if iteration % 100 == 0:
                lr_scheduler.step()

        # choose next samples for the next round (except the last)
        if use_active_learning and rnd != (n_rounds - 1):
            query_train_mask = strategy.query(probs.detach().cpu(), labels.cpu(), features=features.detach().cpu())

    pd.DataFrame(probs.detach().cpu().numpy()).sample(100).to_excel(excel_writer, sheet_name="prob")
    excel_writer.close()

    # inference at the last round
    print("**** TEST ****")
    test_scores = predict(
        model, graph, target_node, inputs, edge_weight, labels, test_mask, threshold=threshold
    )
    log(writer, test_scores, 0, type="test")
    print(test_scores)
    os.makedirs("results", exist_ok=True)
    report = pd.Series(test_scores).sort_index() * 100
    report.to_csv(os.path.join("results", f"{exp_name}.csv"), float_format="%.2f")
    
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

def log(writer: SummaryWriter, scores: Dict, iteration: int = 0, type: str = "train"):
    for k, v in scores.items():
        writer.add_scalar(f"{k}/{type}", v, iteration)
