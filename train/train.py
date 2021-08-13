from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim
from dgl import DGLGraph
from .metrics import compute_metrics


def train(model: nn.Module, graph: DGLGraph, target_node: str, lr: float, epochs: int, threshold: float = 0.5):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch : lr * 0.9 ** epoch)
    criterion = nn.BCEWithLogitsLoss()
    train_mask = graph.nodes[target_node].data["train_mask"]
    val_mask = graph.nodes[target_node].data["val_mask"]
    test_mask = graph.nodes[target_node].data["test_mask"]
    labels = graph.nodes[target_node].data["label"]

    pbar = trange(epochs, desc="Training")
    for epoch in pbar:
        optimizer.zero_grad()
        logits = model(graph, target_node)
        loss = criterion(logits[train_mask], labels[train_mask].type_as(logits))
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item(), lr="{:.1e}".format(lr_scheduler.get_last_lr()[0]))
        if (epoch + 1) % 100 == 0:
            print(eval(labels, logits, val_mask, threshold=threshold))
            lr_scheduler.step()

    print("**** TEST ****")
    print(eval(labels, logits, test_mask, threshold=threshold))

def eval(labels, logits, mask, threshold: float = 0.5):
    y_true = labels[mask].detach().cpu().numpy()
    y_prob = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    return compute_metrics(y_true, y_prob, threshold=threshold)