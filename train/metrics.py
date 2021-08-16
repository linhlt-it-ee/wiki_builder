from torchmetrics.classification import F1, Precision
from sklearn.metrics import ndcg_score


def compute_metrics(y_true, y_prob, threshold: float = 0.5):
    metrics = {}
    metrics["micro_f1"] = F1(threshold=threshold, average="micro")(y_prob, y_true).item()
    for k in (1, 3, 5):
        metrics[f"p@{k}"] = Precision(threshold=threshold, average="micro", top_k=k)(y_prob, y_true).item()
        metrics[f"ndcg@{k}"] = ndcg_score(y_true, y_prob, k=k)
    return metrics
