import numpy as np
from sklearn.metrics import f1_score, ndcg_score

def precision(y_true, y_prob, k: int):
    ky_pred = np.argsort(y_prob)[::-1][:, :k]
    score = y_true.take(ky_pred).sum(axis=1) / k
    return score.mean()

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    metrics = {}
    y_pred = np.where(y_prob >= threshold, 1, 0)
    # y_pred = np.argmax(y_prob, axis=1)
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro")
    for k in (1, 3, 5):
        metrics[f"p@{k}"] = precision(y_true, y_prob, k=k)
        metrics[f"ndcg@{k}"] = ndcg_score(y_true, y_prob, k=k)
    return metrics
