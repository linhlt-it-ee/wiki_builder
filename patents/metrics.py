import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = np.vectorize(lambda p : int(p >= threshold))(y_prob)
    metrics = {}
    clf_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    for reduce_type in ("micro avg", "macro avg"):
        for metric, score in clf_report[reduce_type].items():
            metrics[f"{reduce_type}_{metric}"] = score
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    # metrics["macro avg_auc"] = roc_auc_score(y_true, y_prob, average="macro")
    # metrics["micro avg_auc"] = roc_auc_score(y_true, y_prob, average="micro")
    return metrics
