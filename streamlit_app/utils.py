import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torchmetrics.classification import AUROC, Accuracy, F1Score


def init_metrics(num_labels: int = 57, device: str = "cpu"):
    metrics = {
        "accuracy": Accuracy(task="multilabel", num_labels=num_labels).to(device),
        "f1": F1Score(task="multilabel", num_labels=num_labels).to(device),
        "auc": AUROC(task="multilabel", num_labels=num_labels).to(device),
    }
    return metrics


def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = torch.nn.Sigmoid()(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels.numpy()
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        "micro_f1": f1_micro_average,
        "micro_roc_auc": roc_auc,
        "accuracy": accuracy,
        "accuracy_manual": (y_true == y_pred).flatten().sum()
        / (y_true.shape[0] * y_true.shape[1]),
    }
    return metrics, y_pred
