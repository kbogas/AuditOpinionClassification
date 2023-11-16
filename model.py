from typing import Tuple

import numpy as np
import torch
import tqdm
from scipy import datasets
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AutoModel, DistilBertModel, LongformerModel

from utils import multi_label_metrics

# HARD-CODED values for the weighted loss for the bankruptcy labels
label_index = [4, 10, 11, 35, 42, 44]
weight_value = [40.46, 9.02, 5.38, 3.75, 4.69, 8.27]

weights = torch.ones(55)
for ind, val in zip(label_index, weight_value):
    weights[ind] = val


class LMClassifier(torch.nn.Module):
    """Wrapper Model for language models.
    It simply employs a lanugage model as encoder
    Keeps the CLS token of the last hidden layer
    and propagets it to a trainable linear layer


    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_hidden: int = 768,
        num_output: int = 57,
        device: str = "cpu",
    ):
        super(LMClassifier, self).__init__()
        self.model_name = model_name
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.device = device

        # Load the model
        self.transformer = AutoModel.from_pretrained(self.model_name)

        # Init Dropout
        self.dropout = torch.nn.Dropout(0.1)
        # Final Classification layer
        self.classifier = torch.nn.Linear(self.num_hidden, self.num_output)
        self.classifier.requires_grad_()

    def forward(self, input_ids, attention_mask):
        # Encode the inputs
        output_1 = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # last hidden layer
        hidden_state = output_1.last_hidden_state
        # keep cls repr
        pooler = hidden_state[:, 0, :]
        # dropout
        pooler = self.dropout(pooler)
        # classification projection
        output = self.classifier(pooler)
        return output


def loss_fn(outputs, targets):
    # simple wrapper for BCE with logits
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def focal_binary_cross_entropy(logits, targets, gamma=2):
    """
    Taken from
    https://www.kaggle.com/code/thedrcat/focal-multilabel-loss-in-pytorch-explained
    and changed for the multi-label case
    """
    # Flatten
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    # Logits -> [0,1]
    p = torch.sigmoid(l)
    # Thresholding
    p = torch.where(t >= 0.5, p, 1 - p)
    # Log-score
    logp = -torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))
    loss = logp * ((1 - p) ** gamma)
    loss = targets.shape[0] * loss.mean()
    return loss


def train_epoch(
    model: LMClassifier,
    dataloader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    metrics: dict,
    loss_strategy: str,
) -> Tuple[LMClassifier, dict]:
    """Wrapper function for training for one epoch

    Args:
        model (LMClassifier): The LM + Classifier model
        dataloader (DataLoader): The train dataloader.
        device (str): GPU or CPU.
        optimizer (torch.optim.Optimizer): The initialized optimizer.
        scheduler (torch.optim.lr_scheduler): The initialized scheduler.
        metrics (dict): A dictionary with metrics.
        loss_strategy (str): The loss strategy.
        Available are ["default", "weighted_bankruptcy", "focal_loss_weighted_bankruptcy"].

    Raises:
        AttributeError: In case the loss strateg is not understood.

    Returns:
        Tuple[LMClassifier, dict]: The model trained for one epoch and the corresponding metrics
    """
    # Init metrics
    epoch_metrics = {key: torch.Tensor([]).to(device) for key in metrics} | {
        "loss": torch.Tensor([]).to(device)
    }

    model.to(device)
    model.train()
    batch_iterator = tqdm.tqdm(list(enumerate(dataloader, 0)), ascii=True)
    # Iterate over the batches
    for batch_id, data in batch_iterator:
        optimizer.zero_grad()
        # Data
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["labels"].to(device, dtype=torch.float)
        # Model predictions
        outputs = model(ids, mask)
        # Loss modification
        if loss_strategy == "default":
            loss = torch.nn.BCEWithLogitsLoss()(outputs, targets)
        elif loss_strategy == "weighted_bankruptcy":
            # simple weighted loss
            loss = torch.nn.BCEWithLogitsLoss(
                pos_weight=weights.to(device, dtype=torch.float)
            )(outputs, targets)
        elif loss_strategy == "focal_loss_weighted_bankruptcy":
            # Focal weighted loss
            from focal_loss import FocalLoss

            fl = FocalLoss(alpha=weights.to(device, dtype=torch.float))
            loss = fl(outputs, targets)
        else:
            raise AttributeError(f"Loss strategy: {loss_strategy} not understood...")

        loss.backward()
        optimizer.step()
        # Update the metrics
        for metric_name, metric in metrics.items():
            epoch_metrics[metric_name] = torch.cat(
                (
                    epoch_metrics[metric_name],
                    metric(torch.sigmoid(outputs), targets.long()).unsqueeze(0),
                )
            )

        epoch_metrics["loss"] = torch.cat((epoch_metrics["loss"], loss.unsqueeze(0)))

        batch_iterator.set_description(
            f"Batch: [{batch_id}/{len(batch_iterator)}] "
            + " ".join(
                [
                    "{}: {:.2f}".format(key, value.mean().item())
                    for key, value in epoch_metrics.items()
                ]
            )
        )

    scheduler.step()

    return model, epoch_metrics


def eval_model(
    model: LMClassifier,
    dataloader: DataLoader,
    device: str,
    class_names: list,
    print_classification_report: bool = False,
) -> tuple[dict[str, list], str]:
    """Helper function to evaluate the models.

    Args:
        model (LMClassifier): The LM + Classifier model.
        dataloader (DataLoader): The validation/test dataloader
        device (str): GPU or CPU.
        class_names (list): The class names. For visualization purposes.
        print_classification_report (bool, optional): Whether to print the classification_report. Defaults to False.

    Returns:
        Tuple[dict[str, list], str]: A dict with the metrics wanted and the classification_report string.
    """

    # Initis
    val_results = {
        "micro_f1": [],
        "micro_roc_auc": [],
        "accuracy_manual": [],
        "accuracy": [],
    }
    predictions, true = [], []
    classification_report_str = ""

    # Model eval
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(dataloader, 0):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["labels"].to(device, dtype=torch.float)

            outputs = model(ids, mask)

            cur_results, cur_pred = multi_label_metrics(
                outputs.cpu(), targets.cpu(), threshold=0.5
            )

            predictions.append(cur_pred)
            true.append(targets.cpu().numpy())

            for k, v in cur_results.items():
                val_results[k].append(v)

    print(f"Validation results")

    for k, v in val_results.items():
        print(f"Mean {k}: {np.mean(v):.4f}")

    true = np.vstack(true)
    predictions = np.vstack(predictions)

    avg_prec = (true * predictions).sum() / predictions.sum()
    avg_rec = (true * predictions).sum() / true.sum()
    print(f"Avg. Prec: {avg_prec}, Avg. Rec: {avg_rec}")

    if print_classification_report:
        classification_report_str = classification_report(
            true, predictions, target_names=class_names
        )
        print(classification_report_str)

    print("\n\n")
    return val_results, classification_report_str  # type: ignore
