"""
This is for showcasing properties as the actual data with labels are not publicly available
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import tqdm

from early_stopper import EarlyStopping
from load_data import DataPreprocessor
from model import LMClassifier, eval_model, train_epoch
from utils import init_metrics

###################################################################

# Keep track of datetime for logging
date_time = datetime.now().strftime("%d-%m-%Y__%H:%M")

# Parse config file
with open(sys.argv[1], "r") as f:
    settings = json.load(f)

# Set/Change ad-hoc settings
settings["frozen"] = settings["frozen"] == "True"
settings["weight_decay"] = 0.01


if "roberta" in settings["model_name"]:
    settings["num_hidden"] = 1024
    settings["batch_size"] = 4
    settings["weight_decay"] = 0.1
if "longformer" in settings["model_name"]:
    settings["max_tokens"] = 4096
if settings["frozen"]:
    settings["learning_rate"] = 5e-3
    settings["num_epochs"] = 50

if "loss_strategy" not in settings:
    settings["loss_strategy"] = "default"

print("\n\n")
print("#" * 20)
print(f"Will use the following settings: {settings}")
print("#" * 20)
print("\n\n")

# Create timestamped run id
name_of_run = "_" + sys.argv[1].split("/")[-1].split(".")[0] + date_time

# Load data
preprocessor = DataPreprocessor(
    path_to_root=settings["path_to_data_folder"],
    tokenizer_name=settings["model_name"],
    max_tokens=settings["max_tokens"],
    truncation_strategy=settings["truncation_strategy"],
)

num_classes = preprocessor.mlb.classes_.shape[0]
train_dataset, val_dataset, test_dataset = preprocessor.get_data_splits()
train_loader, val_loader, test_loader = preprocessor.get_dataloaders(
    batch_size=settings["batch_size"]
)


# Initialize model

model = LMClassifier(
    model_name=settings["model_name"],
    num_hidden=settings["num_hidden"],
    num_output=num_classes,
)

# If frozen, no finetuning, so we pass only the final classifier
# layer weights (bias + weights)
if settings["frozen"]:
    optimizer = torch.optim.AdamW(
        params=[mp for mp in model.parameters()][-2:],
        weight_decay=settings["weight_decay"],
        lr=settings["learning_rate"],
    )
else:
    # else pass optimize the whole model
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=settings["weight_decay"],
        lr=settings["learning_rate"],
    )

# Init scheduler following
# https://github.com/uds-lsv/bert-stable-fine-tuning
scheduler = torch.optim.lr_scheduler.OneCycleLR(  # type: ignore
    optimizer,
    max_lr=settings["learning_rate"],
    steps_per_epoch=len(train_loader),
    epochs=settings["num_epochs"],
    anneal_strategy="linear",
    pct_start=0.1,
)

metrics = init_metrics(num_labels=num_classes, device=settings["device"])
early_stopping = EarlyStopping(
    path=os.path.join(
        settings["path_to_model_folder"],
        name_of_run + "_.pt",
    ),
    patience=5,
)

epoch_iterator = tqdm.tqdm(list(range(settings["num_epochs"])), ascii=True)

# Train the model
for epoch in epoch_iterator:
    model, epoch_metrics = train_epoch(
        model=model,
        dataloader=train_loader,
        device=settings["device"],
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        loss_strategy=settings["loss_strategy"],
    )

    # Eval after each epoch
    epoch_metrics, classification_report_str = eval_model(
        model=model,
        dataloader=val_loader,
        device=settings["device"],
        class_names=preprocessor.mlb.classes_.tolist(),
        print_classification_report=False,
    )

    # Check for early stopping and if stopped, load the best model
    early_stopping(np.mean(epoch_metrics["micro_f1"]), model)
    if early_stopping.early_stop:
        print("Early stopped. Will load the best model for testing..")
        model = early_stopping.load_checkpoint(model)
        break

    epoch_iterator.set_description(
        f"Epoch: [{epoch}/{len(epoch_iterator)}] "
        + " ".join(
            [
                "{}: {:.2f}".format(key, np.mean(value))
                for key, value in epoch_metrics.items()
            ]
        )
    )

# Test using the best model
print(f"\n Testing..\n ")
test_metrics, classification_report_str = eval_model(
    model=model,
    dataloader=test_loader,
    device=settings["device"],
    class_names=preprocessor.mlb.classes_.tolist(),
    print_classification_report=True,
)


with open(f"./results/{name_of_run}.txt", "w+") as f:
    f.write(classification_report_str)
