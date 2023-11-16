import ast
import json
import os

import torch
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class GCDataset(Dataset):

    """Helper class for the going concerns"""

    def __init__(self, encodings):
        self.encodings = encodings
        self.labels = [item["labels"] for item in self.encodings]

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = self.encodings[idx]["input_ids"]
        item["attention_mask"] = self.encodings[idx]["attention_mask"]
        item["labels"] = self.labels[idx].detach().clone()
        return item

    def __len__(self):
        return len(self.labels)


class DataPreprocessor(object):
    """A preprocessing helper data class"""

    def __init__(
        self,
        path_to_root: str,
        tokenizer_name: str = "distilbert-base-uncased",
        max_tokens: int = 512,
        truncation_strategy: str = "first",
        use_splits_for_mlb: list[str] = ["train"],
    ):
        self.path_to_root = path_to_root
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.use_splits_for_mlb = use_splits_for_mlb
        print(f"Will tokenize with {self.tokenizer_name}..")

        self.max_tokens = max_tokens
        self.truncation_strategy = truncation_strategy

        self.setup_label_binarizer()

    def setup_label_binarizer(self):
        # This will initialize and fit a multi-label binarizer.
        labels_list = []
        for split_name in self.use_splits_for_mlb:
            with open(os.path.join(self.path_to_root, f"{split_name}.jsonl"), "r") as f:
                cur_data = json.load(f)
                for datum in cur_data:
                    labels_list.append(
                        ast.literal_eval(datum["Going Concern Issue Phrase List"])
                    )
        self.mlb = MultiLabelBinarizer()
        y_labels = self.mlb.fit_transform(labels_list)
        print(f"Number of classes in train: {y_labels.shape[1]}")
        return None

    def get_data_splits(self):
        # Create the dataset splits

        # Load each dataset split
        dataset_train = load_dataset(
            "json",
            data_files=os.path.join(
                self.path_to_root,
                "train_opinions_summaries.jsonl",
            ),
            split="train[:]",
        )
        dataset_val = load_dataset(
            "json",
            data_files=os.path.join(self.path_to_root, "val_opinions_summaries.jsonl"),
            split="train[:]",
        )
        dataset_test = load_dataset(
            "json",
            data_files=os.path.join(self.path_to_root, "test_opinions_summaries.jsonl"),
            split="train[:]",
        )

        print(
            f"Len train: {len(dataset_train)}\tLen val: {len(dataset_val)}\t Len test: {len(dataset_test)}"  # type: ignore
        )

        # Encode them
        dataset_train_encoded = dataset_train.map(self.preprocess_data, batched=False)
        dataset_val_encoded = dataset_val.map(self.preprocess_data, batched=False)
        dataset_test_encoded = dataset_test.map(self.preprocess_data, batched=False)

        dataset_train_encoded.set_format("pytorch")  # type: ignore
        dataset_val_encoded.set_format("pytorch")  # type: ignore
        dataset_test_encoded.set_format("pytorch")  # type: ignore

        # Transofrm them to Dataset classes
        self.train_dataset = GCDataset(dataset_train_encoded)
        self.val_dataset = GCDataset(dataset_val_encoded)
        self.test_dataset = GCDataset(dataset_test_encoded)

        return self.train_dataset, self.val_dataset, self.test_dataset

    def get_dataloaders(self, batch_size: int = 16):
        # Having the datasets create the corresponding dataloaders

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False
        )
        return self.train_loader, self.val_loader, self.test_loader

    def preprocess_data(self, examples):
        # Following # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb

        # take a batch of texts
        text = examples["OPINION_TEXT"]
        # If we simple use the first 512 tokens do nothing
        if self.truncation_strategy == "first":
            pass
        # If we want the middel tokens, find them.
        elif self.truncation_strategy == "middle":
            middle_point = len(text) // 2
            window_step = self.max_tokens // 2
            text = text[middle_point - window_step : middle_point + window_step]
        # else use the summary of the audit
        elif self.truncation_strategy == "summary":
            text = examples["OPINION_SUMMARY"]
        # encode them
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            add_special_tokens=True,
        )
        # Add the going-cocnern input labels
        encoding["labels"] = self.mlb.transform(  # type: ignore
            [examples["Going Concern Issue Phrase List"]]
        ).astype(float)[0]

        return encoding


if __name__ == "__main__":
    # Datasets
    root_path = f"./frozen_splits"
    preprocessor = DataPreprocessor(root_path, truncation_strategy="middle")
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_splits()
    train_loader, val_loader, test_loader = preprocessor.get_dataloaders()
