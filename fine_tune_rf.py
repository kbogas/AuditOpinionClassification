"""
This is for showcasing properties as the actual data with labels are not publicly available
"""

import json
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# Binarizer for multi-label case
mlb = MultiLabelBinarizer()


with open("./frozen_splits/train_opinions_with_labels.jsonl", "r") as f:
    data = json.load(f)
    X_train = [d["OPINION_TEXT"] for d in data]
    y_train = [d["Going Concern Issue Phrase List"] for d in data]
    y_train = mlb.fit_transform(y_train)

with open("./frozen_splits/val_opinions_with_labels.jsonl", "r") as f:
    data = json.load(f)
    X_val = [d["OPINION_TEXT"] for d in data]
    y_val = [d["Going Concern Issue Phrase List"] for d in data]
    y_val = mlb.transform(y_val)

with open("./frozen_splits/test_opinions_with_labels.jsonl", "r") as f:
    data = json.load(f)
    X_test = [d["OPINION_TEXT"] for d in data]
    y_test = [d["Going Concern Issue Phrase List"] for d in data]
    y_test = mlb.transform(y_test)


name_of_run = "random_forest"
pipe = Pipeline(
    [("tf", TfidfVectorizer()), ("clf", RandomForestClassifier(random_state=42))]
)

pds = PredefinedSplit(
    test_fold=[-1 for i in range(len(y_train))] + [1 for i in range(len(y_val))]
)

parameters = {
    "tf__max_df": [1.0, 0.99, 0.95, 0.90],
    "tf__min_df": [1, 2, 3, 4, 5, 10],
    "tf__ngram_range": [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
    "clf__max_depth": [
        None,
        10,
        20,
        50,
        100,
    ],
    "clf__max_features": [
        "sqrt",
        2,
        3,
    ],
    "clf__min_samples_leaf": [
        1,
        3,
        4,
        5,
    ],
    "clf__criterion": ["gini", "entropy"],
    "clf__n_estimators": [100, 300],
}
grid_search = GridSearchCV(
    pipe,
    parameters,
    n_jobs=8,
    verbose=1,
    cv=pds,
    error_score="raise",  # type: ignore
    scoring="f1_micro",
    refit=False,
)


time_s = time.time()
grid_search.fit(X_train + X_val, np.vstack((y_train, y_val)))  # type: ignore
print(grid_search.best_params_)
print(grid_search.best_score_)

pipe = Pipeline(
    [("tf", TfidfVectorizer()), ("clf", RandomForestClassifier(random_state=42))]
)

pipe.set_params(**grid_search.best_params_)

pipe.fit(X_train, y_train)

y_pred_test = pipe.predict(X_test)
classification_report_str = classification_report(
    y_test, y_pred_test, target_names=mlb.classes_
)
print(classification_report_str)
print(f"Time taken {(time.time() - time_s)/60:.2f} mins..")

with open(f"./results/{name_of_run}_gs.txt", "w+") as f:
    f.write(classification_report_str)  # type: ignore
