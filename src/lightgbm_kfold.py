import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier

print("Loading data...")

df = pd.read_csv("data/raw/application_train.csv")
bureau = pd.read_csv("data/raw/bureau_aggregated.csv")

df = df.merge(
    bureau,
    on="SK_ID_CURR",
    how="left"
)

prev = pd.read_csv("data/raw/previous_aggregated.csv")

df = df.merge(
    prev,
    on="SK_ID_CURR",
    how="left"
)
inst = pd.read_csv("data/raw/installments_aggregated.csv")
df = df.merge(
    inst,
    on="SK_ID_CURR",
    how="left"
)
cc = pd.read_csv("data/raw/creditcard_aggregated.csv")
df = df.merge(
    cc,
    on="SK_ID_CURR",
    how="left"
)
pos = pd.read_csv("data/raw/pos_cash_aggregated.csv")
df = df.merge(
    pos,
    on="SK_ID_CURR",
    how="left"
)
inst_nested = pd.read_csv(
    "data/raw/installments_nested_features.csv"
)
inst_nested = inst_nested.rename(
    columns=lambda c: (
        "INST_NESTED_" + c
        if c != "SK_ID_CURR"
        else c
    )
)

df = df.merge(
    inst_nested,
    on="SK_ID_CURR",
    how="left"
)
y = df["TARGET"]
X = df.drop(columns=["TARGET"])
# Feature Engineering

X["INCOME_PER_PERSON"] = (
    X["AMT_INCOME_TOTAL"] /
    X["CNT_FAM_MEMBERS"]
)

X["CREDIT_INCOME_RATIO"] = (
    X["AMT_CREDIT"] /
    X["AMT_INCOME_TOTAL"]
)

X["ANNUITY_INCOME_RATIO"] = (
    X["AMT_ANNUITY"] /
    X["AMT_INCOME_TOTAL"]
)

X["CREDIT_TERM"] = (
    X["AMT_CREDIT"] /
    X["AMT_ANNUITY"]
)

# Known Home Credit bug feature
X["IS_UNEMPLOYED"] = (
    X["DAYS_EMPLOYED"] == 365243
).astype(int)

X["DAYS_EMPLOYED"] = X["DAYS_EMPLOYED"].replace(
    365243,
    float("nan")
)

X["EMPLOYED_BIRTH_RATIO"] = (
    X["DAYS_EMPLOYED"] /
    X["DAYS_BIRTH"]
)
X["EXT_SOURCE_MEAN"] = X[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
].mean(axis=1)

X["EXT_SOURCE_STD"] = X[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
].std(axis=1)

X["EXT_SOURCE_PROD"] = (
    X["EXT_SOURCE_1"].fillna(1)
    * X["EXT_SOURCE_2"].fillna(1)
    * X["EXT_SOURCE_3"].fillna(1)
)
X["DOCUMENT_COUNT"] = X[
    [c for c in X.columns if c.startswith("FLAG_DOCUMENT")]
].sum(axis=1)

X["CREDIT_GOODS_RATIO"] = (
    X["AMT_CREDIT"] /
    X["AMT_GOODS_PRICE"]
)

X["GOODS_INCOME_RATIO"] = (
    X["AMT_GOODS_PRICE"] /
    X["AMT_INCOME_TOTAL"]
)

X["MISSING_COUNT"] = X.isnull().sum(axis=1)
# Convert categoricals to dummy variables
X = pd.get_dummies(X)

# LightGBM doesn't like special chars in column names
X.columns = [
    re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
    for col in X.columns
]

# Fill missing values
X = X.fillna(-999)

print("Feature Shape:", X.shape)
print("Starting 5-Fold Cross Validation...")

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_preds = np.zeros(len(X))

scores = []

for fold, (train_idx, valid_idx) in enumerate(
    kf.split(X, y),
    start=1
):

    print(f"\n========== Fold {fold} ==========")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict_proba(X_valid)[:, 1]

    oof_preds[valid_idx] = preds

    fold_auc = roc_auc_score(
        y_valid,
        preds
    )

    scores.append(fold_auc)

    print(
        f"Fold {fold} ROC-AUC: {fold_auc:.5f}"
    )

print("\n====================")
print("Fold Scores:")
print(scores)

print(
    f"\nMean ROC-AUC: {np.mean(scores):.5f}"
)

print(
    f"OOF ROC-AUC: {roc_auc_score(y, oof_preds):.5f}"
)