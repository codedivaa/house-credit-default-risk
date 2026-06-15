import pandas as pd
import re
import numpy as np

from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

print("Loading data...")

train = pd.read_csv("data/raw/application_train.csv")
test = pd.read_csv("data/raw/application_test.csv")

# Load aggregated tables
bureau = pd.read_csv("data/raw/bureau_aggregated.csv")
prev = pd.read_csv("data/raw/previous_aggregated.csv")
inst = pd.read_csv("data/raw/installments_aggregated.csv")
cc = pd.read_csv("data/raw/creditcard_aggregated.csv")
pos = pd.read_csv("data/raw/pos_cash_aggregated.csv")

# Merge train
for extra in [bureau, prev, inst, cc, pos]:
    train = train.merge(extra, on="SK_ID_CURR", how="left")
    test = test.merge(extra, on="SK_ID_CURR", how="left")

y = train["TARGET"]

X_train = train.drop(columns=["TARGET"])
X_test = test.copy()

# ===== Feature Engineering =====

for df in [X_train, X_test]:

    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] /
        df["CNT_FAM_MEMBERS"]
    )

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_TERM"] = (
        df["AMT_CREDIT"] /
        df["AMT_ANNUITY"]
    )

    df["IS_UNEMPLOYED"] = (
        df["DAYS_EMPLOYED"] == 365243
    ).astype(int)

    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        365243,
        float("nan")
    )

    df["EMPLOYED_BIRTH_RATIO"] = (
        df["DAYS_EMPLOYED"] /
        df["DAYS_BIRTH"]
    )

    df["EXT_SOURCE_MEAN"] = df[
        ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
    ].mean(axis=1)

    df["EXT_SOURCE_STD"] = df[
        ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
    ].std(axis=1)

    df["EXT_SOURCE_PROD"] = (
        df["EXT_SOURCE_1"].fillna(1)
        * df["EXT_SOURCE_2"].fillna(1)
        * df["EXT_SOURCE_3"].fillna(1)
    )

    df["DOCUMENT_COUNT"] = df[
        [c for c in df.columns if c.startswith("FLAG_DOCUMENT")]
    ].sum(axis=1)

    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"] /
        df["AMT_GOODS_PRICE"]
    )

    df["GOODS_INCOME_RATIO"] = (
        df["AMT_GOODS_PRICE"] /
        df["AMT_INCOME_TOTAL"]
    )

    df["MISSING_COUNT"] = df.isnull().sum(axis=1)

# One-hot encoding together
combined = pd.concat([X_train, X_test])

combined = pd.get_dummies(combined)

combined.columns = [
    re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
    for col in combined.columns
]

combined = combined.fillna(-999)

X_train = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]

# ===== 5 Fold Training =====

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(
    kf.split(X_train, y),
    start=1
):

    print(f"Training Fold {fold}")

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
        X_train.iloc[train_idx],
        y.iloc[train_idx]
    )

    test_preds += (
        model.predict_proba(X_test)[:,1] / 5
    )

submission = pd.DataFrame({
    "SK_ID_CURR": test["SK_ID_CURR"],
    "TARGET": test_preds
})

submission.to_csv(
    "submission_kfold.csv",
    index=False
)

print("submission_kfold.csv created!")