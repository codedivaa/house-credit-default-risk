import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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

X["EMPLOYED_BIRTH_RATIO"] = (
    X["DAYS_EMPLOYED"] /
    X["DAYS_BIRTH"]
)

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

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training LightGBM...")

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

model.fit(X_train, y_train)

preds = model.predict_proba(X_valid)[:, 1]

auc = roc_auc_score(y_valid, preds)

print(f"\nROC-AUC: {auc:.5f}")