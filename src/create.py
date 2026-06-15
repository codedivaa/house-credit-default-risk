import pandas as pd
import re

from lightgbm import LGBMClassifier

print("Loading datasets...")

train = pd.read_csv("data/raw/application_train.csv")
test = pd.read_csv("data/raw/application_test.csv")

# Merge all aggregated features
bureau = pd.read_csv("data/raw/bureau_aggregated.csv")
prev = pd.read_csv("data/raw/previous_aggregated.csv")
inst = pd.read_csv("data/raw/installments_aggregated.csv")
cc = pd.read_csv("data/raw/creditcard_aggregated.csv")
pos = pd.read_csv("data/raw/pos_cash_aggregated.csv")

for df in [train, test]:
    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    )

    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_TERM"] = (
        df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    )

    df["EMPLOYED_BIRTH_RATIO"] = (
        df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    )

    df = df.merge(bureau, on="SK_ID_CURR", how="left")
    df = df.merge(prev, on="SK_ID_CURR", how="left")
    df = df.merge(inst, on="SK_ID_CURR", how="left")
    df = df.merge(cc, on="SK_ID_CURR", how="left")
    df = df.merge(pos, on="SK_ID_CURR", how="left")

y = train["TARGET"]

X_train = train.drop(columns=["TARGET"])
X_test = test.copy()

# One-hot encoding together
combined = pd.concat([X_train, X_test], axis=0)

combined = pd.get_dummies(combined)

combined.columns = [
    re.sub(r"[^A-Za-z0-9_]+", "_", str(col))
    for col in combined.columns
]

combined = combined.fillna(-999)

X_train = combined.iloc[:len(train)]
X_test = combined.iloc[len(train):]

print("Training final model...")

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

model.fit(X_train, y)

preds = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "SK_ID_CURR": test["SK_ID_CURR"],
    "TARGET": preds
})

submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv created!")