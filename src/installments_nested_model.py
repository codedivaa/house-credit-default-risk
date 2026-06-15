import pandas as pd

print("Loading data...")

inst = pd.read_csv(
    "data/raw/installments_payments.csv"
)

train = pd.read_csv(
    "data/raw/application_train.csv"
)[["SK_ID_CURR", "TARGET"]]

# Merge target onto installment rows
inst = inst.merge(
    train,
    on="SK_ID_CURR",
    how="left"
)

# Feature engineering on installment rows

inst["PAYMENT_DIFF"] = (
    inst["AMT_PAYMENT"]
    - inst["AMT_INSTALMENT"]
)

inst["PAYMENT_RATIO"] = (
    inst["AMT_PAYMENT"]
    / inst["AMT_INSTALMENT"]
)

inst["LATE_PAYMENT"] = (
    inst["DAYS_ENTRY_PAYMENT"]
    - inst["DAYS_INSTALMENT"]
)

# Aggregate per customer

features = inst.groupby(
    "SK_ID_CURR"
).agg({
    "PAYMENT_DIFF": [
        "mean",
        "max",
        "min",
        "std"
    ],
    "PAYMENT_RATIO": [
        "mean",
        "max",
        "min"
    ],
    "LATE_PAYMENT": [
        "mean",
        "max"
    ],
    "SK_ID_PREV": [
        "count"
    ]
})

features.columns = [
    "_".join(col)
    for col in features.columns
]

features = features.reset_index()

features.to_csv(
    "data/raw/installments_nested_features.csv",
    index=False
)

print(features.head())
print("\nSaved installments_nested_features.csv")