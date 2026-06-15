import pandas as pd

print("Loading bureau data...")

bureau = pd.read_csv("data/raw/bureau.csv")

print("Original Shape:", bureau.shape)

bureau_agg = bureau.groupby("SK_ID_CURR").agg({
    "AMT_CREDIT_SUM": [
    "mean",
    "max",
    "min",
    "sum",
    "std",
    "median"
],
    "AMT_CREDIT_SUM_DEBT": [
        "mean",
        "max",
        "sum",
        "std"
    ],
    "CREDIT_DAY_OVERDUE": [
    "mean",
    "max",
    "sum",
    "std"
],
    "SK_ID_BUREAU": ["count"]
})

bureau_agg.columns = [
    "_".join(col)
    for col in bureau_agg.columns
]

bureau_agg = bureau_agg.reset_index()

print("Aggregated Shape:", bureau_agg.shape)

bureau_agg.to_csv(
    "data/raw/bureau_aggregated.csv",
    index=False
)

print("Saved bureau_aggregated.csv")