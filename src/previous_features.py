import pandas as pd

print("Loading previous applications...")

prev = pd.read_csv("data/raw/previous_application.csv")

prev_agg = prev.groupby("SK_ID_CURR").agg({
    "AMT_CREDIT": ["mean", "max"],
    "AMT_ANNUITY": ["mean"],
    "SK_ID_PREV": ["count"]
})

prev_agg.columns = [
    "_".join(col)
    for col in prev_agg.columns
]

prev_agg = prev_agg.reset_index()

prev_agg.to_csv(
    "data/raw/previous_aggregated.csv",
    index=False
)

print("Saved previous_aggregated.csv")