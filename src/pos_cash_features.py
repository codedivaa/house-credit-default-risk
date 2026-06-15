import pandas as pd

print("Loading POS cash data...")

pos = pd.read_csv("data/raw/POS_CASH_balance.csv")

pos_agg = pos.groupby("SK_ID_CURR").agg({
    "MONTHS_BALANCE": ["mean", "min", "max"],
    "SK_DPD": ["mean", "max"],
    "SK_DPD_DEF": ["mean", "max"],
    "SK_ID_PREV": ["count"]
})

pos_agg.columns = [
    "_".join(col)
    for col in pos_agg.columns
]

pos_agg = pos_agg.reset_index()

pos_agg.to_csv(
    "data/raw/pos_cash_aggregated.csv",
    index=False
)

print("Saved pos_cash_aggregated.csv")