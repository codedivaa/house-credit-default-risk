import pandas as pd

print("Loading credit card balance...")

cc = pd.read_csv("data/raw/credit_card_balance.csv")

cc_agg = cc.groupby("SK_ID_CURR").agg({
    "AMT_BALANCE": ["mean", "max"],
    "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
    "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
    "SK_ID_PREV": ["count"]
})

cc_agg.columns = [
    "_".join(col)
    for col in cc_agg.columns
]

cc_agg = cc_agg.reset_index()

cc_agg.to_csv(
    "data/raw/creditcard_aggregated.csv",
    index=False
)

print("Saved creditcard_aggregated.csv")