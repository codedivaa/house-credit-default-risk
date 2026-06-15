import pandas as pd

print("Loading installments...")

inst = pd.read_csv("data/raw/installments_payments.csv")

inst["PAYMENT_DIFF"] = (
    inst["AMT_PAYMENT"] -
    inst["AMT_INSTALMENT"]
)

inst_agg = inst.groupby("SK_ID_CURR").agg({
    "PAYMENT_DIFF": ["mean", "min", "max"],
    "AMT_PAYMENT": ["mean", "sum"],
    "AMT_INSTALMENT": ["mean", "sum"]
})

inst_agg.columns = [
    "_".join(col)
    for col in inst_agg.columns
]

inst_agg = inst_agg.reset_index()

inst_agg.to_csv(
    "data/raw/installments_aggregated.csv",
    index=False
)

print("Saved installments_aggregated.csv")