import pandas as pd

train = pd.read_csv("data/raw/application_train.csv")

# Income per family member
train["INCOME_PER_PERSON"] = (
    train["AMT_INCOME_TOTAL"] /
    train["CNT_FAM_MEMBERS"]
)

# Credit to income
train["CREDIT_INCOME_RATIO"] = (
    train["AMT_CREDIT"] /
    train["AMT_INCOME_TOTAL"]
)

# Annuity to income
train["ANNUITY_INCOME_RATIO"] = (
    train["AMT_ANNUITY"] /
    train["AMT_INCOME_TOTAL"]
)

# Credit term
train["CREDIT_TERM"] = (
    train["AMT_CREDIT"] /
    train["AMT_ANNUITY"]
)

# Employment age ratio
train["EMPLOYED_BIRTH_RATIO"] = (
    train["DAYS_EMPLOYED"] /
    train["DAYS_BIRTH"]
)

print(train[
    [
        "INCOME_PER_PERSON",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_TERM",
        "EMPLOYED_BIRTH_RATIO"
    ]
].head())
