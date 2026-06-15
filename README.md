# 🏦 Home Credit Default Risk Prediction

> End-to-End Credit Risk Modeling using LightGBM, CatBoost, Advanced Feature Engineering, and Multi-Source Financial Data Aggregation

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-green)
![CatBoost](https://img.shields.io/badge/CatBoost-ML-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.78124-success)

---

# 📌 Project Overview

Financial institutions often struggle to determine whether a borrower will repay a loan. Rejecting reliable borrowers reduces business opportunities, while approving risky borrowers increases default rates.

This project builds a large-scale machine learning pipeline to predict the probability of loan default using customer application data, credit history, installment behavior, credit card usage, and previous borrowing records.

The solution was developed using the Home Credit Default Risk Kaggle competition dataset and focuses on feature engineering, risk modeling, and scalable tabular machine learning.

---

# 🎯 Problem Statement

Predict the probability that a customer will experience payment difficulties and default on a loan.

### Input

Customer information from:

* Application Data
* Bureau Credit History
* Previous Applications
* Installment Payments
* Credit Card Balance
* POS Cash Balance

### Output

Probability of default:

```text
0.00 → Low Risk
1.00 → High Risk
```

---

# 📊 Dataset Statistics

| Metric              | Value          |
| ------------------- | -------------- |
| Training Samples    | 307,511        |
| Test Samples        | 48,744         |
| Original Features   | 122            |
| Engineered Features | 250+           |
| Target Distribution | 8.07% Defaults |
| Evaluation Metric   | ROC-AUC        |

---

# 🏗 Architecture

```text
                        ┌──────────────────┐
                        │ Application Data │
                        └─────────┬────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼

 ┌─────────────┐         ┌─────────────┐         ┌─────────────┐
 │   Bureau    │         │ Previous App│         │ Installments│
 └──────┬──────┘         └──────┬──────┘         └──────┬──────┘
        │                       │                       │
        ▼                       ▼                       ▼

  Feature Aggregation    Feature Aggregation    Feature Aggregation

        │                       │                       │
        └──────────────┬────────┴──────────────┬────────┘
                       │                       │

                       ▼                       ▼

                Credit Card          POS Cash Balance
                Aggregation          Aggregation

                       │
                       ▼

             Feature Engineering Layer

                       │
                       ▼

                 Data Preprocessing

                       │
                       ▼

               LightGBM / CatBoost

                       │
                       ▼

                Default Probability
```

---

# ⚙️ Feature Engineering

## Applicant Features

### Income Per Person

```python
AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
```

### Credit Income Ratio

```python
AMT_CREDIT / AMT_INCOME_TOTAL
```

### Annuity Income Ratio

```python
AMT_ANNUITY / AMT_INCOME_TOTAL
```

### Credit Term

```python
AMT_CREDIT / AMT_ANNUITY
```

### Employment Age Ratio

```python
DAYS_EMPLOYED / DAYS_BIRTH
```

### Unemployment Indicator

```python
DAYS_EMPLOYED == 365243
```

### External Risk Features

* EXT_SOURCE_MEAN
* EXT_SOURCE_STD
* EXT_SOURCE_PROD

### Additional Risk Features

* DOCUMENT_COUNT
* CREDIT_GOODS_RATIO
* GOODS_INCOME_RATIO
* MISSING_COUNT

---

# 📈 Bureau Features

Aggregated customer credit history:

* Credit Sum Mean
* Credit Sum Max
* Credit Sum Min
* Credit Sum Median
* Debt Statistics
* Credit Overdue Statistics
* Bureau Record Count

---

# 💳 Previous Application Features

* Previous Credit Mean
* Previous Credit Max
* Previous Annuity Mean
* Historical Application Count

---

# 💰 Installment Features

* Payment Difference
* Payment Ratio
* Late Payment Indicators
* Historical Payment Behavior

---

# 🏦 Credit Card Features

* Average Balance
* Maximum Balance
* Credit Limit Statistics
* Drawing Amount Statistics

---

# 🛒 POS Cash Features

* Delinquency Statistics
* Balance History
* Previous POS Accounts

---

# 🤖 Models Used

## Random Forest Baseline

```text
ROC-AUC: 0.70053
```

## LightGBM

```python
LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8
)
```

## CatBoost

```python
CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8
)
```

---

# 📊 Model Performance

| Version                 | ROC-AUC |
| ----------------------- | ------- |
| Random Forest           | 0.70053 |
| LightGBM Baseline       | 0.75987 |
| Manual Features         | 0.76809 |
| Bureau Features         | 0.76942 |
| Previous Applications   | 0.77174 |
| Installments            | 0.77562 |
| Credit Card Balance     | 0.77719 |
| POS Cash Balance        | 0.77964 |
| Tuned LightGBM          | 0.78230 |
| Feature Enhanced K-Fold | 0.78124 |

---

# 🔄 Cross Validation Strategy

Implemented:

* Stratified 5-Fold Cross Validation
* Out-of-Fold Evaluation
* ROC-AUC Monitoring

Results:

```text
Mean ROC-AUC : 0.78126
OOF ROC-AUC  : 0.78124
```

---

# 📁 Project Structure

```text
HomeCreditDefaultRisk/

│
├── data/
│   └── raw/
│       ├── application_train.csv
│       ├── application_test.csv
│       ├── bureau.csv
│       ├── previous_application.csv
│       ├── installments_payments.csv
│       ├── credit_card_balance.csv
│       ├── POS_CASH_balance.csv
│
├── src/
│   ├── baseline.py
│   ├── lightgbm_baseline.py
│   ├── lightgbm_kfold.py
│   ├── catboost_model.py
│   ├── bureau_features.py
│   ├── previous_features.py
│   ├── installments_features.py
│   ├── creditcard_features.py
│   ├── pos_cash_features.py
│   └── create_submission_kfold.py
│
├── requirements.txt
├── README.md
└── submission_kfold.csv
```

---

# 🚀 Installation

```bash
git clone https://github.com/codedivaa/HomeCreditDefaultRisk.git

cd HomeCreditDefaultRisk

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

---

# ▶️ Run

```bash
python src/lightgbm_kfold.py
```

Generate submission:

```bash
python src/create_submission_kfold.py
```

---

# 📦 Requirements

```text
pandas
numpy
scikit-learn
lightgbm
catboost
joblib
```

---

# 🌍 Real World Applications

* Credit Risk Assessment
* Loan Approval Systems
* Banking Analytics
* Financial Inclusion Programs
* Consumer Lending Platforms
* FinTech Risk Engines
* Fraud & Default Prevention

---

# 👩‍💻 Author

GitHub: https://github.com/codedivaa
