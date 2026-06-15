import pandas as pd

train = pd.read_csv("data/raw/application_train.csv")
test = pd.read_csv("data/raw/application_test.csv")

print("=" * 50)
print("TRAIN SHAPE")
print(train.shape)

print("=" * 50)
print("TEST SHAPE")
print(test.shape)

print("=" * 50)
print("FIRST 5 ROWS")
print(train.head())

print("=" * 50)
print("MISSING VALUES")
print(train.isnull().sum().sort_values(ascending=False).head(20))

print("=" * 50)
print("TARGET DISTRIBUTION")
print(train["TARGET"].value_counts(normalize=True))