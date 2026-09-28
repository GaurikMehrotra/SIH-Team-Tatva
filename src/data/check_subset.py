import pandas as pd

df = pd.read_parquet("/mnt/d/TSRD/train_subset_1m.parquet")

print(df.head())
print()
print("Shape:", df.shape)
print()
print("Classes:", df["label"].nunique())
print()
print(df["label"].value_counts().head(20))