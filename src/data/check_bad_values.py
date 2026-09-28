import pandas as pd
import numpy as np

df = pd.read_parquet("/mnt/d/TSRD/train_subset_1m.parquet")

for col in df.columns:
    if col == "label":
        continue

    print(f"\n=== {col} ===")

    print("dtype:", df[col].dtype)

    print("NaN:", df[col].isna().sum())

    print("Inf:", np.isinf(df[col]).sum())

    print("Min:", df[col].min())

    print("Max:", df[col].max())