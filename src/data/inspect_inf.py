import pandas as pd
import numpy as np

df = pd.read_parquet("/mnt/d/TSRD/train_subset_1m.parquet")

mask = np.isinf(df["PA"])

print("Rows with inf:", mask.sum())
print()

print(df.loc[mask].head(20))