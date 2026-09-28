import h5py
import numpy as np
import pandas as pd
from pathlib import Path

FEATURES = ["UTCTime", "RF", "PulseWidth", "AOA", "PA"]

train_dir = Path("/mnt/d/TSRD/raw/archive/train")

rows_per_file = 400

dfs = []

for i, f in enumerate(train_dir.glob("*.h5")):
    with h5py.File(f, "r") as h:
        X = h["data"][:]
        y = h["labels"][:].flatten()

        n = min(rows_per_file, len(X))
        idx = np.random.choice(len(X), n, replace=False)

        df = pd.DataFrame(X[idx], columns=FEATURES)
        df["label"] = y[idx]

        dfs.append(df)

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1} files")

subset = pd.concat(dfs, ignore_index=True)

subset.to_parquet(
    "/mnt/d/TSRD/train_subset_1m.parquet",
    index=False
)

print("Rows:", len(subset))
print("Saved subset.")