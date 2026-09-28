import h5py
import pandas as pd
from pathlib import Path
from tqdm import tqdm

FEATURES = ["UTCTime", "RF", "PulseWidth", "AOA", "PA"]

BASE = Path("/mnt/d/TSRD/raw/archive")
OUT = Path("/mnt/d/TSRD/processed")

OUT.mkdir(parents=True, exist_ok=True)


def process_split(split_name):
    split_dir = BASE / split_name

    all_frames = []

    files = sorted(split_dir.glob("*.h5"))

    print(f"\nProcessing {split_name}: {len(files)} files")

    for file in tqdm(files):
        with h5py.File(file, "r") as f:

            X = f["data"][:]
            y = f["labels"][:].flatten()

            df = pd.DataFrame(X, columns=FEATURES)
            df["Class"] = y

            all_frames.append(df)

    final_df = pd.concat(all_frames, ignore_index=True)

    print(f"{split_name} shape:", final_df.shape)

    output_file = OUT / f"{split_name}.parquet"

    final_df.to_parquet(output_file)

    print(f"Saved -> {output_file}")


for split in ["train", "validation", "test"]:
    process_split(split)

print("\nDone.")