import h5py
from pathlib import Path

base = Path("/mnt/d/TSRD/raw/archive/train")

total = 0

for f in base.glob("*.h5"):
    with h5py.File(f, "r") as h:
        total += h["data"].shape[0]

print("Total rows:", total)