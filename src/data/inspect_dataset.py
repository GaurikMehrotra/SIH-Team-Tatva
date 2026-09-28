import h5py
import numpy as np

FILE = "/mnt/d/TSRD/raw/archive/test/test_0.h5"

with h5py.File(FILE, "r") as f:
    print("Feature names:")
    print(f["metadata"]["feature_names"][:])

    print("\nFirst 10 rows of data:")
    print(f["data"][:10])

    print("\nUnique labels:")
    print(np.unique(f["labels"][:]))

    print("\nLabel counts:")
    labels = f["labels"][:].flatten()
    unique, counts = np.unique(labels, return_counts=True)

    for u, c in zip(unique, counts):
        print(u, c)