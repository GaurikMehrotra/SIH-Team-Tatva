from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
    repo_type="dataset",
    local_dir="/mnt/d/TSRD/raw",
    cache_dir="/mnt/d/TSRD/cache",
    resume_download=True,
)

print("Download Complete")