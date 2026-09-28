from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="alan-turing-institute/turing-synthetic-radar-dataset",
    repo_type="dataset"
)

print(f"Found {len(files)} files\n")

for file in files[:50]:
    print(file)
