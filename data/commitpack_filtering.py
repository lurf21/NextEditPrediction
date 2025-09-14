import os
import modal
import argparse

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "jsonlines", "tiktoken", "huggingface_hub")
    .apt_install("git", "git-lfs")
    .add_local_file("data_processing.py", "/root/data_processing.py")
)

app = modal.App("commitpack_filtering", image=image)

def get_folder_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size

@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", create_if_missing=True
        ),
        "/root/dataset": modal.Volume.from_name(
            "dataset", create_if_missing=True
        ),
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def main(*arglist):
    import os
    import shutil
    import tiktoken
    import subprocess
    from datasets import load_dataset, Dataset
    from huggingface_hub import HfApi, HfFolder, create_repo
    from data_processing import (count_characters, count_tokens, edit_chunk_num,
                                 edit_chunk_len, edit_window_size, add_only, compute_diff)

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", default=8192)
    parser.add_argument("--max_edit_length", default=5)
    parser.add_argument("--max_window_size", default=80)
    parser.add_argument("--context_lines", default=10)
    parser.add_argument("--append", action="store_true")

    args = parser.parse_args(arglist)

    dataset_name = "commitpackft"

    HF_DATASETS_CACHE="/root/hf_cache"

    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    language_list = ["python", "java", "go", "c", "c++", "javascript", "typescript"]

    for language in language_list:
        print(f"Processing {language} dataset...")
        # Clean up the cache directory
        if os.path.exists(HF_DATASETS_CACHE):
            shutil.rmtree(HF_DATASETS_CACHE)
        os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
        # Check cache_dir size
        size_bytes = get_folder_size(HF_DATASETS_CACHE)
        print(f"Cache dir size: {size_bytes / (1024 ** 2):.2f} MB")
        ds = load_dataset(f"bigcode/{dataset_name}", language, num_proc=16, trust_remote_code=True, cache_dir=HF_DATASETS_CACHE, split="train")
        # Check cache_dir size
        size_bytes = get_folder_size(HF_DATASETS_CACHE)
        print(f"Cache dir size: {size_bytes / (1024 ** 2):.2f} MB")
        print(f"Loaded {len(ds)} commits from the dataset.")

        # Filter out commits which are too long.
        # Since tiktoken's processing of excessively long sequences can lead to 
        # stack overflow, the string must first be filtered based on its length.
        # Ref: https://github.com/openai/tiktoken/issues/15
        ds = ds.filter(lambda x: count_characters(x), num_proc=16)
        ds = ds.filter(lambda x: count_tokens(x, enc, args.max_tokens), num_proc=16)
        
        ds = ds.map(lambda x: {"diff": compute_diff(x)}, num_proc=16)

        ds = ds.filter(lambda x: edit_chunk_num(x["diff"]), num_proc=16)
        print(f"After Edit Chunk Number filter: {len(ds)} commits")
        ds = ds.filter(lambda x: edit_chunk_len(x["diff"], args.max_edit_length), num_proc=16)
        print(f"After Edit Chunk Length filter: {len(ds)} commits")
        ds = ds.filter(lambda x: edit_window_size(x["diff"], args.max_window_size), num_proc=16)
        print(f"After Edit Window Size filter: {len(ds)} commits")
        ds = ds.filter(lambda x: add_only(x["diff"]), num_proc=16)
        print(f"After Add Only filter: {len(ds)} commits")

        ds.to_json(f"/root/dataset/{language}.jsonl")

    username = ""
    repo_name = f"{dataset_name}_filtered"
    repo_type = "dataset"
    hf_token = os.environ["HF_TOKEN"]

    HfFolder.save_token(hf_token)
    api = HfApi()
    try:
        create_repo(repo_id=f"{username}/{repo_name}", repo_type=repo_type, private=True)
        print("Dataset repo created.")
    except Exception as e:
        print("Repo already exists, skipping creation.")

    clone_url = f"https://{username}:{hf_token}@huggingface.co/datasets/{username}/{repo_name}"
    subprocess.run(["git", "lfs", "install"])
    subprocess.run(["git", "clone", clone_url])
    os.chdir(repo_name)

    subprocess.run(["git", "config", "user.name", ""])
    subprocess.run(["git", "config", "user.email", ""])

    shutil.copytree("/root/dataset", "./data", dirs_exist_ok=True)

    subprocess.run(["huggingface-cli", "lfs-enable-largefiles", "."])
    subprocess.run(["git", "lfs", "track", "*.jsonl"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", f"{dataset_name}"])
    subprocess.run(["git", "push"])