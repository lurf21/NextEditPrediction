import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "jsonlines", "huggingface_hub")
    .apt_install("git", "git-lfs")
    .add_local_file("data_processing.py", "/root/data_processing.py")
)

app = modal.App("commitpack_processing", image=image)

def split_text_field(example):
    prompt = example["text"].split("<|next_version|>\n")[0]
    prompt += "<|next_version|>\n"
    response = example["text"].split("<|next_version|>\n")[1]
    return {
        "prompt": prompt,
        "response": response,
    }

@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", create_if_missing=True
        ),
        "/root/training_data": modal.Volume.from_name(
            "training_data", create_if_missing=True
        ),
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def main():
    import os
    import shutil
    import subprocess
    from datasets import load_dataset
    from huggingface_hub import HfApi, HfFolder, create_repo
    from data_processing import remove_last_edit_chunk, build_text_field

    dataset_name = "commitpackft"

    dataset = load_dataset(f"{dataset_name}_labeled", split="train")

    dataset = dataset.filter(lambda x: x["label"] == 1, num_proc=16)
    dataset = dataset.map(remove_last_edit_chunk, num_proc=16)
    dataset = dataset.filter(lambda x: x["incomplete_new_contents"] is not None, num_proc=16)
    dataset = dataset.map(build_text_field, num_proc=16)
    dataset = dataset.remove_columns(["label"])


    dataset.to_json("/root/training_data/train.jsonl", orient="records", lines=True)

    username = ""
    repo_name = f"{dataset_name}_training_data"
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

    shutil.copytree("/root/training_data", "./data", dirs_exist_ok=True)

    subprocess.run(["huggingface-cli", "lfs-enable-largefiles", "."])
    subprocess.run(["git", "lfs", "track", "*.jsonl"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", f"{dataset_name}"])
    subprocess.run(["git", "push"])