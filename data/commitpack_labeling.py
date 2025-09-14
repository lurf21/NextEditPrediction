import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "tqdm", "openai", "tiktoken", "huggingface_hub")
    .apt_install("git", "git-lfs")
)

app = modal.App("commitpack_labeling", image=image)

task_description = 'I currently have a GitHub repository commit dataset, \
where each entry includes the unified diff results. \
We define a fragment composed of consecutive edit lines as an edit chunk. \
Please determine whether the edit chunks in the given commit diff are relevant, \
and respond with "yes" or "no" accordingly, without any additional explanation.'

pos_example = '''--- samples/plugins/scenario/scenario_plugin.py
+++ samples/plugins/scenario/scenario_plugin.py
@@ -13,13 +13,14 @@
 # License for the specific language governing permissions and limitations
 # under the License.
 
-from rally.task.scenarios import base
+from rally.plugins.openstack import scenario
+from rally.task import atomic
 
 
-class ScenarioPlugin(base.Scenario):
+class ScenarioPlugin(scenario.OpenStackScenario):
     """Sample plugin which lists flavors."""
 
-    @base.atomic_action_timer("list_flavors")
+    @atomic.action_timer("list_flavors")
     def _list_flavors(self):
         """Sample of usage clients - list flavors
 
@@ -28,12 +29,12 @@
         """
         self.clients("nova").flavors.list()
 
-    @base.atomic_action_timer("list_flavors_as_admin")
+    @atomic.action_timer("list_flavors_as_admin")
     def _list_flavors_as_admin(self):
         """The same with admin clients."""
         self.admin_clients("nova").flavors.list()
 
-    @base.scenario()
+    @scenario.configure()
     def list_flavors(self):
         """List flavors."""
         self._list_flavors()'''

neg_example = '''--- setup.py
+++ setup.py
@@ -4,7 +4,7 @@
 setup(
     name="punk",
 
-    version="1.0.1",
+    version="1.0.2",
 
     description="Primitives for Uncovering New Knowledge.",
     long_description="Machine Learning pipeline elements.",
@@ -12,7 +12,7 @@
     url="https://github.com/NewKnowledge/punk",
 
     author="New Knowledge",
-    author_email="alarcj137@gmail.com",
+    author_email="support@newknowledge.io",
 
     license="MIT",'''

prompt_template = """### Task
{task_description}

### Positive Example
{pos_example}

### Negative Example
{neg_example}

### Given Commit Diff
{diff}"""

@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", create_if_missing=True
        ),
        "/root/labeled_dataset": modal.Volume.from_name(
            "labeled_dataset", create_if_missing=True
        ),
    },
    secrets=[modal.Secret.from_name("huggingface-secret"),
            modal.Secret.from_name("openai-secret")],
)
async def main():
    import os
    import shutil
    import asyncio
    import tiktoken
    import subprocess
    from tqdm import trange
    from openai import AsyncOpenAI
    from datasets import load_dataset
    from huggingface_hub import HfApi, HfFolder, create_repo

    dataset_name = "commitpackft"

    enc = tiktoken.encoding_for_model("gpt-4o-mini")

    language_list = ["java", "go", "c", "c++", "javascript", "typescript"]

    for language in language_list:
        print(f"Processing {language} dataset...")
        dataset = load_dataset(f"{dataset_name}_filtered", data_files=f"data/{language}.jsonl", verification_mode="no_checks", split="train")
        print(f"Loaded {len(dataset)} commits from the dataset.")

        client = AsyncOpenAI()

        async def send_request(prompt: str, model: str, client: AsyncOpenAI):
            response = await client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                model=model,
            )
            
            return response.choices[0].message.content

        prompts = []

        for i in trange(len(dataset)):
            diff = dataset[i]["diff"]
            prompt = prompt_template.format(
                task_description=task_description,
                pos_example=pos_example,
                neg_example=neg_example,
                diff=diff
            )
            prompts.append(prompt)

        # 500 pieces of data per batch
        batch_size = 50
        responses = []
        for i in trange(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            responses.extend(await asyncio.gather(
                *[send_request(prompt, "gpt-4o-mini", client) for prompt in batch_prompts]
            ))

        # Label data based on the collected responses
        labels = [1 if response == "yes" else 0 for response in responses]
        dataset = dataset.add_column("label", labels)

        dataset.to_json(f"/root/labeled_dataset/{language}.jsonl")
    
    username = ""
    repo_name = f"{dataset_name}_labeled"
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

    shutil.copytree("/root/labeled_dataset", "./data", dirs_exist_ok=True)

    subprocess.run(["huggingface-cli", "lfs-enable-largefiles", "."])
    subprocess.run(["git", "lfs", "track", "*.jsonl"])
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", f"{dataset_name}"])
    subprocess.run(["git", "push"])