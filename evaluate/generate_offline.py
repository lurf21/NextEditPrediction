import modal
from prompts import SYSTEM_INSTRUCTION, PROMPT_TEMPLATE

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})  # faster model transfers
    .env({"VLLM_USE_V1": "1"})
    .add_local_file("../dataset/crawl/test.jsonl", "/root/test.jsonl")
    .add_local_file("prompts.py", "/root/prompts.py")
)

app = modal.App("generate_offline", image=vllm_image)

N_GPU = 1

@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name(
            "huggingface-cache", create_if_missing=True
        ),
        "/root/.cache/vllm": modal.Volume.from_name(
            "vllm-cache", create_if_missing=True
        ),
        "/root/generation_results": modal.Volume.from_name(
            "generation_results", create_if_missing=True
        ),
    },
    timeout=60 * 60 * 1,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def main():
    import gc
    import json
    import torch
    import difflib
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    with open("test.jsonl", "r") as f:
        lines = f.readlines()

    def get_prompt(line):
        prompt = json.loads(line)['text'].split('<|next_version|>\n')[0]
        prompt += "<|next_version|>\n"
        return prompt

    def get_instruction_prompt(line):
        commit = json.loads(line)
        diff = difflib.unified_diff(
            commit['old_contents'].splitlines(),
            commit['current_contents'].splitlines(),
            lineterm='',
            fromfile=commit['old_file'],
            tofile=commit['new_file'],
        )
        prompt = PROMPT_TEMPLATE.format(
            original_code=commit['old_contents'],
            edits="\n".join(diff),
            current_version=commit['current_contents'],
        )
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        return messages

    prompts = [get_prompt(line) for line in lines]
    instruction_prompts = [get_instruction_prompt(line) for line in lines]

    # https://docs.vllm.ai/en/v0.8.1/api/inference_params.html
    sampling_params = SamplingParams(temperature=0, max_tokens=8192) # greedy sampling

    sft_models = [
        "Qwen2.5-Coder-3B-NEP",
        "Qwen2.5-Coder-7B-NEP",
        "Qwen2.5-Coder-14B-NEP",
        "Qwen2.5-Coder-32B-NEP",

        "codegemma-2b-NEP",
        "codegemma-7b-NEP",

        "Codestral-22B-v0.1-NEP",

        "Qwen2.5-Coder-7B-Instruct-NEP",
        "codegemma-7b-it-NEP",
    ]

    base_models = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "google/codegemma-7b-it",
    ]

    models = sft_models + base_models

    for model in models:
        print(f"Loading model: {model}")
        llm = LLM(model=model, tensor_parallel_size=N_GPU)

        generation_results = []

        if model in sft_models:
            outputs = llm.generate(prompts, sampling_params)
        elif model in base_models:
            if "codegemma" in model:
                new_instruction_prompts = []
                for old_prompt in instruction_prompts:
                    new_instruction_prompts.append(
                        [
                            {"role": "user", "content": SYSTEM_INSTRUCTION + "\n" + old_prompt[1]['content']},
                        ]
                    )
                outputs = llm.chat(new_instruction_prompts, sampling_params)
            else:
                outputs = llm.chat(instruction_prompts, sampling_params)
        
        for line, output in zip(lines, outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text

            def extract(text, a, b):
                try:
                    start = text.index(a) + len(a)
                    end = text.index(b, start)
                    return text[start:end]
                except:
                    return text

            start_tag = "<next_version>\n"
            end_tag = "\n</next_version>"

            generated_text = extract(generated_text, start_tag, end_tag)

            generation_results.append({
                "prompt": prompt,
                "model_output": generated_text,
                "ground_truth": json.loads(line)['text'].split('<|next_version|>\n')[1],
            })
        
        with open(f"generation_results/{model.split('/')[1]}_generation_results.jsonl", "w") as f:
            for result in generation_results:
                f.write(json.dumps(result) + "\n")
        
        print(f"Results saved to generation_results/{model.split('/')[1]}_generation_results.jsonl")

        del llm
        gc.collect()
        torch.cuda.empty_cache()
