import os
import json
import difflib
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google.genai.client import Client
from google.genai.types import GenerateContentConfig
from tqdm.asyncio import tqdm_asyncio
from prompts import SYSTEM_INSTRUCTION, PROMPT_TEMPLATE

load_dotenv()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

anthropic_client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

gemini_client = Client(
    api_key=os.getenv("GEMINI_API_KEY"),
)

deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

async def send_request(client, model_name, model_group, prompt):
    MAX_OUTPUT_TOKENS = 8192

    if model_group == "openai" or model_group == "deepseek":     
        completion = await client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )
        print(completion)
        response = completion.choices[0].message.content
    elif model_group == "anthropic":
        completion = await client.messages.create(
            model=model_name,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_INSTRUCTION,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        print(completion)
        response = completion.content[0].text
    elif model_group == "gemini":
        completion = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        print(completion.text)
        response = completion.text

    if response is None or response.strip() == "":
        return None, False

    if response.startswith("```"):
        first_newline = response.find("\n")
        response = response[first_newline + 1:-4]

    if response.endswith("\n"):
        response = response[:-1]

    start_tag = "<next_version>\n"
    end_tag = "\n</next_version>"
    
    if response.startswith(start_tag) and response.endswith(end_tag):
        response = response[len(start_tag):-len(end_tag)]
        return response, True
    else:
        return None, False

async def process_line(client, line, model_name, model_group, semaphore):
    async with semaphore:
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
        ground_truth = commit['new_contents']

        valid = False
        while not valid:
            response, valid = await send_request(client, model_name, model_group, prompt)

        return {
            "prompt": prompt,
            "model_output": response,
            "ground_truth": ground_truth,
        }

async def main():
    generation_results_dir = "generation_results"
    os.makedirs(generation_results_dir, exist_ok=True)

    openai_models = [
        "gpt-4o", # gpt-4o-2024-08-06
        "gpt-4o-mini", # gpt-4o-mini-2024-07-18
        "gpt-4.1", # gpt-4.1-2025-04-14
        "gpt-4.1-mini", # gpt-4.1-mini-2025-04-14
        "gpt-4.1-nano", # gpt-4.1-nano-2025-04-14
    ]

    anthropic_models = [
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]

    gemini_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    deepseek_models = [
        "deepseek-chat", # DeepSeek-V3-0324
        "deepseek-reasoner", # DeepSeek-R1-0528
    ]

    models = openai_models + anthropic_models + gemini_models + deepseek_models

    for model in models:
        print(f"Processing model: {model}")
        # Determine the client and model group based on the model name
        if model in openai_models:
            client, model_group = openai_client, "openai"
        elif model in anthropic_models:
            client, model_group = anthropic_client, "anthropic"
        elif model in gemini_models:
            client, model_group = gemini_client, "gemini"
        elif model in deepseek_models:
            client, model_group = deepseek_client, "deepseek"

        generation_results = []
        max_concurrent_requests = 10
        semaphore = asyncio.Semaphore(max_concurrent_requests)

        with open("../dataset/crawl/test.jsonl", "r") as f:
            lines = f.readlines()

        tasks = [process_line(client, line, model, model_group, semaphore) for line in lines]
        generation_results = await tqdm_asyncio.gather(*tasks, desc="Processing lines")

        with open(f"{generation_results_dir}/{model.split('/')[-1]}_generation_results.jsonl", "w") as f:
            for result in generation_results:
                f.write(json.dumps(result) + "\n")

        print(f"Results saved to {generation_results_dir}/{model.split('/')[-1]}_generation_results.jsonl")

if __name__ == "__main__":
    asyncio.run(main())