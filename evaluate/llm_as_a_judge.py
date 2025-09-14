import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from prompts import LLM_AS_A_JUDGE_PROMPT

load_dotenv()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

async def send_request(prompt: str, model: str, client: AsyncOpenAI):
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
    )
    
    return response.choices[0].message.content

async def main():
    llm_as_a_judge_results_dir = "llm_as_a_judge_results"
    os.makedirs(llm_as_a_judge_results_dir, exist_ok=True)

    models = [
        "gpt-4o", # gpt-4o-2024-08-06
        "gpt-4o-mini", # gpt-4o-mini-2024-07-18
        "gpt-4.1", # gpt-4.1-2025-04-14
        "gpt-4.1-mini", # gpt-4.1-mini-2025-04-14
        "gpt-4.1-nano", # gpt-4.1-nano-2025-04-14

        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",

        "gemini-2.5-pro",
        "gemini-2.5-flash",

        "deepseek-chat", # DeepSeek-V3-0324
        "deepseek-reasoner", # DeepSeek-R1-0528

        "Qwen2.5-Coder-3B-NEP",
        "Qwen2.5-Coder-7B-NEP",
        "Qwen2.5-Coder-14B-NEP",
        "Qwen2.5-Coder-32B-NEP",

        "codegemma-2b-NEP",
        "codegemma-7b-NEP",

        "Codestral-22B-v0.1-NEP",

        "Qwen2.5-Coder-3B",
        "Qwen2.5-Coder-7B",
        "Qwen2.5-Coder-14B",

        "Codestral-22B-v0.1",

        "Qwen2.5-Coder-7B-Instruct",
        "codegemma-7b-it",

        "Qwen2.5-Coder-7B-Instruct-NEP",
        "codegemma-7b-it-NEP",
    ]

    for model in models:
        print(f"Processing model: {model}")

        prompt_list = []
        with open("../dataset/crawl/prompts.jsonl", "r") as f:
            for line in f:
                result = json.loads(line)
                prompt = result['prompt']
                prompt_list.append(prompt)

        evaluation_prompt_list = []
        with open(f"generation_results/{model}_generation_results.jsonl", "r") as f:
            for index, line in enumerate(f):
                result = json.loads(line)
                prompt = prompt_list[index]
                ground_truth = result['ground_truth']
                model_output = result['model_output']

                evaluation_prompt = LLM_AS_A_JUDGE_PROMPT.format(
                    prompt=prompt,
                    ground_truth=ground_truth,
                    model_output=model_output
                )

                evaluation_prompt_list.append(evaluation_prompt)

        batch_size = 50
        responses = []
        for i in range(0, len(evaluation_prompt_list), batch_size):
            batch_prompts = evaluation_prompt_list[i:i + batch_size]
            batch_requests = [send_request(prompt, "gpt-4.1-mini", openai_client) for prompt in batch_prompts]
            batch_responses = await asyncio.gather(*batch_requests)
            responses.extend(batch_responses)
            print(f"Processed {i + len(batch_responses)} responses out of {len(evaluation_prompt_list)}")

        correct_predictions = 0
        for i, response in enumerate(responses):
            if response == "yes":
                correct_predictions += 1
        accuracy = round((correct_predictions / len(responses)) * 100, 2)
        print(f"Model: {model}, LLM-as-a-Judge Accuracy: {accuracy}")

        with open(f"llm_as_a_judge_results/{model}_llm_as_a_judge_results.jsonl", "w") as f:
            for i, response in enumerate(responses):
                f.write(json.dumps({"response": response}) + "\n")

        print(f"Evaluation results saved to llm_as_a_judge_results/{model}_llm_as_a_judge_results.jsonl")


if __name__ == "__main__":
    asyncio.run(main())