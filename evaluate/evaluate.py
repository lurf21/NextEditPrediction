import re
import json
import difflib
import pandas as pd

def is_empty_line_modification(chunk):
    for line in chunk:
        if line.startswith('+') or line.startswith('-'):
            if line[1:].strip() != '':
                return False
    return True

def split_diff_into_chunks(original, modified):
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()
    diff = list(difflib.unified_diff(original_lines, modified_lines, lineterm=''))
    chunks = []
    chunk_start_num = []
    chunk_to_remove = []

    for line_num, line in enumerate(diff):
        if line.startswith('@@'):
            chunk_start_num.append(line_num)

    for i in range(len(chunk_start_num) - 1):
        start = chunk_start_num[i]
        end = chunk_start_num[i + 1]
        chunk = diff[start:end]
        chunks.append(chunk)

    if chunk_start_num:
        chunks.append(diff[chunk_start_num[-1]:])

    # Remove chunks that are empty line modifications
    for i in range(len(chunks)):
        if is_empty_line_modification(chunks[i]):
            chunk_to_remove.append(i)

    chunks = [chunk for i, chunk in enumerate(chunks) if i not in chunk_to_remove]

    return chunks

def extract_changed_line_numbers(original, modified):
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()

    diff = list(difflib.unified_diff(original_lines, modified_lines, lineterm=''))

    changed_lines = set()
    orig_line_num = 0

    for line in diff:
        if line.startswith('---') or line.startswith('+++'):
            continue
        if line.startswith('@@'):
            match = re.match(r'@@ -(\d+)', line)
            if match:
                orig_line_num = int(match.group(1))
            continue
        if line.startswith(' '):
            orig_line_num += 1
        elif line.startswith('-'):
            changed_lines.add(orig_line_num)
            orig_line_num += 1
        elif line.startswith('+'):
            changed_lines.add(orig_line_num)
            orig_line_num += 1

    return changed_lines

def partial_match(original, edit1, edit2):
    chunks1 = split_diff_into_chunks(original, edit1)
    chunks2 = split_diff_into_chunks(original, edit2)

    if chunks1[0] in chunks2:
        return True

def position_match(original, edit1, edit2):
    chunks1 = split_diff_into_chunks(original, edit1)
    chunks2 = split_diff_into_chunks(original, edit2)

    context1 = []
    context2 = []

    for chunk in chunks1:
        context = [line for line in chunk if line.startswith(' ')]
        context1.append(context)

    for chunk in chunks2:
        context = [line for line in chunk if line.startswith(' ')]
        context2.append(context)

    if context1[0] in context2:
        return True

    return False

def main():
    original_code_list = []
    ground_truth_list = []
    model_output_list = []

    with open("../dataset/crawl/test.jsonl", "r") as f:
        lines = f.readlines()
        original_code_list = [json.loads(line)["current_contents"] for line in lines]

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

    summary = []

    for model_name in models:
        with open (f"generation_results/{model_name}_generation_results.jsonl", "r") as f:
            lines = f.readlines()
            ground_truth_list = [json.loads(line)["ground_truth"] for line in lines]
            model_output_list = [json.loads(line)["model_output"] for line in lines]

        results = []
        with open(f"llm_as_a_judge_results/{model_name}_llm_as_a_judge_results.jsonl", "r") as f:
            for line in f:
                if json.loads(line)["response"] == "yes":
                    results.append({"llm_as_a_judge": 1})
                else:
                    results.append({"llm_as_a_judge": 0})

        for idx, (original_code, edited_code_1, edited_code_2) in enumerate(zip(original_code_list, ground_truth_list, model_output_list)):
            if edited_code_1 == edited_code_2:
                results[idx]["exact_match"] = 1
            else:
                results[idx]["exact_match"] = 0
            if partial_match(original_code, edited_code_1, edited_code_2):
                results[idx]["partial_match"] = 1
            else:
                results[idx]["partial_match"] = 0
            if position_match(original_code, edited_code_1, edited_code_2):
                results[idx]["position_match"] = 1
            else:
                results[idx]["position_match"] = 0

        exact_acc = round(sum(result["exact_match"] for result in results) / len(results) * 100, 2)
        partial_acc = round(sum(result["partial_match"] for result in results) / len(results) * 100, 2)
        position_acc = round(sum(result["position_match"] for result in results) / len(results) * 100, 2)
        llm_as_a_judge = round(sum(result["llm_as_a_judge"] for result in results) / len(results) * 100, 2)

        count_1, count_2, count_3 = 0, 0, 0
        consistency_1, consistency_2, consistency_3 = 0, 0, 0

        for result in results:
            if result["exact_match"] == result["llm_as_a_judge"]:
                count_1 += 1
            if result["partial_match"] == result["llm_as_a_judge"]:
                count_2 += 1
            if result["position_match"] == result["llm_as_a_judge"]:
                count_3 += 1

        if count_1 > 0:
            consistency_1 = round(count_1 / len(results) * 100, 2)
        if count_2 > 0:
            consistency_2 = round(count_2 / len(results) * 100, 2)
        if count_3 > 0:
            consistency_3 = round(count_3 / len(results) * 100, 2)

        print(f"Model: {model_name}, Consistency with LLM as a Judge - Exact Match: {consistency_1}, Partial Match: {consistency_2}, Position Match: {consistency_3}")

        summary.append({
            "model_name": model_name,
            "exact_match_acc": exact_acc,
            "partial_match_acc": partial_acc,
            "position_match_acc": position_acc,
            "llm_as_a_judge": llm_as_a_judge,
        })

        with open(f"evaluation_results/{model_name}_evaluation_results.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Model: {model_name}, Exact Match Accuracy: {exact_acc}, Partial Match Accuracy: {partial_acc}, Position Match Accuracy: {position_acc}, LLM as a Judge: {llm_as_a_judge}")

    df = pd.DataFrame(summary)
    df.to_csv("evaluation_summary.csv", index=False)

if __name__ == "__main__":
    main()