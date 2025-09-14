import os
import re
import random
import difflib
import argparse
import jsonlines
from tqdm import tqdm
from datasets import Dataset

def count_characters(example):
    text = example["old_contents"] + example["new_contents"]
    return len(text) <= 32768

def count_tokens(example, enc, max_tokens):
    try:
        text = example["old_contents"] + example["new_contents"]
        tokens = enc.encode(text)
        return 2 * len(tokens) <= max_tokens
    except Exception as e:
        print(f"Error in count_tokens: {e}")
        print(f"Example: {example['commit']}")
        return False

def file_ext(commit, ext):
    return commit["old_file"].endswith(ext) and commit["new_file"].endswith(ext)

def edit_chunk_num(diff_content):
    """
    Filters out entries in a diff that contain only one edit chunk.

    :param diff_content: str, content of the diff
    :return: bool, True if the diff has only one edit chunk, False otherwise
    """
    lines = diff_content.split('\n')
    lines = lines[2:]

    edit_chunks = 0
    in_edit_chunk = False

    for line in lines:
        if line.startswith('+') or line.startswith('-'):
            if not in_edit_chunk:
                edit_chunks += 1
                in_edit_chunk = True
        else:
            in_edit_chunk = False

    return edit_chunks != 1

def diff_code_snippets(code1, code2, file1, file2):
    diff = difflib.unified_diff(
        code1.splitlines(),
        code2.splitlines(),
        lineterm='',
        fromfile=file1,
        tofile=file2
    )

    return "\n".join(diff)

def compute_diff(example):
    return diff_code_snippets(example["old_contents"], example["new_contents"], example["old_file"], example["new_file"])

def edit_chunk_len(diff_content, max_len):
    """
    Ensure that the length of every edit chunk in a diff is less than a given threshold.
    
    :param diff_content: str, content of the diff
    :param max_len: int, maximum allowed length of an edit chunk
    :return: bool, True if all edit chunks are within the length limit, False otherwise
    """
    lines = diff_content.split('\n')
    lines = lines[2:]

    current_chunk_length = 0
    in_edit_chunk = False

    for line in lines:
        if line.startswith('+') or line.startswith('-'):
            if not in_edit_chunk:
                in_edit_chunk = True
                current_chunk_length = 1
            else:
                current_chunk_length += 1
        else:
            if in_edit_chunk and current_chunk_length > max_len:
                return False
            in_edit_chunk = False
            current_chunk_length = 0

    if in_edit_chunk and current_chunk_length > max_len:
        return False

    return True

def edit_window_size(diff_content, max_size):
    """
    Ensure that the gap between the first edit chunk and the last edit chunk in a diff is less than a given threshold.
    
    :param diff_content: str, content of the diff
    :param max_size: int, maximum allowed gap between the first and last edit chunks
    :return: bool, True if the gap between the first and last edit chunks is within the size limit, False otherwise
    """
    edit_chunks = re.findall(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', diff_content)

    if not edit_chunks:
        return False

    first_chunk = edit_chunks[0]
    last_chunk = edit_chunks[-1]

    original_first_line = int(first_chunk[0])
    original_last_line = int(last_chunk[0]) + int(last_chunk[1]) - 1
    
    edited_first_line = int(first_chunk[2])
    edited_last_line = int(last_chunk[2]) + int(last_chunk[3]) - 1

    original_window_size = original_last_line - original_first_line + 1
    edited_window_size = edited_last_line - edited_first_line + 1
    
    return original_window_size <= max_size and edited_window_size <= max_size

# To initially test the effectiveness of the training, only select data where all edit chunks are add.
def add_only(diff_content):
    for line in diff_content.split('\n')[2:]:
        if line.startswith('-'):
            return False

    return True

def filter_commit(commit, args):
    diff = diff_code_snippets(commit["old_contents"], commit["new_contents"], commit["old_file"], commit["new_file"])

    if not file_ext(commit, args.file_extension):
        return False
    elif not edit_chunk_num(diff):
        return False
    elif not edit_chunk_len(diff, args.max_edit_length):
        return False
    elif not edit_window_size(diff, args.max_window_size):
        return False
    elif not add_only(diff):
        return False
        
    return True

def extract_edit_context(commit, context_lines):
    """
    Extracts the context that surrounds the edit chunks
    
    :param commit: dict, commit data
    :param context_lines: int, number of lines of context to extract
    :return: str, extracted context
    """
    diff_content = diff_code_snippets(commit["old_contents"], commit["new_contents"], commit["old_file"], commit["new_file"])
    old_lines = commit["old_contents"].split('\n')
    new_lines = commit["new_contents"].split('\n')
    edit_chunks = re.findall(r'@@ -(\d+),(\d+) \+(\d+),(\d+) @@', diff_content)

    first_chunk = edit_chunks[0]
    last_chunk = edit_chunks[-1]

    original_first_line = int(first_chunk[0])
    original_last_line = int(last_chunk[0]) + int(last_chunk[1]) - 1
    
    edited_first_line = int(first_chunk[2])
    edited_last_line = int(last_chunk[2]) + int(last_chunk[3]) - 1

    extracted_old_contents = []
    extracted_new_contents = []

    for i in range(original_first_line - context_lines, original_last_line + context_lines + 1):
        if 0 <= i < len(old_lines):
            extracted_old_contents.append(old_lines[i])

    for i in range(edited_first_line - context_lines, edited_last_line + context_lines + 1):
        if 0 <= i < len(new_lines):
            extracted_new_contents.append(new_lines[i])

    return "\n".join(extracted_old_contents), "\n".join(extracted_new_contents)

def remove_random_edit_chunk(example):
    old_contents = example["old_contents"]
    new_contents = example["new_contents"]

    old_lines = old_contents.splitlines(keepends=True)
    new_lines = new_contents.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()
    
    insertions = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "insert":
            insertions.append((j1, j2))

    # If there are no insertions, return None as `current_contents`. Because SequenceMatcher
    # unlike e.g. UNIX(tm) diff, the fundamental notion is the longest *contiguous* 
    # & junk-free matching subsequence. See https://github.com/python/cpython/blob/3.13/Lib/difflib.py
    if insertions == []:
        return {
            "current_contents": None,
        }

    j1, j2 = random.choice(insertions)
    
    modified_lines = new_lines[:j1] + new_lines[j2:]
    current_contents = "".join(modified_lines)
    
    return {
        "current_contents": current_contents,
    }

def remove_last_edit_chunk(example):
    old_contents = example["old_contents"]
    new_contents = example["new_contents"]

    old_lines = old_contents.splitlines(keepends=True)
    new_lines = new_contents.splitlines(keepends=True)

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = matcher.get_opcodes()
    
    insertions = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "insert":
            insertions.append((j1, j2))

    if insertions == []:
        return {
            "current_contents": None,
        }

    j1, j2 = insertions[-1]
    
    modified_lines = new_lines[:j1] + new_lines[j2:]
    current_contents = "".join(modified_lines)
    
    return {
        "current_contents": current_contents,
    }

def build_text_field(example):
    template = """<|original_code|>
{old_contents}
<|edits_diff|>
{diff}
<|current_version|>
{current_contents}
<|next_version|>
{new_contents}
"""
    diff = difflib.unified_diff(
        example["old_contents"].splitlines(),
        example["current_contents"].splitlines(),
        lineterm='',
        fromfile=example["old_file"],
        tofile=example["new_file"],
    )
    prompt = template.format(
        old_contents=example["old_contents"],
        diff="\n".join(diff),
        current_contents=example["current_contents"],
        new_contents=example["new_contents"],
    )
    return {
        "text": prompt,
    }

def generate_dataset(args):
    dataset = []
    with jsonlines.open(f"{args.input_dir}/{args.repo_name.replace('/', '_')}_raw.jsonl", 'r') as reader:
        for commit in tqdm(reader):
            if filter_commit(commit, args):
                commit["old_contents"], commit["new_contents"] = extract_edit_context(commit, args.context_lines)
                dataset.append(commit)

    return dataset

def map_dataset(args, dataset):
    save_mode = "w" if not args.append else "a"
    dataset = Dataset.from_list(dataset)
    dataset = dataset.map(remove_last_edit_chunk)
    dataset = dataset.filter(lambda x: x["current_contents"] is not None)
    dataset = dataset.map(build_text_field)
    os.makedirs(args.output_dir, exist_ok=True)
    with jsonlines.open(f"{args.output_dir}/{args.repo_name.replace('/', '_')}_processed.jsonl", save_mode) as writer:
        for example in tqdm(dataset):
            writer.write(example)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--repo_name", required=True)
    parser.add_argument("--file_extension", default=".py")
    parser.add_argument("--max_edit_length", default=5)
    parser.add_argument("--max_window_size", default=80)
    parser.add_argument("--context_lines", default=20)
    parser.add_argument("--append", action="store_true")

    args = parser.parse_args()

    dataset = generate_dataset(args)
    map_dataset(args, dataset)


if __name__ == "__main__":
    main()
