import argparse
import os
import subprocess
import json
import pickle
from glob import glob
import numpy as np
from tqdm import tqdm
import math
from transformers import PreTrainedTokenizer
from lift.model import load_tokenizer

MP_INPUT_DIR = 'mp/input'
MP_OUTPUT_DIR = 'mp/output'

def generate_sample(tokenizer: PreTrainedTokenizer, context: str, context_length: int, needle_depth: int, needle: str, prompt: str, zh: bool):
    if zh:
        num_words = len(context)
    else:
        num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)

    if zh:
        description = "以下上下文中隐藏着重要信息。找到并记住这些信息。我会问你关于其中重要信息的问题。\n"
    else:
        description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)
    prompt_input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)
    prompt_length = len(prompt_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - prompt_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length - prompt_length, truncation=True, add_special_tokens=False)

    context_ids = sum([context_input_ids[:needle_pos], needle_input_ids, context_input_ids[needle_pos:]], [])
    context_return = tokenizer.decode(context_ids)

    return prompt, context_return, needle


def generate_niah_input(cache_input_path, haystack_path, num_samples_per_case, test_length, test_depth, needle, prompt, zh, tokenizer: PreTrainedTokenizer):
    if os.path.isfile(haystack_path):
        raise ValueError("Expected a directory but got a file.")
    elif os.path.isdir(haystack_path):
        contexts = []
        haystack_files = [file for file in glob(f"{haystack_path}/*.txt")]
        for i in range(num_samples_per_case):
            num_tokens = 0
            contexts.append("")
            np.random.shuffle(haystack_files)
            for file in haystack_files:
                with open(file, 'r') as f:
                    this_file_context = f.read()
                    num_tokens += len(tokenizer(this_file_context, add_special_tokens=False))
                    contexts[i] += this_file_context
                    if num_tokens > max(test_length):
                        break
    else:
        raise ValueError(f"Cannot find haystack: {haystack_path}")
    all_inputs = []
    for length in tqdm(test_length, desc="Constructing Data"):
        for depth in test_depth:
            for i in range(num_samples_per_case):
                prompt_case, context_case, needle_case = generate_sample(
                    tokenizer=tokenizer,
                    context=contexts[i],
                    context_length=length, 
                    needle_depth=depth,
                    needle=needle,
                    prompt=prompt,
                    zh=zh
                )
                all_inputs.append(dict(
                    prompt=prompt_case,
                    context=context_case,
                    needle=needle_case,
                    length=length,
                    depth=depth
                ))
    if cache_input_path is not None:
        with open(cache_input_path, 'wb') as f:
            pickle.dump(all_inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return all_inputs

def main(args):
    os.makedirs(MP_INPUT_DIR, exist_ok=True)
    os.makedirs(MP_OUTPUT_DIR, exist_ok=True)

    # generate if no input cache
    if not os.path.exists(args.cache_input_path):
        all_inputs = generate_niah_input(args.cache_input_path, args.haystack_path, args.num_samples_per_case, args.test_length, args.test_depth, args.needle, args.prompt, args.zh, load_tokenizer(args.model_name_or_path))
    else:
        with open(args.cache_input_path, "rb") as f:
            all_inputs = pickle.load(f)

    # split input data
    mp_input_data = [[] for _ in range(args.num_process)]
    for i, d in enumerate(all_inputs):
        mp_input_data[i % args.num_process].append(d)

    mp_cache_input_paths = prepare_split_filename(MP_INPUT_DIR, args.cache_input_path, args.num_process)
    print('mp_cache_input_paths: ', mp_cache_input_paths)

    for mp_cache_input_path, mp_input_d in zip(mp_cache_input_paths, mp_input_data):
        with open(mp_cache_input_path, 'wb') as f:
            pickle.dump(mp_input_d, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # prepare output filename
    mp_output_paths = prepare_split_filename(MP_OUTPUT_DIR, args.output_path, args.num_process)
    print('mp_output_paths: ', mp_output_paths)

    # run subprocess 
    processes = []
    for rank in range(args.num_process):
        processes.append(run_command(rank, args, mp_cache_input_paths[rank], mp_output_paths[rank]))
    
    for p in processes:
        p.wait()

    # collect output
    collect_output(args.output_path, mp_output_paths, args.num_process)

def prepare_split_filename(dir, file_path, num_process):
    return [os.path.join(dir, f'{os.path.basename(file_path)}_{i}_in_{num_process}') for i in range(num_process)]

def collect_output(output_path, mp_output_paths, num_process):
    mp_output_res = [[] for _ in range(num_process)]
    for i, mp_output_path in enumerate(mp_output_paths):
        with open(mp_output_path, 'r') as f:
            mp_output_res[i] = [json.loads(line) for line in f]

    output = []
    len_output0 = len(mp_output_res[0])
    for i in range(len_output0):
        for j in range(args.num_process):
            if i >= len(mp_output_res[j]):
                break
            output.append(mp_output_res[j][i])
    
    with open(output_path, 'w') as f:
        for o in output:
            f.write(json.dumps(o) + '\n')

def run_command(rank: int, args, cache_input_path, output_path):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(rank)
    command = ['python', args.script] + args.subprocess_args \
                + ['--cache_input_path', cache_input_path,
                   '--output_path', output_path,
                   '--haystack_path', args.haystack_path,
                   '--num_samples_per_case', args.num_samples_per_case,
                   '--test_length', *args.test_length,
                   '--test_depth', *args.test_depth,
                   '--needle', args.needle,
                   '--prompt', args.prompt,
                   '--zh', args.zh,
                   '--model_name_or_path', args.model_name_or_path,
                   ]
    command = list(map(str, command))
    print('command: ', command)
    return subprocess.Popen(command, env=env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, required=True)
    parser.add_argument('--num_process', type=int, required=True)

    parser.add_argument('--cache_input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--haystack_path', type=str, default="long-llm:needle/PaulGrahamEssays")
    parser.add_argument('--num_samples_per_case', type=int, default=1)
    parser.add_argument('--test_length', type=int, nargs='*', default=[]) 
    parser.add_argument('--test_depth', type=int, nargs='*', default=[])
    parser.add_argument('--needle', type=str, default="\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n")
    parser.add_argument('--prompt', type=str, default='\n\nWhat is the best thing to do in San Francisco?\nAnswer:')
    parser.add_argument('--zh', type=bool, default=False)

    parser.add_argument('--model_name_or_path', type=str, required=True)

    parser.add_argument('--subprocess_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    main(args)
    