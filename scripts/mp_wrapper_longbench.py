import argparse
import os
import subprocess
import json

MP_INPUT_DIR = 'mp/input'
MP_OUTPUT_DIR = 'mp/output'
SUB_TASK_FILES = [
    'gov_report.jsonl',
    'musique.jsonl',
    'narrativeqa.jsonl',
    'passage_retrieval_en.jsonl',
    'qmsum.jsonl',
]

def main(args):
    os.makedirs(MP_INPUT_DIR, exist_ok=True)
    os.makedirs(MP_OUTPUT_DIR, exist_ok=True)

    data = {}
    for file_name in SUB_TASK_FILES: 
        with open(os.path.join(args.input_dir, file_name), 'r') as f:
            data[file_name] = [json.loads(line) for line in f]

    # split input data
    mp_input_data ={key : [[] for _ in range(args.num_process)] for key in data.keys()}
    for key, sub_data in data.items():
        for i, d in enumerate(sub_data):
            mp_input_data[key][i % args.num_process].append(d)

    mp_input_dirs = prepare_split_filename(MP_INPUT_DIR, args.input_dir, args.num_process)
    print('mp_input_dirs: ', mp_input_dirs)
    for mp_input_dir in mp_input_dirs:
        os.makedirs(mp_input_dir, exist_ok=True)

    for file_name in data.keys():
        for mp_input_dir, mp_input_d in zip(mp_input_dirs, mp_input_data[file_name]):
            mp_input_file = os.path.join(mp_input_dir, file_name)
            if os.path.exists(mp_input_file):
                continue

            with open(mp_input_file, 'w') as f:
                for d in mp_input_d:
                    f.write(json.dumps(d) + '\n')
        
    # prepare output filename
    mp_output_files = prepare_split_filename(MP_OUTPUT_DIR, args.output_file, args.num_process)
    print('mp_output_files: ', mp_output_files)

    # run subprocess 
    processes = []
    for rank in range(args.num_process):
        processes.append(run_command(rank, args, mp_input_dirs[rank], mp_output_files[rank]))
    
    for p in processes:
        p.wait()

    # collect output
    collect_output(args.output_file, mp_output_files, args.num_process)

def prepare_split_filename(dir, file_path, num_process):
    return [os.path.join(dir, f'{os.path.basename(file_path)}_{i}_in_{num_process}') for i in range(num_process)]

def collect_output(output_file, mp_output_files, num_process):
    mp_output_res = [[] for _ in range(num_process)]
    for i, mp_output_file in enumerate(mp_output_files):
        with open(mp_output_file, 'r') as f:
            mp_output_res[i] = [json.loads(line) for line in f]

    output = []
    len_output0 = len(mp_output_res[0])
    for i in range(len_output0):
        for j in range(args.num_process):
            if i >= len(mp_output_res[j]):
                break
            output.append(mp_output_res[j][i])
    
    with open(output_file, 'w') as f:
        for o in output:
            f.write(json.dumps(o) + '\n')

def run_command(rank: int, args, input_dir, output_file):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(rank)
    command = ['python', args.script] + args.subprocess_args \
                + ['--input_dir', input_dir, '--output_path', output_file]
    print('command: ', command)
    return subprocess.Popen(command, env=env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, required=True)
    parser.add_argument('--num_process', type=int, required=True)

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--subprocess_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    main(args)
    