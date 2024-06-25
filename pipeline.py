import argparse
import json
import random
import os
from datetime import datetime
import subprocess
import multiprocessing

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline for training and evaluating Hugging Face models.')
    parser.add_argument('--strategies', type=str, nargs='+', choices=['sft', 'dpo', 'kto'], default=['sft', 'dpo', 'kto'], help='Training strategies')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='Name of the model')
    parser.add_argument('--dataset_name', type=str, default="ultrafeedback", help='Name of the dataset')
    return parser.parse_args()

def get_random_filename(model_name, technique):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(1000, 9999)
    filename = f"{model_name}_{technique}_{random_number}_{timestamp}.json"
    return filename

def save_results(results, model_name, technique):
    filename = get_random_filename(model_name, technique)
    results_path = os.path.join('results', filename)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)
    print(f"Results saved to {results_path}")

def run_training_script(script_path, additional_args):
    cmd = ['python', script_path] + additional_args
    # cmd.extend(['--use_peft', '--lora_r', '64', '--lora_alpha', '16'])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}: {result.stderr}")
        return None
    return json.loads(result.stdout)

def execute_strategy(strategy, model_name, dataset_name):
    script_info = {
        'sft': {'script': 'src/sft.py', 'default_args': ['--dataset_name', dataset_name, '--model_id', strategy]},
        'dpo': {'script': 'src/dpo.py', 'default_args': ['--dataset_name', dataset_name, '--model_id', strategy]},
        'kto': {'script': 'src/kto.py', 'default_args': ['--dataset_name', dataset_name, '--model_id', strategy]}
    }.get(strategy)

    if script_info:
        print(f"Starting training with {strategy.upper()} strategy...")
        results = run_training_script(script_info['script'], script_info['default_args'])
        if results:
            save_results(results, model_name, strategy)
        else:
            print(f"Failed to run strategy: {strategy}")
    else:
        print(f"Invalid strategy selected: {strategy}")

def main():
    args = parse_arguments()
    strategies = args.strategies
    strategies = ['kto']
    model_name = args.model_name
    dataset_name = args.dataset_name

    # Use multiprocessing to run strategies in parallel
    processes = []
    for strategy in strategies:
        p = multiprocessing.Process(target=execute_strategy, args=(strategy, model_name, dataset_name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Run evaluation
    eval_cmd = ['python', 'src/eval.py']
    subprocess.run(eval_cmd)

if __name__ == "__main__":
    main()
