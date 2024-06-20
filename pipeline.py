import argparse
import json
import random
import os
from datetime import datetime
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description='Pipeline for training and evaluating Hugging Face models.')
    parser.add_argument('--strategies', type=str, nargs='+', choices=['sft', 'dpo', 'kto'], default=['sft', 'dpo', 'kto'], help='Training strategies')
    parser.add_argument('--use_peft', action='store_true', default=True, help='Use PEFT')
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

def run_training_script(script_path, args, additional_args):
    cmd = ['python', script_path] + additional_args
    if args.use_peft:
        cmd.extend(['--use_peft', '--lora_r', '64', '--lora_alpha', '16'])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}: {result.stderr}")
        return None
    return json.loads(result.stdout)

def main():
    args = parse_arguments()
    
    strategies = args.strategies
    strategies = ['dpo']
    # Dictionary to map strategy to corresponding script path and default arguments
    strategy_scripts = {
        'sft': {
            'script': 'src/sft.py',
            'default_args': [
                '--model_name_or_path', 'meta-llama/Llama-2-7b-hf',
                '--report_to', 'wandb',
                '--learning_rate', '1.41e-5',
                '--per_device_train_batch_size', '64',
                '--gradient_accumulation_steps', '16',
                '--output_dir', './models/SFT',
                '--logging_steps', '1',
                '--num_train_epochs', '3',
                '--max_steps', '-1',
                '--push_to_hub',
                '--gradient_checkpointing'
            ]
        },
        'dpo': {
            'script': 'src/dpo.py',
            'default_args': [
                '--dataset_name', 'trl-internal-testing/hh-rlhf-helpful-base-trl-style',
                '--model_name_or_path', 'meta-llama/Llama-2-7b-hf',
                '--per_device_train_batch_size', '4',
                '--learning_rate', '1e-3',
                '--gradient_accumulation_steps', '1',
                '--logging_steps', '10',
                '--eval_steps', '500',
                '--output_dir', './models/DPO',
                '--warmup_steps', '150',
                '--report_to', 'wandb',
                '--bf16',
                '--logging_first_step',
                '--no_remove_unused_columns'
            ]
        },
        'kto': {
            'script': 'src/kto.py',
            'default_args': [
                '--model_name_or_path', 'meta-llama/Llama-2-7b-hf',
                '--per_device_train_batch_size', '16',
                '--num_train_epochs', '1',
                '--learning_rate', '1e-5',
                '--lr_scheduler_type', 'cosine',
                '--gradient_accumulation_steps', '1',
                '--logging_steps', '10',
                '--eval_steps', '500',
                '--output_dir', './models/KTO',
                '--warmup_ratio', '0.1',
                '--report_to', 'wandb',
                '--bf16',
                '--logging_first_step'
            ]
        }
    }

    # Execute the training strategies
    for strategy in strategies:
        if strategy in strategy_scripts:
            script_info = strategy_scripts[strategy]
            print(f"Starting training with {strategy.upper()} strategy...")
            results = run_training_script(script_info['script'], args, script_info['default_args'])
            if results:
                save_results(results, script_info['default_args'][1], strategy)
        else:
            print(f"Invalid strategy selected: {strategy}")

    # Run evaluation
    eval_cmd = ['python', 'src/eval.py']
    subprocess.run(eval_cmd)

if __name__ == "__main__":
    main()
