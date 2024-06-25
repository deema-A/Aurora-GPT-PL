#!/usr/bin/env python3
import os
import sys
import subprocess
from utils import *

# Function to count the number of GPUs available, based on command line argument.
# By default, this returns 4. If you want to use a different number of GPUs, pass it as the first argument.
def gpu_count():
    if len(sys.argv) >= 2:  # if n GPU is preselected as an argument
        try:
            return int(sys.argv[1])
        except Exception as e:
            print(e)
            print("First argument should be gpu count!")
            exit(1)

    return 4

# Function to generate a command for model evaluation. This includes various checks and formatting.
def generate_command(model, task, fewshot, CUDA_VISIBLE_DEVICES=0, bf16=True, write_out=False, overwrite=False):
    if not os.path.exists(model):
        print(f"Skipping model {model} for {task} because the model doesn't exist", file=sys.stderr)
        return None

    # Set batch size based on task type. bsz = 1 in most cases to avoid padding.
    if task.startswith("hendrycksTest"):
        bsz = 4
    else:
        bsz = 1

    # Prepare the base command with CUDA and lm_eval configuration
    base_command = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} lm_eval --model=hf --batch_size={bsz}"

    # Setting up directories and file names for metrics and generation files
    output_dir = "openllm_leaderboard_result"
    model_formatted = os.path.basename(model)
    if model_formatted.startswith("checkpoint_"):
        stage = os.path.basename(os.path.dirname(model))  # keep the parent dir name
        model_formatted = f"{stage}@{model_formatted}"
    model_formatted = model_formatted.replace("/", "@")
    if bf16:
        model_formatted += "+bf16"

    metric_file = f"{output_dir}/metrics/{model_formatted}/{task}_{fewshot}_shot.json"
    metric_file = metric_file.replace("*", "")

    generation_file_dir = f"{output_dir}/generations/{model_formatted}/{task}_{fewshot}_shot_gen.out"
    generation_file_dir = generation_file_dir.replace("*", "")

    debug_generation_file_dir = generation_file_dir + ".debug"
    dirname = os.path.dirname(debug_generation_file_dir)
    os.makedirs(dirname, exist_ok=True)

    # Skip if metric file exists and overwrite is not allowed
    if not overwrite and os.path.exists(metric_file):
        print(f"Skipping {metric_file} because it already exists", file=sys.stderr)
        return None

    # Constructing the model argument for the command
    model_arg = f"pretrained={model},trust_remote_code=True"
    if bf16:
        model_arg += ",dtype=bfloat16"
    command = (f"{base_command} --model_args=\"{model_arg}\" "
               f"--tasks={task} --num_fewshot={fewshot} --output_path={metric_file}")
    if write_out:
        command += f" --write_out --output_base_path={generation_file_dir}"

    # run in background
    command += f" >{debug_generation_file_dir}"
    command += " &"
    return command


if __name__ == "__main__":
    config_file = "src/eval_config.yaml"
    config = load_yaml_config(config_file)

    # Base directories containing models
    base_dirs = ["./models/SFT", "./models/DPO", "./models/KTO"]
    
    # List all model directories in the specified base directories
    models = list_model_dirs(base_dirs)

    task_fewshot_pairs = config['task_fewshot_pairs']

    # Print loaded configuration for verification
    print("Models to evaluate:", models)
    print("Task and few-shot pairs to evaluate:", task_fewshot_pairs)

    # Variable for tracking GPU ID assignment
    next_id = 0
    # List to store the generated commands
    commands = []

    for task, fewshot in task_fewshot_pairs:
        gpu_cnt = gpu_count()

        for model in models:
            command = generate_command(model, task, fewshot, CUDA_VISIBLE_DEVICES=next_id)
            if command is None:
                continue
            next_id = (next_id + 1) % gpu_cnt
            commands.append((next_id, command))
            if next_id == 0:
                commands.append((next_id, "wait $(jobs -rp)"))

    # Execute the commands
    for command in commands:
        try:
            subprocess.run(command[1], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running command: {command[1]}\nError: {e}")
