import yaml
import os
from peft.tuners.lora import LoraLayer
from datasets import load_dataset


def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def upcast_layer_for_flash_attention(model, torch_dtype):
    # LlamaRMSNorm layers are in fp32 after kbit_training, so we need to
    # convert them back to fp16/bf16 for flash-attn compatibility.
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.to(torch_dtype)
        if "norm" in name:
            module.to(torch_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(torch_dtype)

    return model

def list_model_dirs(base_dirs):
    model_dirs = []
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            for model_name in os.listdir(base_dir):
                model_path = os.path.join(base_dir, model_name)
                if os.path.isdir(model_path):
                    model_dirs.append(model_path)
    return model_dirs

################## DATASETS ##################

DEFAULT_SYSTEM_MESSAGE = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

def get_ultrafeedback(tokenizer):
  def format_ultrafeedback(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
    # Prepend a system message if the first message is not a system message
    prompt_messages = example["chosen"]
    if example["chosen"][0]["role"] != "system":
        prompt_messages.insert(0, {"role": "system", "content": default_system_message})
    # apply template to the messages and return
    return {"prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False)}

  dataset = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')
  return dataset.map(format_ultrafeedback, remove_columns=dataset.features, fn_kwargs={"tokenizer": tokenizer})

def get_ultrafeedback_dpo(tokenizer):  
    def format_ultrafeedback_dpo(example, tokenizer, default_system_message=DEFAULT_SYSTEM_MESSAGE):
    
        def rec_extract_assistant_messages(messages, index=-1):
            """Recursively extract the last assistant messages from the end of the conversation."""
            if messages[index]["role"] == "assistant":
                return [messages[index]]
            else:
                return rec_extract_assistant_messages(messages, index-1)
    
        # Extract the N-1 turns to form the prompt
        # Prepend a system message if the first message is not a system message
        prompt_messages = example["chosen"][:-1]
        if example["chosen"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": default_system_message})
        # Now we extract the final assistant turn to define chosen/rejected responses 
        chosen_messages = rec_extract_assistant_messages(example["chosen"])
        rejected_messages = rec_extract_assistant_messages(example["rejected"])
        
        # apply template to the messages and return the triplets
        return {
            "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False),
            "chosen": tokenizer.apply_chat_template(chosen_messages, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        }
    dataset = load_dataset('argilla/ultrafeedback-binarized-preferences-cleaned', split='train')
    dataset = dataset.map(format_ultrafeedback_dpo, remove_columns=dataset.features, fn_kwargs={"tokenizer": tokenizer})

def get_ultrafeedback_kto(tokenizer): 
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned-kto", split="train")

    # Apply chat template
    def get_ultrafeedback_kto(example):
        # example["prompt"] = tokenizer.apply_chat_template(example["prompt"], tokenize=False)
        # example["completion"] = tokenizer.apply_chat_template(example["completion"], tokenize=False)
        # return example

        example["prompt"] = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        example["completion"] = f"{example['completion']}<|im_end|>"
        return example

    return dataset.map(get_ultrafeedback_kto)
