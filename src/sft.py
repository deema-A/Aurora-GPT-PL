from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, setup_chat_format
from utils import *
import torch
import argparse


def train_model(model_id, dataset_name):
    # Load Tokenizer from the hub
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        use_cache=False, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    model, tokenizer = setup_chat_format(model, tokenizer)
    
    dataset_functions = {
        "ultrafeedback": get_ultrafeedback
    }
    dataset = dataset_functions[dataset_name](tokenizer) # depends on the user passed dataset

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Upcast layer for flash attention
    model = upcast_layer_for_flash_attention(model, torch.bfloat16)
    model = get_peft_model(model, peft_config)

    model_name_part = model_id.split('/')[1].split('-')[0]
    args = TrainingArguments(
        output_dir=f"./models/SFT/{model_name_part}_{dataset_name}",
        num_train_epochs=3,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=4,           # batch size for evaluation
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        # fp16=False,
        # tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,  # disable tqdm since with packing values are incorrect
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        dataset_text_field="prompt", 
        args=args,
    )
    # Train
    trainer.train()  # there will not be a progress bar since tqdm is disabled

    # Save model
    trainer.save_model(f"./models/SFT/{model_name_part}_{dataset_name}")

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train multiple models with different strategies.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to be used for training')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name to be used for training')
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_id = args.model_id
    dataset_name = args.dataset_name

    train_model(model_id=model_id, dataset_name=dataset_name)

if __name__ == "__main__":
    main()