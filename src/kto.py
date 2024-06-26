from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import KTOTrainer, setup_chat_format, KTOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from utils import *
import torch
import argparse

def train_model(model_id, dataset_name):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                device_map="auto",
                                                use_cache=False, 
                                                attn_implementation="flash_attention_2",
                                                torch_dtype=torch.bfloat16,
                                                quantization_config=bnb_config)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    # If we are aligning a base model, we use ChatML as the default template
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    dataset = get_ultrafeedback_kto(tokenizer)

    # Initialize the KTO trainer
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

    # Extract the first word of the second part of the model ID
    model_name_part = model_id.split('/')[1].split('-')[0]
    training_args = KTOConfig(
        output_dir=f"./models/KTO/{model_name_part}_{dataset_name}",
        per_device_train_batch_size=8,
        max_steps=3,
        num_train_epochs=1,                     # number of training epochs
        lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        eval_strategy="steps",
        # beta=0.1,
        logging_steps=10,
        eval_steps=500,                        # when to evaluate
        warmup_ratio=0.1,                      # warmup ratio based on QLoRA paper
        report_to="wandb",                      # report metrics to wanb
        logging_first_step=True,
        bf16=True,                             # use bfloat16 precision
    )

    kto_trainer = KTOTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    kto_trainer.save_model(f"./models/KTO/{model_name_part}_{dataset_name}")

    # free the memory again
    del model
    del kto_trainer
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