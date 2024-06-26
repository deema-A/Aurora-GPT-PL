from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer, setup_chat_format
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch
from utils import *
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
      quantization_config=bnb_config
  )
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = 'left' # to prevent errors with FA
  tokenizer.truncation_side = 'left' # to prevent cutting off last generation

  model, tokenizer = setup_chat_format(model, tokenizer)

  dataset_functions = {
      "ultrafeedback": get_ultrafeedback_dpo
  }    
  dataset = dataset_functions[dataset_name](tokenizer) # depends on the user passed dataset

  # LoRA config based on QLoRA paper & Sebastian Raschka experiment
  peft_config = LoraConfig(
          lora_alpha=128,
          lora_dropout=0.05,
          r=256,
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
      output_dir=f"./models/DPO/{model_name_part}_{dataset_name}",   # directory to save and repository id
      num_train_epochs=1,                     # number of training epochs
      per_device_train_batch_size=12,         # batch size per device during training
      per_device_eval_batch_size=4,           # batch size for evaluation
      gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
      gradient_checkpointing=True,            # use gradient checkpointing to save memory
      optim="adamw_torch_fused",              # use fused adamw optimizer
      learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
      max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
      warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
      lr_scheduler_type="cosine",             # use cosine learning rate scheduler
      logging_steps=25,                       # log every 25 steps
      save_steps=500,                         # when to save checkpoint
      save_total_limit=2,                     # limit the total amount of checkpoints
      evaluation_strategy="steps",            # evaluate every 1000 steps
      eval_steps=700,                         # when to evaluate
      bf16=True,                              # use bfloat16 precision
      tf32=True,                              # use tf32 precision
      push_to_hub=False,                      # push model to hub
      report_to="wanb",                       # report metrics to wanb
  )

  dpo_args = {
      "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
      "loss_type": "sigmoid"                  # The loss type for DPO.
  }

  model_kwargs = dict(
      use_flash_attention_2=True,
      torch_dtype=torch.bfloat16,
    )

  trainer = DPOTrainer(
      model,
      ref_model=None, # set to none since we use peft
      peft_config=peft_config,
      model_init_kwargs = None,
      args=args,
      train_dataset=dataset,
      tokenizer=tokenizer,
      beta=dpo_args["beta"],
      loss_type=dpo_args["loss_type"],
  )

  # start training, the model will be automatically saved to the hub and the output directory
  trainer.train()

  # save model at the end of training
  trainer.save_model(f"./models/DPO/{model_name_part}_{dataset_name}")

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
