# Model Alignment Pipeline [WIP]

This repository provides a pipeline for applying alignment techniques such as Supervied Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Kahneman-Tversky Optimization (KTO) on Hugging Face-supported models (for now). It is currently supporting ultrafeedback dataset and operates with QLORA 4-bit quantization on a single GPU.

## Features

- **Model Compatibility**: Compatible with models supported by Hugging Face.
- **Dataset Support**: Currently supporting the ultrafeedback dataset.
- **Quantization**: Utilizes QLORA 4-bit quantization for GPU efficiency.

## Tested Models

The pipeline has been tested with the following models:
- Llama2_7B

## Getting Started

### Clone the Repository

Start by cloning this repository to your local machine:

```
git clone https://github.com/deema-A/Aurora-GPT-PL.git
cd Aurora-GPT-PL
```

To run the alignment techniques sequentially:

```
python pipeline.py
```

The aligned model should be found under ./models/alignment_used/model_dataset

To run a specific technique:

```
python pipeline.py --strategies kto model_name "meta-llama/Llama-2-7b-hf" --dataset_name ultrafeedback
```

