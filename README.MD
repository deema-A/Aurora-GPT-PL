# Model Alignment Pipeline

This repository provides a pipeline for applying alignment techniques such as Supervied Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Kahneman-Tversky Optimization (KTO) on Hugging Face-supported models (for now). It is currently supporting UltraFeed dataset and operates with QLORA 4-bit quantization on a single GPU.

## Features

- **Model Compatibility**: Compatible with models supported by Hugging Face.
- **Dataset Support**: Currently supporting the UltraFeed dataset.
- **Quantization**: Utilizes QLORA 4-bit quantization for GPU efficiency.

## Tested Models

The pipeline has been tested with the following models:
- Mistral 7B version 0.3
- LLaMA 2 7B

## Getting Started

### Clone the Repository

Start by cloning this repository to your local machine:

```
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
