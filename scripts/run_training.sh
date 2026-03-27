#!/bin/bash
# Run Cantonese ASR fine-tuning
# Usage: bash scripts/run_training.sh

set -e

echo "=== Cantonese ASR LoRA Fine-tuning ==="
echo "Model: openai/whisper-small"
echo "Data: Common Voice zh-HK + yue"
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO)

from src.data_preparation import prepare_cantonese_dataset
from src.finetune import train_whisper_lora

# Prepare dataset
print('Step 1: Preparing dataset...')
dataset = prepare_cantonese_dataset(
    cv_version='17.0',
    splits=['train', 'validation'],
    language_codes=['zh-HK', 'yue'],
    add_jyutping=True,
    tag_code_switch=True,
    max_duration_s=30.0,
    num_proc=4,
)

# Train
print('Step 2: Starting LoRA fine-tuning...')
model, trainer = train_whisper_lora(
    model_name='openai/whisper-small',
    dataset=dataset,
    lora_r=8,
    lora_alpha=16,
    learning_rate=1e-4,
    num_epochs=10,
    batch_size=16,
    output_dir='./checkpoints/whisper-small-cantonese-lora',
)

print('Training complete!')
"

echo "=== Done ==="
