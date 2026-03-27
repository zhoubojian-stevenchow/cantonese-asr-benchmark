#!/bin/bash
# Run evaluation and error analysis
# Usage: bash scripts/run_eval.sh [checkpoint_path]

set -e

CHECKPOINT=${1:-"./checkpoints/whisper-small-cantonese-lora"}

echo "=== Cantonese ASR Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo ""

python -c "
import logging
logging.basicConfig(level=logging.INFO)

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data_preparation import load_common_voice
from src.evaluate import evaluate_model, save_results
from src.error_analysis import analyze_errors

checkpoint = '${CHECKPOINT}'

# Load model
print('Loading model...')
processor = WhisperProcessor.from_pretrained(checkpoint)
base_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')
model = PeftModel.from_pretrained(base_model, checkpoint)
model = model.merge_and_unload()

# Load test data
print('Loading test data...')
from datasets import Audio
test_ds = load_common_voice('zh-HK', '17.0', 'test')
test_ds = test_ds.cast_column('audio', Audio(sampling_rate=16000))

# Evaluate
print('Running evaluation...')
results = evaluate_model(model, processor, test_ds, batch_size=8)

# Save results
save_results(results, 'results/evaluation.json')

# Error analysis
print('Running error analysis...')
analyzer = analyze_errors(
    predictions=results['predictions'],
    references=results['references'],
)
analyzer.save('results/error_analysis.json')

print(f'Overall CER: {results[\"cer\"]:.4f}')
print('Done! Results saved to results/')
"

echo "=== Done ==="
