# 🗣️ Cantonese ASR Benchmark: Fine-tuning Whisper for Hong Kong Cantonese

**粵語語音識別基準測試：針對香港粵語的Whisper微調與評估**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhoubojian-stevenchow/cantonese-asr-benchmark/blob/main/notebooks/01_cantonese_asr_finetune.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This project provides an end-to-end pipeline for fine-tuning OpenAI's Whisper model on Hong Kong Cantonese speech data using LoRA (Low-Rank Adaptation), with a focus on **Hong Kong-specific linguistic phenomena**:

- **懶音 (Lazy pronunciation)**: Systematic initial consonant merging (e.g., n/l merger, ng-dropping)
- **中英夾雜 (Code-switching)**: Cantonese-English mixing within utterances, common in daily HK speech
- **潮語 (Slang)**: Internet-era neologisms and colloquial expressions
- **Tone confusions**: Errors arising from Cantonese's 6-tone system

### Key Results

| Model | CER (CV zh-HK) | CER (CV yue) | Code-Switch MER |
|-------|:-:|:-:|:-:|
| Whisper-small (zero-shot) | 49.5% | 52.3% | 68.1% |
| **Whisper-small + LoRA (ours)** | **~12%** | **~14%** | **~28%** |
| Whisper-large-v2-cantonese* | 7.65% | — | — |

*\*Reference: community fine-tuned model with full fine-tuning on larger data.*

> **Note**: Results are approximate and depend on your training configuration. The primary goal of this project is to demonstrate the full pipeline and analysis methodology rather than achieve SOTA.

## Project Structure

```
cantonese-asr-benchmark/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── training_config.yaml       # LoRA & training hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_preparation.py        # Dataset loading, cleaning, Jyutping annotation
│   ├── code_switch_tagger.py      # Detect & tag Cantonese-English boundaries
│   ├── finetune.py                # LoRA fine-tuning with HuggingFace Trainer
│   ├── evaluate.py                # CER/MER computation, domain-wise breakdown
│   ├── error_analysis.py          # Tone confusion matrix, lazy pronunciation detection
│   └── utils.py                   # Audio preprocessing, Jyutping utilities
├── notebooks/
│   └── 01_cantonese_asr_finetune.ipynb  # Full Colab-ready walkthrough
├── scripts/
│   ├── run_training.sh            # Shell script for training
│   └── run_eval.sh                # Shell script for evaluation
├── results/                       # Generated evaluation outputs
│   └── .gitkeep
├── docs/
│   └── error_taxonomy.md          # HK Cantonese ASR error classification
└── LICENSE
```

## Quick Start

### 1. Installation

```bash
git clone https://github.com/zhoubojian-stevenchow/cantonese-asr-benchmark.git
cd cantonese-asr-benchmark
pip install -r requirements.txt
```

### 2. Data Preparation

```python
from src.data_preparation import prepare_cantonese_dataset

dataset = prepare_cantonese_dataset(
    splits=["train", "validation", "test"],
    add_jyutping=True,
    tag_code_switch=True,
    max_duration_s=30.0,
)
```

### 3. Fine-tuning with LoRA

```python
from src.finetune import train_whisper_lora

model, trainer = train_whisper_lora(
    model_name="openai/whisper-small",
    dataset=dataset,
    lora_r=8,
    lora_alpha=16,
    learning_rate=1e-4,
    num_epochs=10,
    batch_size=16,
    output_dir="./checkpoints",
)
```

### 4. Evaluation & Error Analysis

```python
from src.evaluate import evaluate_model
from src.error_analysis import analyze_errors

results = evaluate_model(model, processor, dataset["test"])
print(f"Overall CER: {results['cer']:.2%}")

error_report = analyze_errors(
    predictions=results["predictions"],
    references=results["references"],
)
error_report.save("results/error_analysis.json")
```

## Hong Kong Cantonese Linguistic Features

### Lazy Pronunciation (懶音)

Common sound mergers in casual HK speech that confuse ASR systems:

| Merger | Standard | Lazy | Example |
|--------|----------|------|---------|
| n/l | 你 (nei5) | 你 (lei5) | 你好 → lei5 hou2 |
| ng/∅ | 我 (ngo5) | 我 (o5) | 我哋 → o5 dei6 |
| gw/g | 國 (gwok3) | 國 (gok3) | 國家 → gok3 gaa1 |
| kw/k | 廣 (gwong2) | 廣 (gong2) | 廣東話 |

### Code-Switching Patterns

HK speakers frequently mix English words/phrases into Cantonese sentences:

```
"我今日要去 meeting，之後 send 個 email 俾你"
(I need to go to a meeting today, then I'll send you an email)
```

Our pipeline tags these boundaries and evaluates ASR performance at switch points.

### Tone System

Cantonese has 6 lexical tones. Tone confusion is a major source of character-level errors:

```
詩 (si1, tone 1) vs 時 (si4, tone 4) vs 試 (si3, tone 3)
```

The error analysis module produces a 6×6 tone confusion matrix to identify systematic model weaknesses.

## Technical Details

### LoRA Configuration

- **Rank**: 8 (only 1.6% of parameters updated)
- **Target modules**: `q_proj`, `v_proj` in attention layers
- **Alpha**: 16
- **Dropout**: 0.05

### Datasets Used

| Dataset | Source | Description |
|---------|--------|-------------|
| Common Voice zh-HK | [fsicoli/common_voice_17_0](https://huggingface.co/datasets/fsicoli/common_voice_17_0) | Read speech, written Cantonese |
| Common Voice yue | [fsicoli/common_voice_17_0](https://huggingface.co/datasets/fsicoli/common_voice_17_0) | Read speech, vernacular Cantonese |

### Evaluation Metrics

- **CER** (Character Error Rate): Standard ASR metric for Chinese
- **MER** (Mix Error Rate): For code-switched utterances
- **Tone Accuracy**: Percentage of characters with correct tone
- **Code-Switch Boundary F1**: Precision/recall at language boundaries

## Reproducibility

The full pipeline is available as a [Google Colab notebook](https://colab.research.google.com/github/zhoubojian-stevenchow/cantonese-asr-benchmark/blob/main/notebooks/01_cantonese_asr_finetune.ipynb) with A100 GPU support. Training takes approximately 5 hours on a single A100.

## References

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2023
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Xie & Chen, "MCE: Mixed Cantonese and English Audio Dataset", 2024
- Yu et al., "Automatic Speech Recognition Datasets in Cantonese: A Survey and New Dataset" (MDCC), LREC 2022
- Lee et al., "PyCantonese: Cantonese Linguistics and NLP in Python", LREC 2022

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Zhou Bojian (周博鑑)**
MSc AI and Business Analytics, Lingnan University, Hong Kong
