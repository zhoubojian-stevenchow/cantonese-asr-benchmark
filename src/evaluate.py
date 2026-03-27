"""
Evaluation module for Cantonese ASR models.

Computes standard ASR metrics (CER) and Hong Kong-specific
metrics including Mix Error Rate for code-switched utterances
and code-switch boundary F1.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .code_switch_tagger import Language, count_code_switches, tag_code_switches
from .utils import normalize_text

logger = logging.getLogger(__name__)


def transcribe_batch(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_batch: list[dict],
    language: str = "zh",
    task: str = "transcribe",
    device: str = "cuda",
) -> list[str]:
    """Transcribe a batch of audio samples.

    Args:
        model: Whisper model (with or without LoRA).
        processor: Whisper processor.
        audio_batch: List of audio dicts with 'array' and 'sampling_rate'.
        language: Language token.
        task: Task token ('transcribe' or 'translate').
        device: Device to run on.

    Returns:
        List of transcription strings.
    """
    input_features = processor.feature_extractor(
        [a["array"] for a in audio_batch],
        sampling_rate=audio_batch[0]["sampling_rate"],
        return_tensors="pt",
        padding=True,
    ).input_features.to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task=task
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    transcriptions = processor.tokenizer.batch_decode(
        predicted_ids, skip_special_tokens=True
    )
    return transcriptions


def compute_cer(predictions: list[str], references: list[str]) -> float:
    """Compute Character Error Rate.

    Args:
        predictions: ASR output texts.
        references: Ground truth texts.

    Returns:
        CER as a float (0.0 = perfect, 1.0 = all errors).
    """
    cer_metric = evaluate.load("cer")

    # Normalize both sides
    pred_norm = [normalize_text(p) for p in predictions]
    ref_norm = [normalize_text(r) for r in references]

    # Filter out empty references
    pairs = [
        (p, r) for p, r in zip(pred_norm, ref_norm) if len(r) > 0
    ]
    if not pairs:
        return 0.0

    pred_filtered, ref_filtered = zip(*pairs)
    return cer_metric.compute(
        predictions=list(pred_filtered),
        references=list(ref_filtered),
    )


def compute_code_switch_mer(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute Mix Error Rate for code-switched utterances.

    Evaluates CER separately for:
    - Pure Cantonese segments
    - Pure English segments
    - Code-switched utterances overall

    Args:
        predictions: ASR output texts.
        references: Ground truth texts.

    Returns:
        Dict with 'cantonese_cer', 'english_wer', 'overall_mer',
        'cs_utterance_cer', 'pure_utterance_cer'.
    """
    cs_preds, cs_refs = [], []
    pure_preds, pure_refs = [], []

    for pred, ref in zip(predictions, references):
        if count_code_switches(ref) > 0:
            cs_preds.append(pred)
            cs_refs.append(ref)
        else:
            pure_preds.append(pred)
            pure_refs.append(ref)

    results = {
        "overall_cer": compute_cer(predictions, references),
        "num_code_switched": len(cs_refs),
        "num_pure": len(pure_refs),
    }

    if cs_refs:
        results["cs_utterance_cer"] = compute_cer(cs_preds, cs_refs)
    else:
        results["cs_utterance_cer"] = None

    if pure_refs:
        results["pure_utterance_cer"] = compute_cer(pure_preds, pure_refs)
    else:
        results["pure_utterance_cer"] = None

    return results


def compute_code_switch_boundary_f1(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Evaluate how well the model preserves code-switch boundaries.

    Compares the number and approximate positions of language
    switch points between reference and predicted text.

    Args:
        predictions: ASR outputs.
        references: Ground truth texts.

    Returns:
        Dict with precision, recall, F1 for switch detection.
    """
    tp, fp, fn = 0, 0, 0

    for pred, ref in zip(predictions, references):
        ref_switches = count_code_switches(ref)
        pred_switches = count_code_switches(pred)

        # Simplified boundary matching:
        # True positives = min of predicted and actual switches
        # False positives = excess predicted switches
        # False negatives = missed switches
        matched = min(ref_switches, pred_switches)
        tp += matched
        fp += max(0, pred_switches - ref_switches)
        fn += max(0, ref_switches - pred_switches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "total_ref_switches": tp + fn,
        "total_pred_switches": tp + fp,
    }


def evaluate_model(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    test_dataset,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run full evaluation on a test dataset.

    Args:
        model: Trained Whisper model.
        processor: Whisper processor.
        test_dataset: Dataset with 'audio' and 'sentence' columns.
        batch_size: Inference batch size.
        device: Device string.

    Returns:
        Dict with all metrics and per-sample predictions.
    """
    model.eval()
    model.to(device)

    all_predictions = []
    all_references = []

    logger.info("Running inference on test set...")

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i : i + batch_size]

        preds = transcribe_batch(
            model=model,
            processor=processor,
            audio_batch=batch["audio"],
            device=device,
        )

        all_predictions.extend(preds)
        all_references.extend(batch["sentence"])

    # Compute metrics
    logger.info("Computing metrics...")

    overall_cer = compute_cer(all_predictions, all_references)
    cs_metrics = compute_code_switch_mer(all_predictions, all_references)
    boundary_metrics = compute_code_switch_boundary_f1(
        all_predictions, all_references
    )

    results = {
        "cer": overall_cer,
        "code_switch": cs_metrics,
        "boundary": boundary_metrics,
        "num_samples": len(all_references),
        "predictions": all_predictions,
        "references": all_references,
    }

    logger.info(f"Overall CER: {overall_cer:.4f}")
    logger.info(f"Code-switch CER: {cs_metrics.get('cs_utterance_cer')}")
    logger.info(f"Boundary F1: {boundary_metrics['boundary_f1']:.4f}")

    return results


def save_results(
    results: dict, output_path: str = "results/evaluation.json"
) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Evaluation results dict.
        output_path: File path for output.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable values
    serializable = {}
    for k, v in results.items():
        if k in ("predictions", "references"):
            serializable[k] = v[:50]  # Save first 50 for inspection
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {path}")
