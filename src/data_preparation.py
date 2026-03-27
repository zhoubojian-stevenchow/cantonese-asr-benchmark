"""
Data preparation pipeline for Cantonese ASR fine-tuning.

Loads Common Voice zh-HK and yue splits, preprocesses audio,
normalizes text, and annotates with Jyutping romanization
and code-switch tags.
"""

import logging
from typing import Optional

from datasets import (
    Audio,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from transformers import WhisperProcessor

from .code_switch_tagger import annotate_dataset_entry
from .utils import characters_to_jyutping, normalize_text, validate_audio

logger = logging.getLogger(__name__)


def load_common_voice(
    language_code: str,
    cv_version: str = "17.0",
    split: str = "train",
    streaming: bool = False,
) -> "Dataset":
    """Load a split of Common Voice for a given language code.

    Args:
        language_code: Language code (e.g., 'zh-HK' or 'yue').
        cv_version: Common Voice version.
        split: Dataset split ('train', 'validation', 'test').
        streaming: Whether to use streaming mode.

    Returns:
        HuggingFace Dataset object.
    """
    dataset_name = f"mozilla-foundation/common_voice_{cv_version.replace('.', '_')}"
    logger.info(f"Loading {dataset_name} ({language_code}, {split})...")

    ds = load_dataset(
        dataset_name,
        language_code,
        split=split,
        streaming=streaming,
        trust_remote_code=True,
    )
    return ds


def preprocess_text(
    example: dict,
    remove_punctuation: bool = True,
    add_jyutping: bool = True,
    tag_code_switch: bool = True,
) -> dict:
    """Preprocess a single dataset example's text fields.

    Args:
        example: Dataset row with 'sentence' field.
        remove_punctuation: Strip punctuation.
        add_jyutping: Add Jyutping annotation.
        tag_code_switch: Add code-switch tags.

    Returns:
        Example with added annotation fields.
    """
    sentence = example["sentence"]

    # Normalize text
    example["sentence_normalized"] = normalize_text(
        sentence, remove_punctuation=remove_punctuation
    )

    # Add Jyutping romanization
    if add_jyutping:
        try:
            jp_result = characters_to_jyutping(sentence)
            example["jyutping"] = " ".join(
                jp if jp else word for word, jp in jp_result
            )
        except Exception:
            example["jyutping"] = ""

    # Tag code-switching
    if tag_code_switch:
        cs_info = annotate_dataset_entry(sentence)
        example["num_code_switches"] = cs_info["num_switches"]
        example["en_ratio"] = cs_info["en_ratio"]
        example["is_code_switched"] = cs_info["is_code_switched"]

    return example


def filter_by_duration(
    example: dict,
    min_duration_s: float = 1.0,
    max_duration_s: float = 30.0,
) -> bool:
    """Filter function for audio duration.

    Args:
        example: Dataset row with 'audio' field.
        min_duration_s: Minimum duration.
        max_duration_s: Maximum duration.

    Returns:
        True if sample should be kept.
    """
    audio = example["audio"]
    duration = len(audio["array"]) / audio["sampling_rate"]
    return min_duration_s <= duration <= max_duration_s


def prepare_for_whisper(
    example: dict,
    processor: WhisperProcessor,
) -> dict:
    """Convert audio and text into Whisper-compatible format.

    Args:
        example: Dataset row with 'audio' and 'sentence' fields.
        processor: Whisper processor for feature extraction.

    Returns:
        Example with 'input_features' and 'labels' fields.
    """
    audio = example["audio"]

    # Extract log-mel spectrogram features
    input_features = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    # Tokenize the target text
    labels = processor.tokenizer(
        example["sentence_normalized"],
        return_tensors="np",
    ).input_ids[0]

    example["input_features"] = input_features
    example["labels"] = labels

    return example


def prepare_cantonese_dataset(
    cv_version: str = "17.0",
    splits: list[str] = ["train", "validation", "test"],
    language_codes: list[str] = ["zh-HK", "yue"],
    add_jyutping: bool = True,
    tag_code_switch: bool = True,
    max_duration_s: float = 30.0,
    min_duration_s: float = 1.0,
    model_name: str = "openai/whisper-small",
    num_proc: int = 4,
    streaming: bool = False,
) -> DatasetDict:
    """Prepare the full Cantonese ASR dataset.

    Loads, cleans, annotates, and formats Common Voice data
    for Whisper fine-tuning.

    Args:
        cv_version: Common Voice version.
        splits: Which splits to load.
        language_codes: Language codes to combine.
        add_jyutping: Whether to add Jyutping annotations.
        tag_code_switch: Whether to tag code-switching.
        max_duration_s: Maximum audio duration.
        min_duration_s: Minimum audio duration.
        model_name: Whisper model name for processor.
        num_proc: Number of processes for mapping.
        streaming: Whether to use streaming mode.

    Returns:
        DatasetDict with processed train/validation/test splits.
    """
    processor = WhisperProcessor.from_pretrained(model_name)

    result = {}

    for split in splits:
        logger.info(f"Preparing {split} split...")

        # Load and concatenate language variants
        datasets_to_merge = []
        for lang_code in language_codes:
            try:
                ds = load_common_voice(
                    lang_code, cv_version, split, streaming
                )
                # Resample audio to 16kHz
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                datasets_to_merge.append(ds)
                logger.info(
                    f"  Loaded {lang_code}: {len(ds)} examples"
                )
            except Exception as e:
                logger.warning(
                    f"  Could not load {lang_code}/{split}: {e}"
                )

        if not datasets_to_merge:
            logger.warning(f"  No data found for {split}, skipping.")
            continue

        # Concatenate
        combined = concatenate_datasets(datasets_to_merge)
        logger.info(f"  Combined: {len(combined)} examples")

        # Filter by duration
        combined = combined.filter(
            lambda ex: filter_by_duration(
                ex, min_duration_s, max_duration_s
            ),
            num_proc=num_proc,
        )
        logger.info(f"  After duration filter: {len(combined)} examples")

        # Preprocess text (normalize, Jyutping, code-switch)
        combined = combined.map(
            lambda ex: preprocess_text(
                ex,
                add_jyutping=add_jyutping,
                tag_code_switch=tag_code_switch,
            ),
            num_proc=num_proc,
        )

        # Prepare Whisper features
        combined = combined.map(
            lambda ex: prepare_for_whisper(ex, processor),
            num_proc=num_proc,
            remove_columns=combined.column_names,
        )

        result[split] = combined

    return DatasetDict(result)
