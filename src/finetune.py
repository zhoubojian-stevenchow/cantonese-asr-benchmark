"""
LoRA fine-tuning pipeline for Whisper on Cantonese ASR.

Uses PEFT (Parameter-Efficient Fine-Tuning) to train only ~1.6%
of Whisper's parameters, making fine-tuning feasible on a single GPU.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

import evaluate
import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2Seq:
    """Custom data collator for Whisper fine-tuning.

    Handles padding of input features (mel spectrograms) and
    label sequences (token IDs) to create batches.
    """

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Separate input features and labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels (use -100 for padding to ignore in loss)
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present (Whisper adds it during generation)
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def setup_lora_model(
    model_name: str = "openai/whisper-small",
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    """Load Whisper and apply LoRA adapters.

    Args:
        model_name: HuggingFace model identifier.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout in LoRA layers.
        target_modules: Which modules to apply LoRA to.

    Returns:
        Tuple of (PEFT model, processor).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    logger.info(f"Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_cache=False,  # Required for gradient checkpointing
    )

    # Force Cantonese/Chinese language and transcribe task
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="zh", task="transcribe"
    )
    model.config.suppress_tokens = []

    # Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)

    # Log parameter counts
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, processor


def train_whisper_lora(
    model_name: str = "openai/whisper-small",
    dataset: Optional[Any] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 2,
    output_dir: str = "./checkpoints/whisper-cantonese-lora",
    fp16: bool = True,
    eval_steps: int = 500,
    save_steps: int = 500,
    warmup_ratio: float = 0.1,
) -> tuple:
    """Run the full LoRA fine-tuning pipeline.

    Args:
        model_name: Base Whisper model.
        dataset: DatasetDict with train/validation splits.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        learning_rate: Peak learning rate.
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation.
        output_dir: Checkpoint save directory.
        fp16: Whether to use mixed precision.
        eval_steps: Evaluation frequency.
        save_steps: Checkpoint save frequency.
        warmup_ratio: Warmup proportion.

    Returns:
        Tuple of (trained model, trainer).
    """
    # Setup model
    model, processor = setup_lora_model(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    # Data collator
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # CER metric
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = processor.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True
        )

        cer = cer_metric.compute(
            predictions=pred_str, references=label_str
        )
        return {"cer": cer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size // 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if dataset else None,
        eval_dataset=dataset.get("validation") if dataset else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    # Train
    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    # Save the final LoRA adapter
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    return model, trainer
