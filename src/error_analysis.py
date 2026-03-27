"""
Hong Kong Cantonese-specific ASR error analysis.

Analyzes common error patterns unique to HK Cantonese:
- Tone confusion matrix (6-tone system)
- Lazy pronunciation detection (onset mergers)
- Code-switch boundary errors
- Character-level error taxonomy
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .code_switch_tagger import Language, tag_code_switches
from .utils import (
    characters_to_jyutping,
    detect_lazy_pronunciation,
    extract_tones,
    normalize_text,
    parse_jyutping_syllables,
    TONE_MAP,
)

logger = logging.getLogger(__name__)


class CantoneseErrorAnalyzer:
    """Comprehensive error analyzer for HK Cantonese ASR output.

    Examines prediction errors through the lens of Cantonese
    phonology and Hong Kong speech patterns.
    """

    def __init__(self):
        self.tone_confusions: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.lazy_pronunciation_counts: Counter = Counter()
        self.code_switch_errors: list[dict] = []
        self.character_substitutions: Counter = Counter()
        self.total_chars: int = 0
        self.total_errors: int = 0

    def analyze_pair(self, prediction: str, reference: str) -> dict:
        """Analyze a single prediction-reference pair.

        Args:
            prediction: ASR output.
            reference: Ground truth.

        Returns:
            Per-sample error analysis dict.
        """
        sample_errors = {
            "prediction": prediction,
            "reference": reference,
            "tone_errors": [],
            "lazy_errors": [],
            "cs_errors": [],
        }

        # Normalize
        pred_norm = normalize_text(prediction, remove_punctuation=False)
        ref_norm = normalize_text(reference, remove_punctuation=False)

        # Tone analysis
        tone_errors = self._analyze_tones(pred_norm, ref_norm)
        sample_errors["tone_errors"] = tone_errors

        # Lazy pronunciation
        lazy_errors = detect_lazy_pronunciation(ref_norm, pred_norm)
        sample_errors["lazy_errors"] = lazy_errors
        for err in lazy_errors:
            self.lazy_pronunciation_counts[err["type"]] += 1

        # Code-switch boundary errors
        cs_errors = self._analyze_code_switch_errors(pred_norm, ref_norm)
        sample_errors["cs_errors"] = cs_errors

        # Character substitutions
        self._count_substitutions(pred_norm, ref_norm)

        return sample_errors

    def _analyze_tones(
        self, prediction: str, reference: str
    ) -> list[dict]:
        """Compare tones between prediction and reference.

        Builds a confusion matrix of predicted vs actual tones
        by aligning Jyutping annotations character by character.

        Args:
            prediction: Predicted text.
            reference: Reference text.

        Returns:
            List of tone error instances.
        """
        errors = []

        try:
            pred_jp = characters_to_jyutping(prediction)
            ref_jp = characters_to_jyutping(reference)
        except Exception:
            return errors

        # Align by iterating through paired Jyutping annotations
        for (ref_word, ref_jyutping), (pred_word, pred_jyutping) in zip(
            ref_jp, pred_jp
        ):
            if ref_jyutping is None or pred_jyutping is None:
                continue

            ref_tones = extract_tones(ref_jyutping)
            pred_tones = extract_tones(pred_jyutping)

            for rt, pt in zip(ref_tones, pred_tones):
                self.tone_confusions[rt][pt] += 1
                if rt != pt:
                    errors.append(
                        {
                            "ref_char": ref_word,
                            "pred_char": pred_word,
                            "ref_tone": rt,
                            "pred_tone": pt,
                            "ref_tone_name": TONE_MAP.get(rt, "unknown"),
                            "pred_tone_name": TONE_MAP.get(pt, "unknown"),
                        }
                    )

        return errors

    def _analyze_code_switch_errors(
        self, prediction: str, reference: str
    ) -> list[dict]:
        """Analyze errors at code-switch boundaries.

        Identifies cases where the model fails to correctly
        handle Cantonese↔English transitions.

        Args:
            prediction: Predicted text.
            reference: Reference text.

        Returns:
            List of code-switch error instances.
        """
        errors = []

        ref_spans = tag_code_switches(reference)
        pred_spans = tag_code_switches(prediction)

        # Extract English segments from both
        ref_en = [
            s.text.lower() for s in ref_spans if s.language == Language.ENGLISH
        ]
        pred_en = [
            s.text.lower()
            for s in pred_spans
            if s.language == Language.ENGLISH
        ]

        # Find missed English segments
        for en_word in ref_en:
            if en_word not in pred_en:
                errors.append(
                    {
                        "type": "missed_english",
                        "reference": en_word,
                        "detail": "English segment in reference not found in prediction",
                    }
                )

        # Find hallucinated English segments
        for en_word in pred_en:
            if en_word not in ref_en:
                errors.append(
                    {
                        "type": "hallucinated_english",
                        "prediction": en_word,
                        "detail": "English segment in prediction not in reference",
                    }
                )

        self.code_switch_errors.extend(errors)
        return errors

    def _count_substitutions(
        self, prediction: str, reference: str
    ) -> None:
        """Count character-level substitutions using simple alignment.

        Args:
            prediction: Predicted text.
            reference: Reference text.
        """
        # Simple character-level comparison (not full edit distance)
        min_len = min(len(prediction), len(reference))
        for i in range(min_len):
            self.total_chars += 1
            if prediction[i] != reference[i]:
                self.total_errors += 1
                pair = f"{reference[i]}→{prediction[i]}"
                self.character_substitutions[pair] += 1

    def get_tone_confusion_matrix(self) -> dict:
        """Build the 6x6 tone confusion matrix.

        Returns:
            Dict with 'matrix' (6x6 numpy array),
            'labels' (tone numbers), and 'accuracy' per tone.
        """
        tones = ["1", "2", "3", "4", "5", "6"]
        matrix = np.zeros((6, 6), dtype=int)

        for i, ref_tone in enumerate(tones):
            for j, pred_tone in enumerate(tones):
                matrix[i][j] = self.tone_confusions[ref_tone][pred_tone]

        # Per-tone accuracy
        accuracy = {}
        for i, tone in enumerate(tones):
            row_sum = matrix[i].sum()
            if row_sum > 0:
                accuracy[tone] = float(matrix[i][i] / row_sum)
            else:
                accuracy[tone] = None

        return {
            "matrix": matrix.tolist(),
            "labels": tones,
            "tone_names": [TONE_MAP[t] for t in tones],
            "per_tone_accuracy": accuracy,
            "overall_tone_accuracy": float(
                np.trace(matrix) / matrix.sum()
            )
            if matrix.sum() > 0
            else None,
        }

    def get_lazy_pronunciation_summary(self) -> dict:
        """Summarize lazy pronunciation detections.

        Returns:
            Dict with counts per merger type and total.
        """
        return {
            "counts": dict(self.lazy_pronunciation_counts),
            "total": sum(self.lazy_pronunciation_counts.values()),
        }

    def get_top_substitutions(self, n: int = 20) -> list[tuple[str, int]]:
        """Get the most common character substitution errors.

        Args:
            n: Number of top errors to return.

        Returns:
            List of (substitution_pair, count) tuples.
        """
        return self.character_substitutions.most_common(n)

    def generate_report(self) -> dict:
        """Generate a complete error analysis report.

        Returns:
            Comprehensive error analysis dict.
        """
        return {
            "tone_analysis": self.get_tone_confusion_matrix(),
            "lazy_pronunciation": self.get_lazy_pronunciation_summary(),
            "code_switch_errors": {
                "total": len(self.code_switch_errors),
                "missed_english": sum(
                    1
                    for e in self.code_switch_errors
                    if e["type"] == "missed_english"
                ),
                "hallucinated_english": sum(
                    1
                    for e in self.code_switch_errors
                    if e["type"] == "hallucinated_english"
                ),
            },
            "top_substitutions": self.get_top_substitutions(),
            "summary": {
                "total_chars_analyzed": self.total_chars,
                "total_char_errors": self.total_errors,
                "char_error_rate": self.total_errors / self.total_chars
                if self.total_chars > 0
                else 0.0,
            },
        }

    def save(self, path: str = "results/error_analysis.json") -> None:
        """Save the report to JSON.

        Args:
            path: Output file path.
        """
        report = self.generate_report()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Error analysis saved to {output_path}")


def analyze_errors(
    predictions: list[str],
    references: list[str],
    analyze_tones: bool = True,
    analyze_lazy_pronunciation: bool = True,
    analyze_code_switch: bool = True,
) -> CantoneseErrorAnalyzer:
    """Run full error analysis on prediction-reference pairs.

    Args:
        predictions: ASR output texts.
        references: Ground truth texts.
        analyze_tones: Whether to analyze tone confusions.
        analyze_lazy_pronunciation: Whether to detect lazy pronunciation.
        analyze_code_switch: Whether to analyze code-switch errors.

    Returns:
        CantoneseErrorAnalyzer with all results populated.
    """
    analyzer = CantoneseErrorAnalyzer()

    logger.info(f"Analyzing {len(predictions)} prediction pairs...")

    for pred, ref in zip(predictions, references):
        analyzer.analyze_pair(pred, ref)

    report = analyzer.generate_report()

    # Log summary
    tone_info = report["tone_analysis"]
    if tone_info["overall_tone_accuracy"] is not None:
        logger.info(
            f"Overall tone accuracy: {tone_info['overall_tone_accuracy']:.2%}"
        )

    lazy_info = report["lazy_pronunciation"]
    logger.info(f"Lazy pronunciation detections: {lazy_info['total']}")

    cs_info = report["code_switch_errors"]
    logger.info(
        f"Code-switch errors: {cs_info['total']} "
        f"(missed: {cs_info['missed_english']}, "
        f"hallucinated: {cs_info['hallucinated_english']})"
    )

    return analyzer
