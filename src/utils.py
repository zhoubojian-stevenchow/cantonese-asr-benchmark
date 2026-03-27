"""
Utility functions for Cantonese ASR pipeline.
Includes Jyutping processing, audio preprocessing, and text normalization.
"""

import re
import unicodedata
from typing import Optional

import pycantonese


# ── Jyutping utilities ──────────────────────────────────────────────

# Cantonese tone inventory (6 citation tones)
TONE_MAP = {
    "1": "high-level",      # 陰平 55
    "2": "high-rising",     # 陰上 25
    "3": "mid-level",       # 陰去 33
    "4": "low-falling",     # 陽平 21
    "5": "low-rising",      # 陽上 23
    "6": "low-level",       # 陽去 22
}


def characters_to_jyutping(text: str) -> list[tuple[str, Optional[str]]]:
    """Convert Cantonese characters to Jyutping romanization.

    Uses PyCantonese's built-in conversion model based on
    HKCanCor and rime-cantonese data.

    Args:
        text: Chinese character string.

    Returns:
        List of (word, jyutping) tuples.
        Jyutping is None for non-Chinese tokens.
    """
    return pycantonese.characters_to_jyutping(text)


def extract_tones(jyutping: str) -> list[str]:
    """Extract tone numbers from a Jyutping string.

    Args:
        jyutping: Jyutping romanization (e.g., 'gwong2dung1waa2').

    Returns:
        List of tone numbers as strings.

    Example:
        >>> extract_tones('gwong2dung1waa2')
        ['2', '1', '2']
    """
    return re.findall(r"[1-6]", jyutping)


def parse_jyutping_syllables(jyutping: str) -> list[dict]:
    """Parse Jyutping into structured syllable components.

    Args:
        jyutping: Jyutping string (e.g., 'hoeng1gong2jan4').

    Returns:
        List of dicts with onset, nucleus, coda, tone.
    """
    try:
        parsed = pycantonese.parse_jyutping(jyutping)
        return [
            {
                "onset": syl.onset,
                "nucleus": syl.nucleus,
                "coda": syl.coda,
                "tone": syl.tone,
            }
            for syl in parsed
        ]
    except ValueError:
        return []


# ── Lazy pronunciation detection ────────────────────────────────────

# Common lazy pronunciation mergers in Hong Kong Cantonese
LAZY_MERGERS = {
    # (standard_onset, lazy_onset, description)
    ("n", "l"): "n/l merger (你→lei5)",
    ("ng", ""): "ng-dropping (我→o5)",
    ("gw", "g"): "gw/g merger (國→gok3)",
    ("kw", "k"): "kw/k merger (曠→kong3)",
}

# Known n/l confusion pairs (character: standard_jyutping, lazy_jyutping)
NL_PAIRS = {
    "你": ("nei5", "lei5"),
    "年": ("nin4", "lin4"),
    "男": ("naam4", "laam4"),
    "女": ("neoi5", "leoi5"),
    "腦": ("nou5", "lou5"),
    "暖": ("nyun5", "lyun5"),
    "難": ("naan4", "laan4"),
    "內": ("noi6", "loi6"),
}

# Known ng-dropping pairs
NG_DROP_PAIRS = {
    "我": ("ngo5", "o5"),
    "牛": ("ngau4", "au4"),
    "岸": ("ngon6", "on6"),
    "眼": ("ngaan5", "aan5"),
    "愛": ("ngoi3", "oi3"),
    "鱷": ("ngok6", "ok6"),
}


def detect_lazy_pronunciation(
    reference: str, prediction: str
) -> list[dict]:
    """Detect potential lazy pronunciation patterns in ASR output.

    Compares reference and predicted characters to identify
    systematic onset mergers typical of casual HK Cantonese.

    Args:
        reference: Ground truth text.
        prediction: ASR output text.

    Returns:
        List of detected lazy pronunciation instances.
    """
    detections = []

    ref_jyutping = characters_to_jyutping(reference)
    pred_jyutping = characters_to_jyutping(prediction)

    for (ref_word, ref_jp), (pred_word, pred_jp) in zip(
        ref_jyutping, pred_jyutping
    ):
        if ref_jp is None or pred_jp is None:
            continue
        if ref_word == pred_word:
            continue

        ref_syls = parse_jyutping_syllables(ref_jp)
        pred_syls = parse_jyutping_syllables(pred_jp)

        for ref_syl, pred_syl in zip(ref_syls, pred_syls):
            onset_pair = (ref_syl["onset"], pred_syl["onset"])
            if onset_pair in LAZY_MERGERS:
                detections.append(
                    {
                        "type": LAZY_MERGERS[onset_pair],
                        "reference": ref_word,
                        "prediction": pred_word,
                        "ref_jyutping": ref_jp,
                        "pred_jyutping": pred_jp,
                    }
                )

    return detections


# ── Text normalization ──────────────────────────────────────────────

# Punctuation to remove for CER computation
PUNCT_PATTERN = re.compile(
    r"[，。！？、；：「」『』【】（）《》〈〉"
    r"—…·～\s\.\,\!\?\;\:\"\'\-\(\)\[\]]"
)


def normalize_text(text: str, remove_punctuation: bool = True) -> str:
    """Normalize text for ASR evaluation.

    Args:
        text: Raw text string.
        remove_punctuation: Whether to strip punctuation.

    Returns:
        Normalized text.
    """
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # Convert full-width to half-width for English characters
    result = []
    for ch in text:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            result.append(chr(cp - 0xFEE0))
        else:
            result.append(ch)
    text = "".join(result)

    if remove_punctuation:
        text = PUNCT_PATTERN.sub("", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Audio preprocessing ─────────────────────────────────────────────

def validate_audio(
    audio_array,
    sampling_rate: int,
    min_duration_s: float = 1.0,
    max_duration_s: float = 30.0,
) -> bool:
    """Check whether an audio sample meets duration and quality criteria.

    Args:
        audio_array: Numpy array of audio samples.
        sampling_rate: Sample rate in Hz.
        min_duration_s: Minimum duration in seconds.
        max_duration_s: Maximum duration in seconds.

    Returns:
        True if the sample passes all checks.
    """
    duration = len(audio_array) / sampling_rate
    if duration < min_duration_s or duration > max_duration_s:
        return False

    # Check for silence (energy too low)
    import numpy as np

    rms = np.sqrt(np.mean(audio_array**2))
    if rms < 1e-5:
        return False

    return True
