"""
Code-switch tagger for Cantonese-English mixed text.

Identifies language boundaries in Hong Kong Cantonese text where
speakers mix English words/phrases into Cantonese utterances.
This is essential for evaluating ASR performance on code-switched speech,
a hallmark of everyday Hong Kong communication.

Example:
    "我今日要去meeting，之後send個email俾你"
    → [ZH:我今日要去] [EN:meeting] [ZH:，之後] [EN:send] [ZH:個] [EN:email] [ZH:俾你]
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Language(Enum):
    """Language tag for code-switch segments."""
    CANTONESE = "ZH"
    ENGLISH = "EN"
    MIXED = "MIX"
    PUNCTUATION = "PUNCT"


@dataclass
class CodeSwitchSpan:
    """A contiguous span of text in one language."""
    text: str
    language: Language
    start: int
    end: int

    def __repr__(self) -> str:
        return f"[{self.language.value}:{self.text}]"


# ── Character classification ────────────────────────────────────────

# CJK Unified Ideographs ranges
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Supplement
]

# Common Cantonese-specific characters (口字旁 characters)
CANTONESE_CHARS = set("嘅啲咁咗嘢喺冇啱佢哋噉嗰乜點嚟")

# Common HK English loanwords that are phonetically borrowed
# These are often written in Chinese characters, not English
LOANWORD_CHARS = {
    "巴士": "bus", "的士": "taxi", "士多": "store",
    "芝士": "cheese", "朱古力": "chocolate",
    "三文治": "sandwich", "多士": "toast",
}


def is_cjk(char: str) -> bool:
    """Check if a character is a CJK ideograph."""
    cp = ord(char)
    return any(start <= cp <= end for start, end in CJK_RANGES)


def is_english(char: str) -> bool:
    """Check if a character is an English letter or digit."""
    return char.isascii() and (char.isalpha() or char.isdigit())


def is_punctuation(char: str) -> bool:
    """Check if a character is punctuation."""
    return not is_cjk(char) and not is_english(char) and not char.isspace()


# ── Tagger ──────────────────────────────────────────────────────────

def tag_code_switches(text: str) -> list[CodeSwitchSpan]:
    """Segment text into Cantonese and English spans.

    Identifies language boundaries in mixed Cantonese-English text.
    Consecutive characters of the same language type are grouped
    into spans.

    Args:
        text: Input text potentially containing code-switching.

    Returns:
        List of CodeSwitchSpan objects marking language boundaries.

    Example:
        >>> spans = tag_code_switches("我要去meeting")
        >>> print(spans)
        [[ZH:我要去], [EN:meeting]]
    """
    if not text:
        return []

    spans: list[CodeSwitchSpan] = []
    current_text = ""
    current_lang: Optional[Language] = None
    current_start = 0

    for i, char in enumerate(text):
        if char.isspace():
            current_text += char
            continue

        if is_cjk(char):
            char_lang = Language.CANTONESE
        elif is_english(char):
            char_lang = Language.ENGLISH
        elif is_punctuation(char):
            # Punctuation inherits the language of its context
            current_text += char
            continue
        else:
            current_text += char
            continue

        if current_lang is None:
            current_lang = char_lang
            current_start = i
            current_text = char
        elif char_lang == current_lang:
            current_text += char
        else:
            # Language boundary detected
            if current_text.strip():
                spans.append(
                    CodeSwitchSpan(
                        text=current_text.strip(),
                        language=current_lang,
                        start=current_start,
                        end=i,
                    )
                )
            current_lang = char_lang
            current_start = i
            current_text = char

    # Flush remaining
    if current_text.strip() and current_lang is not None:
        spans.append(
            CodeSwitchSpan(
                text=current_text.strip(),
                language=current_lang,
                start=current_start,
                end=len(text),
            )
        )

    return spans


def count_code_switches(text: str) -> int:
    """Count the number of language switch points in text.

    Args:
        text: Input text.

    Returns:
        Number of Cantonese↔English transition points.
    """
    spans = tag_code_switches(text)
    if len(spans) <= 1:
        return 0

    switches = 0
    for i in range(1, len(spans)):
        if spans[i].language != spans[i - 1].language:
            switches += 1

    return switches


def get_code_switch_ratio(text: str) -> float:
    """Calculate the proportion of English characters in the text.

    Args:
        text: Input text.

    Returns:
        Ratio of English characters to total non-space characters.
    """
    spans = tag_code_switches(text)
    if not spans:
        return 0.0

    en_chars = sum(
        len(s.text) for s in spans if s.language == Language.ENGLISH
    )
    total_chars = sum(len(s.text) for s in spans)

    return en_chars / total_chars if total_chars > 0 else 0.0


def extract_english_segments(text: str) -> list[str]:
    """Extract all English segments from code-switched text.

    Args:
        text: Input text.

    Returns:
        List of English word/phrase segments.
    """
    spans = tag_code_switches(text)
    return [s.text for s in spans if s.language == Language.ENGLISH]


# ── Annotation for dataset ──────────────────────────────────────────

def annotate_dataset_entry(sentence: str) -> dict:
    """Annotate a single dataset entry with code-switch metadata.

    Args:
        sentence: Transcription text.

    Returns:
        Dictionary with code-switch annotations.
    """
    spans = tag_code_switches(sentence)

    return {
        "sentence": sentence,
        "spans": [
            {
                "text": s.text,
                "language": s.language.value,
                "start": s.start,
                "end": s.end,
            }
            for s in spans
        ],
        "num_switches": count_code_switches(sentence),
        "en_ratio": get_code_switch_ratio(sentence),
        "is_code_switched": count_code_switches(sentence) > 0,
        "english_segments": extract_english_segments(sentence),
    }
