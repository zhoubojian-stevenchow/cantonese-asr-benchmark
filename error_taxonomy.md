# Hong Kong Cantonese ASR Error Taxonomy

This document classifies common ASR errors specific to Hong Kong Cantonese speech.

## 1. Phonological Errors

### 1.1 Tone Confusion

Cantonese has 6 lexical tones. Tone confusion is the most frequent error category, especially between tones with similar pitch contours.

**High-confusion pairs:**
- Tone 1 (high-level, 55) ↔ Tone 3 (mid-level, 33): Both level tones, differ only in pitch height
- Tone 2 (high-rising, 25) ↔ Tone 5 (low-rising, 23): Both rising, differ in starting pitch
- Tone 4 (low-falling, 21) ↔ Tone 6 (low-level, 22): Both low register

### 1.2 Lazy Pronunciation (懶音)

Systematic onset mergers in casual HK Cantonese:

| Merger | IPA | Prevalence |
|--------|-----|------------|
| n → l | /n/ → /l/ | Very common |
| ng → ∅ | /ŋ/ → ∅ | Very common |
| gw → g | /kʷ/ → /k/ | Common |
| kw → k | /kʷʰ/ → /kʰ/ | Common |

### 1.3 Checked Tones

Syllables ending in /-p/, /-t/, /-k/ have shorter duration and different tonal realization.

## 2. Code-Switching Errors

### 2.1 Language Boundary Errors

The model may fail to correctly identify where Cantonese ends and English begins:

```
Reference: 我要book個table
Error:     我要「博」個「㤌啵」  (English words transcribed as Cantonese)
```

### 2.2 English Segment Errors

- **Phonetic substitution**: English words transcribed as phonetically similar Chinese characters
- **Complete miss**: English segment dropped entirely
- **Partial recognition**: Only part of the English word captured

### 2.3 Loanword Confusion

| Cantonese | Source | Challenge |
|-----------|--------|-----------|
| 巴士 (baa1 si2) | bus | May be recognized as English "bus" |
| 的士 (dik1 si2) | taxi | Written form differs from pronunciation |
| 士多啤梨 (si6 do1 be1 lei2) | strawberry | Long transliteration |

## 3. Evaluation Recommendations

1. Report CER separately for pure Cantonese and code-switched utterances
2. Include tone accuracy as a supplementary metric
3. Test on conversational data to capture lazy pronunciation effects
4. Document the test domain — read speech CER ≠ conversational CER
