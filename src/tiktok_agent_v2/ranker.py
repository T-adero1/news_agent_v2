import math
import re
from collections import Counter

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "this", "that", "to",
    "of", "for", "in", "on", "with", "as", "is", "are", "was", "were", "be",
    "it", "its", "we", "you", "they", "i", "he", "she", "them", "our", "your",
    "at", "by", "from", "not", "so", "do", "does", "did", "have", "has", "had"
}


def _tokenize(text: str):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def score_transcript_segments(segments):
    """
    Scores segments based on rarity of words + length.
    Returns list of dicts with added 'text_score'.
    """
    all_words = []
    for seg in segments:
        all_words.extend(_tokenize(seg["text"]))

    if not all_words:
        for seg in segments:
            seg["text_score"] = 0.0
        return segments

    freq = Counter(all_words)

    for seg in segments:
        words = _tokenize(seg["text"])
        if not words:
            seg["text_score"] = 0.0
            continue
        rarity_sum = sum(1.0 / freq[w] for w in words)
        length_boost = min(1.5, 0.5 + 0.02 * len(words))
        seg["text_score"] = rarity_sum * length_boost

    return segments


def apply_time_decay(score: float, time_sec: float, half_life: float):
    if half_life <= 0:
        return score
    decay = 0.5 ** (time_sec / half_life)
    return score * decay


def normalize(values):
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]
