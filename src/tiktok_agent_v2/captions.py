from pathlib import Path
import re
from typing import List, Tuple

Cue = Tuple[float, float, str]
# (cue_start, cue_end, [(word_start, word_end, token), ...])
KaraokeCue = Tuple[float, float, List[Tuple[float, float, str]]]


def _format_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _normalize_caption_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    clean = re.sub(r"\s+([,!.?;:])", r"\1", clean)
    return clean


def _word_timeline_for_clip(segments: List[dict], clip_start: float, clip_end: float):
    words = []
    for seg in segments:
        for word in seg.get("words", []):
            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            token = str(word.get("word", "")).strip()
            if not token:
                continue
            if end < clip_start or start > clip_end:
                continue
            words.append((max(start, clip_start), min(end, clip_end), token))
    return words


def _chunk_words(words, max_words: int, max_chars: int) -> List[Cue]:
    cues: List[Cue] = []
    buffer = []
    start_time = None

    for start, end, token in words:
        if start_time is None:
            start_time = start

        tentative_text = _normalize_caption_text(" ".join([w[2] for w in buffer] + [token]))
        should_flush = (
            len(buffer) >= max_words
            or len(tentative_text) > max_chars
            or (buffer and re.search(r"[.!?]$", buffer[-1][2]))
        )

        if should_flush and buffer:
            cue_text = _normalize_caption_text(" ".join(w[2] for w in buffer))
            cues.append((start_time, buffer[-1][1], cue_text))
            buffer = []
            start_time = start

        buffer.append((start, end, token))

    if buffer and start_time is not None:
        cue_text = _normalize_caption_text(" ".join(w[2] for w in buffer))
        cues.append((start_time, buffer[-1][1], cue_text))

    return cues


def _segment_fallback(segments: List[dict], clip_start: float, clip_end: float) -> List[Cue]:
    cues: List[Cue] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end < clip_start or start > clip_end:
            continue
        text = _normalize_caption_text(str(seg.get("text", "")))
        if not text:
            continue
        cues.append((max(start, clip_start), min(end, clip_end), text))
    return cues


def _enforce_min_duration(cues: List[Cue], clip_start: float, clip_end: float, min_duration: float):
    if not cues:
        return cues

    fixed = []
    for idx, (start, end, text) in enumerate(cues):
        min_end = start + min_duration
        next_start = cues[idx + 1][0] if idx + 1 < len(cues) else clip_end
        end = max(end, min_end)
        end = min(end, next_start, clip_end)
        if end <= start:
            continue
        fixed.append((start, end, text))
    return fixed


def create_clip_captions_srt(
    segments: List[dict],
    clip_start: float,
    clip_end: float,
    out_path: Path,
    max_words: int = 4,
    max_chars: int = 28,
    min_duration: float = 0.45,
) -> bool:
    words = _word_timeline_for_clip(segments, clip_start, clip_end)
    cues = _chunk_words(words, max_words=max_words, max_chars=max_chars) if words else []
    if not cues:
        cues = _segment_fallback(segments, clip_start, clip_end)

    cues = _enforce_min_duration(cues, clip_start=clip_start, clip_end=clip_end, min_duration=min_duration)
    if not cues:
        return False

    lines = []
    for idx, (start, end, text) in enumerate(cues, 1):
        start_rel = max(0.0, start - clip_start)
        end_rel = max(start_rel + 0.05, end - clip_start)
        lines.append(str(idx))
        lines.append(f"{_format_srt_time(start_rel)} --> {_format_srt_time(end_rel)}")
        lines.append(text.upper())
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# ASS karaoke caption generation
# ---------------------------------------------------------------------------

def _format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format: H:MM:SS.CC (centiseconds)."""
    total_cs = max(0, int(round(seconds * 100)))
    hours = total_cs // 360000
    total_cs %= 360000
    minutes = total_cs // 6000
    total_cs %= 6000
    secs = total_cs // 100
    cs = total_cs % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _chunk_words_karaoke(
    words: List[Tuple[float, float, str]],
    max_words: int,
    max_chars: int,
) -> List[KaraokeCue]:
    """Group words into display chunks, preserving per-word timing."""
    cues: List[KaraokeCue] = []
    buffer: List[Tuple[float, float, str]] = []

    for start, end, token in words:
        tentative_text = " ".join(w[2] for w in buffer + [(start, end, token)])
        should_flush = (
            len(buffer) >= max_words
            or len(tentative_text) > max_chars
            or (buffer and re.search(r"[.!?]$", buffer[-1][2]))
        )

        if should_flush and buffer:
            cues.append((buffer[0][0], buffer[-1][1], list(buffer)))
            buffer = []

        buffer.append((start, end, token))

    if buffer:
        cues.append((buffer[0][0], buffer[-1][1], list(buffer)))

    return cues


def _build_karaoke_line(words_in_chunk: List[Tuple[float, float, str]]) -> str:
    """Build ASS karaoke override tags for one chunk of words.

    Uses \\kf (smooth fill) so the highlight sweeps across each word.
    """
    parts = []
    for i, (start, end, token) in enumerate(words_in_chunk):
        if i + 1 < len(words_in_chunk):
            duration_cs = int(round((words_in_chunk[i + 1][0] - start) * 100))
        else:
            duration_cs = int(round((end - start) * 100))
        duration_cs = max(10, duration_cs)
        parts.append(f"{{\\kf{duration_cs}}}{token}")
    return " ".join(parts)


_ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,72,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,40,40,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def create_clip_captions_ass(
    segments: List[dict],
    clip_start: float,
    clip_end: float,
    out_path: Path,
    max_words: int = 4,
    max_chars: int = 28,
    min_duration: float = 0.45,
) -> bool:
    """Generate an ASS subtitle file with word-by-word karaoke highlighting."""
    words = _word_timeline_for_clip(segments, clip_start, clip_end)
    karaoke_cues = _chunk_words_karaoke(words, max_words, max_chars) if words else []

    if karaoke_cues:
        events = []
        for cue_start, cue_end, chunk_words in karaoke_cues:
            # Clip-relative times
            rel_start = max(0.0, cue_start - clip_start)
            rel_end = max(rel_start + 0.05, cue_end - clip_start)
            # Enforce minimum duration
            if rel_end - rel_start < min_duration:
                rel_end = rel_start + min_duration
            # Make word times clip-relative for karaoke tag durations
            rel_words = [
                (max(0.0, s - clip_start), max(0.0, e - clip_start), t)
                for s, e, t in chunk_words
            ]
            text = _build_karaoke_line(rel_words)
            events.append(
                f"Dialogue: 0,{_format_ass_time(rel_start)},{_format_ass_time(rel_end)},"
                f"Default,,0,0,0,,{text.upper()}"
            )
    else:
        # Fallback: plain timed text (no karaoke tags)
        fallback = _segment_fallback(segments, clip_start, clip_end)
        fallback = _enforce_min_duration(fallback, clip_start, clip_end, min_duration)
        if not fallback:
            return False
        events = []
        for start, end, text in fallback:
            rel_start = max(0.0, start - clip_start)
            rel_end = max(rel_start + 0.05, end - clip_start)
            events.append(
                f"Dialogue: 0,{_format_ass_time(rel_start)},{_format_ass_time(rel_end)},"
                f"Default,,0,0,0,,{text.upper()}"
            )

    if not events:
        return False

    out_path.write_text(_ASS_HEADER + "\n".join(events) + "\n", encoding="utf-8")
    return True
