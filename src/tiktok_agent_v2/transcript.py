from pathlib import Path
from faster_whisper import WhisperModel


def transcribe(audio_path: Path, model_name: str = "small", device: str = "auto"):
    model = WhisperModel(model_name, device=device, compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path), vad_filter=True, word_timestamps=True
    )

    results = []
    for seg in segments:
        # Build word-level timestamp list
        words_with_ts = []
        if seg.words:
            for w in seg.words:
                words_with_ts.append({
                    "word": w.word.strip(),
                    "start": round(float(w.start), 2),
                    "end": round(float(w.end), 2),
                })

        results.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
            "words": words_with_ts,
        })

    return results, info
