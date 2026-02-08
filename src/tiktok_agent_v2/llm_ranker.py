import json
import logging
import os
import re
from typing import Dict, List, Optional

import requests

log = logging.getLogger("tiktok_agent_v2.llm_ranker")

DEFAULT_BASE_URL = "https://api.openai.com/v1/responses"


def _extract_output_text(resp_json: dict) -> str:
    """Extract concatenated output_text from Responses API output."""
    parts = []
    for item in resp_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "".join(parts).strip()


def _parse_scores(text: str) -> Dict[int, float]:
    """Parse JSON scores from model output."""
    if not text:
        return {}

    # Try direct JSON
    try:
        data = json.loads(text)
        return _coerce_scores(data)
    except Exception:
        pass

    # Try to extract JSON block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group(0))
            return _coerce_scores(data)
        except Exception:
            return {}
    return {}


def _coerce_scores(data) -> Dict[int, float]:
    scores: Dict[int, float] = {}

    if isinstance(data, dict) and "scores" in data:
        data = data["scores"]

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item and "score" in item:
                try:
                    scores[int(item["id"])] = float(item["score"])
                except Exception:
                    continue
        return scores

    if isinstance(data, dict):
        for k, v in data.items():
            try:
                scores[int(k)] = float(v)
            except Exception:
                continue
    return scores


def _build_prompt(batch):
    lines = []
    for seg in batch:
        # Include word-level timestamps if available
        words = seg.get("words", [])
        if words:
            word_parts = []
            for w in words:
                word_parts.append(f"{w['word']}({w['start']:.1f}s)")
            timestamped_text = " ".join(word_parts)
            lines.append(
                f"[id={seg['id']} start={seg['start']:.2f} end={seg['end']:.2f}]\n"
                f"  Text: {seg['text']}\n"
                f"  Words: {timestamped_text}"
            )
        else:
            lines.append(
                f"[id={seg['id']} start={seg['start']:.2f} end={seg['end']:.2f}] {seg['text']}"
            )
    return "\n".join(lines)


def score_segments_with_llm(
    segments: List[dict],
    model: str,
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    batch_size: int = 40,
    timeout_s: int = 60,
) -> Dict[int, float]:
    """
    Returns a mapping {segment_id: score(0..1)} using an LLM.
    If any batch fails, it is skipped.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        return {}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    scores: Dict[int, float] = {}

    system_text = (
        "You are ranking transcript segments for viral TikTok/Shorts clip potential. "
        "Score each segment 0..1 based on: hooks that grab attention, emotional peaks, "
        "punchlines, surprising statements, key revelations, dramatic moments, humor, "
        "and quotable lines. Penalize filler, tangents, and repeated info. "
        "Word timestamps are provided so you can identify exact peak moments. "
        "Return JSON only."
    )

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        prompt = _build_prompt(batch)
        batch_num = i // batch_size + 1

        log.info("=== LLM Batch %d (%d segments) ===", batch_num, len(batch))
        log.info("System prompt: %s", system_text)
        log.info("User prompt:\n%s", prompt)

        user_text = (
            "Segments:\n" + prompt + "\n\n"
            "Return JSON as a list of {id, score} objects, or a dict of id->score."
        )

        payload = {
            "model": model,
            "max_output_tokens": 5000,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": system_text},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_text},
                    ],
                },
            ],
        }

        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            resp_json = resp.json()
            log.info("LLM API response keys: %s", list(resp_json.keys()))
            log.info("LLM API full response:\n%s", json.dumps(resp_json, indent=2)[:3000])
            text = _extract_output_text(resp_json)
            log.info("Extracted output text:\n%s", text if text else "(EMPTY)")
            batch_scores = _parse_scores(text)
            log.info("Parsed scores (%d segments): %s", len(batch_scores), batch_scores)
            scores.update(batch_scores)
        except Exception as exc:
            log.error("Batch %d failed: %s", batch_num, exc)
            continue

    return scores
