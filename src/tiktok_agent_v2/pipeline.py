import logging
import os
from pathlib import Path

from rich.console import Console

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from tiktok_agent_v2.ffmpeg_utils import (
    extract_audio,
    clip_video,
    format_tiktok_blur,
    format_tiktok_center_crop,
    burn_in_captions,
)
from tiktok_agent_v2.motion import compute_motion_scores, aggregate_motion_for_window
from tiktok_agent_v2.transcript import transcribe
from tiktok_agent_v2.ranker import score_transcript_segments, apply_time_decay, normalize
from tiktok_agent_v2.llm_ranker import score_segments_with_llm
from tiktok_agent_v2.captions import create_clip_captions_ass, create_clip_captions_srt

console = Console()
log = logging.getLogger("tiktok_agent_v2.pipeline")

# Max seconds we'll extend or contract a clip boundary to reach a sentence edge
_SNAP_TOLERANCE = 5.0


def _flush_log_handlers():
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _segments_in_window(segments, start, end):
    return [s for s in segments if s["end"] >= start and s["start"] <= end]


def _write_full_transcript(segments, out_path: Path):
    lines = ["# Full Transcript", ""]
    for seg in segments:
        lines.append(f"[{seg['start']:7.2f}s - {seg['end']:7.2f}s] {seg['text']}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_clip_transcript(clip_idx: int, clip_start: float, clip_end: float, segments, out_path: Path):
    lines = [
        f"# Clip {clip_idx:02d} Transcript",
        f"Window: {clip_start:.2f}s - {clip_end:.2f}s",
        "",
    ]
    for seg in _segments_in_window(segments, clip_start, clip_end):
        raw = seg.get("text_score", 0.0)
        txt = seg.get("text_norm", 0.0)
        mot = seg.get("motion_norm", 0.0)
        cmb = seg.get("combined_score", 0.0)
        lines.append(
            f"[{seg['start']:7.2f}s - {seg['end']:7.2f}s] "
            f"raw={raw:.3f} txt={txt:.3f} mot={mot:.3f} cmb={cmb:.3f} | {seg['text']}"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _snap_to_sentences(start, end, segments, video_duration):
    """
    Adjust start/end so the clip doesn't cut mid-sentence.

    - End: if a transcript segment is being cut, extend to its end
      (or pull back to the previous segment's end if extending is too far).
    - Start: if a segment is being cut, pull back to its start
      (or push forward to the next segment's start).
    """
    original_start, original_end = start, end

    # --- snap END to a sentence boundary ---
    # Find the segment that straddles the end boundary
    for seg in segments:
        if seg["start"] < end < seg["end"]:
            # We're cutting this segment mid-sentence
            extend = seg["end"] - end
            contract = end - seg["start"]
            if extend <= _SNAP_TOLERANCE:
                # Small extension — let the sentence finish
                end = seg["end"]
            elif contract <= _SNAP_TOLERANCE:
                # Pull back to before this segment started
                end = seg["start"]
            break

    # --- snap START to a sentence boundary ---
    # Find the segment that straddles the start boundary
    for seg in segments:
        if seg["start"] < start < seg["end"]:
            # We're starting mid-sentence
            contract = seg["end"] - start
            extend = start - seg["start"]
            if extend <= _SNAP_TOLERANCE:
                # Small extension — include the full sentence
                start = seg["start"]
            elif contract <= _SNAP_TOLERANCE:
                # Push forward past this partial sentence
                start = seg["end"]
            break

    # Clamp to video bounds
    start = max(0, start)
    end = min(video_duration, end)

    if start != original_start or end != original_end:
        log.info("    Snapped to sentence boundaries: %.1fs - %.1fs (was %.1fs - %.1fs)",
                 start, end, original_start, original_end)

    return round(start, 2), round(end, 2)


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    num_clips: int,
    clip_duration: float,
    min_gap: float,
    prefer_early: bool,
    early_half_life: float,
    whisper_model: str,
    device: str,
    motion_fps: float,
    motion_weight: float,
    text_weight: float,
    format_method: str,
    out_width: int,
    out_height: int,
    llm_model,
    llm_batch_size: int,
    llm_timeout_s: int,
    captions_enabled: bool,
    captions_max_words: int,
    captions_max_chars: int,
    strict_llm: bool = False,
):
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    console.print("[bold]1) Extracting audio[/bold]")
    audio_path = out_dir / (video_path.stem + "_audio.wav")
    extract_audio(video_path, audio_path)

    console.print("[bold]2) Transcribing[/bold]")
    segments, info = transcribe(audio_path, model_name=whisper_model, device=device)
    if not segments:
        console.print("[red]No transcript segments found[/red]")
        return

    console.print(f"Transcript segments: {len(segments)}")
    log.info("=== TRANSCRIPT SEGMENTS ===")
    for seg in segments:
        log.info("  [%6.1fs - %6.1fs] %s", seg["start"], seg["end"], seg["text"])
    full_transcript_path = out_dir / "transcript_full.txt"
    _write_full_transcript(segments, full_transcript_path)
    log.info("Wrote full transcript: %s", full_transcript_path)

    console.print("[bold]3) Motion scoring[/bold]")
    motion_scores = compute_motion_scores(video_path, sample_fps=motion_fps)

    console.print("[bold]4) Ranking segments[/bold]")

    # Assign stable ids for LLM scoring
    for idx, seg in enumerate(segments):
        seg["id"] = idx

    llm_scores = {}
    if llm_model:
        if not os.environ.get("OPENAI_API_KEY"):
            if strict_llm:
                raise RuntimeError("OPENAI_API_KEY not set, but strict LLM mode is enabled.")
            console.print("[yellow]Warning: OPENAI_API_KEY not set — falling back to heuristic scoring[/yellow]")
        else:
            console.print(f"Using LLM model: {llm_model}")
            llm_scores = score_segments_with_llm(
                segments,
                model=llm_model,
                batch_size=llm_batch_size,
                timeout_s=llm_timeout_s,
            )

    if llm_scores:
        log.info("=== LLM SCORES (raw from API) ===")
        for seg in segments:
            raw = llm_scores.get(seg["id"], 0.0)
            seg["text_score"] = float(raw)
            log.info("  id=%2d [%6.1fs-%6.1fs] llm=%.3f  %s",
                     seg["id"], seg["start"], seg["end"], raw, seg["text"][:80])
    else:
        if strict_llm and llm_model:
            raise RuntimeError("Strict LLM mode enabled, but no LLM scores were returned.")
        log.info("No LLM scores — using heuristic fallback")
        segments = score_transcript_segments(segments)

    # motion score per transcript segment
    motion_vals = []
    text_vals = []
    for seg in segments:
        mscore = aggregate_motion_for_window(motion_scores, seg["start"], seg["end"])
        seg["motion_score"] = mscore
        motion_vals.append(mscore)
        text_vals.append(seg["text_score"])

    motion_norm = normalize(motion_vals)
    text_norm = normalize(text_vals)

    ranked = []
    for i, seg in enumerate(segments):
        tscore = text_norm[i]
        mscore = motion_norm[i]
        combined = (text_weight * tscore) + (motion_weight * mscore)
        if prefer_early:
            combined = apply_time_decay(combined, seg["start"], early_half_life)
        seg["text_norm"] = tscore
        seg["motion_norm"] = mscore
        seg["combined_score"] = combined
        ranked.append(seg)

    ranked.sort(key=lambda s: s["combined_score"], reverse=True)

    log.info("=== SEGMENT RANKING (text_w=%.1f, motion_w=%.1f) ===", text_weight, motion_weight)
    log.info("  %-4s %-14s %-10s %-10s %-10s %-10s  %s",
             "Rank", "Time", "LLM/Text", "TxtNorm", "MotNorm", "Combined", "Text")
    for rank, seg in enumerate(ranked, 1):
        i = seg["id"]
        log.info("  #%-3d [%5.1fs-%5.1fs] raw=%.3f  txt=%.3f  mot=%.3f  cmb=%.3f  %s",
                 rank, seg["start"], seg["end"],
                 seg["text_score"], text_norm[i], motion_norm[i],
                 seg["combined_score"], seg["text"][:60])

    # Get video duration for bounds checking
    import cv2
    _cap = cv2.VideoCapture(str(video_path))
    _video_duration = _cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(_cap.get(cv2.CAP_PROP_FPS), 1)
    _cap.release()
    log.info("Video duration: %.1fs | Requested clip: %.1fs | Requesting %d clip(s)", _video_duration, clip_duration, num_clips)

    # Select top segments with min gap
    selected = []
    log.info("=== CLIP SELECTION PROCESS ===")
    for seg in ranked:
        if len(selected) >= num_clips:
            break

        # Expand window around the key segment to fill clip_duration
        seg_mid = (seg["start"] + seg["end"]) / 2
        half = clip_duration / 2
        raw_start = seg_mid - half
        start = max(0, raw_start)
        end = min(_video_duration, start + clip_duration)
        # Re-adjust start if end was clamped
        start = max(0, end - clip_duration)

        log.info("  Considering seg id=%d [%.1fs-%.1fs] score=%.3f",
                 seg["id"], seg["start"], seg["end"], seg["combined_score"])
        log.info("    Segment midpoint: %.1fs", seg_mid)
        log.info("    Ideal window: %.1fs - %.1fs  (midpoint ± %.1fs)",
                 raw_start, raw_start + clip_duration, half)
        log.info("    Clamped window: %.1fs - %.1fs", start, end)

        # Snap to sentence boundaries so we don't cut mid-sentence
        start, end = _snap_to_sentences(start, end, segments, _video_duration)

        # enforce min gap
        overlaps = False
        for s in selected:
            if not (end + min_gap <= s["start"] or start >= s["end"] + min_gap):
                overlaps = True
                log.info("    SKIPPED — overlaps with already-selected [%.1fs-%.1fs]", s["start"], s["end"])
                break
        if overlaps:
            continue

        log.info("    SELECTED as clip %d: [%.1fs - %.1fs]", len(selected) + 1, start, end)
        selected.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "score": seg["combined_score"],
            "text": seg["text"],
        })

    if not selected:
        console.print("[red]No segments selected[/red]")
        return

    console.print("[bold]5) Extracting clips[/bold]")
    for idx, seg in enumerate(selected, start=1):
        raw_path = out_dir / f"clip_{idx:02d}_{int(seg['start'])}s_{int(seg['end'])}s_raw.mp4"
        final_path = out_dir / f"clip_{idx:02d}_{int(seg['start'])}s_{int(seg['end'])}s.mp4"
        captioned_path = out_dir / f"clip_{idx:02d}_{int(seg['start'])}s_{int(seg['end'])}s_captioned.mp4"
        captions_ass_path = out_dir / f"clip_{idx:02d}_{int(seg['start'])}s_{int(seg['end'])}s.ass"
        captions_srt_path = out_dir / f"clip_{idx:02d}_{int(seg['start'])}s_{int(seg['end'])}s.srt"
        clip_transcript_path = out_dir / f"clip_{idx:02d}_{int(seg['start'])}s_{int(seg['end'])}s_transcript.txt"

        _write_clip_transcript(
            clip_idx=idx,
            clip_start=seg["start"],
            clip_end=seg["end"],
            segments=segments,
            out_path=clip_transcript_path,
        )
        log.info("Wrote clip transcript: %s", clip_transcript_path)

        clip_video(video_path, seg["start"], seg["end"], raw_path)

        if format_method == "center-crop":
            format_tiktok_center_crop(raw_path, final_path, out_width, out_height)
        else:
            format_tiktok_blur(raw_path, final_path, out_width, out_height)

        if captions_enabled:
            log.info("=== CAPTION FLOW (clip %d) ===", idx)
            log.info("Clip %d caption inputs: video=%s start=%.2fs end=%.2fs", idx, final_path, seg["start"], seg["end"])

            has_ass = create_clip_captions_ass(
                segments=segments,
                clip_start=seg["start"],
                clip_end=seg["end"],
                out_path=captions_ass_path,
                max_words=captions_max_words,
                max_chars=captions_max_chars,
            )
            if has_ass:
                log.info(
                    "Clip %d captions: ASS generated (%d bytes), burning into video",
                    idx,
                    captions_ass_path.stat().st_size if captions_ass_path.exists() else 0,
                )
                burn_in_captions(final_path, captions_ass_path, captioned_path)
            else:
                log.warning("Clip %d: ASS generation failed, no captions for this clip", idx)

            if captioned_path.exists():
                final_path.unlink()
                captioned_path.rename(final_path)
                log.info("Clip %d caption burn succeeded (%s)", idx, final_path)
            if captions_ass_path.exists():
                captions_ass_path.unlink()
            log.info("=== END CAPTION FLOW (clip %d) ===", idx)

        # Clean up raw intermediate file
        if raw_path.exists() and final_path.exists():
            raw_path.unlink()

        console.print(f"Saved {final_path.name} | score={seg['score']:.3f} | {seg['text']}")
        _flush_log_handlers()

    # Clean up intermediate audio file
    if audio_path.exists():
        audio_path.unlink()

    console.print("[green]Done[/green]")
