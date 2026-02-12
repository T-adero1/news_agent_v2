import logging
import os
import time
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


def _log_memory(label: str = ""):
    """Log current process memory usage. Works on Linux and Windows."""
    try:
        import sys
        if sys.platform == "linux":
            # /proc/self/status is always available on Linux, no pip install needed
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        log.info("  [MEM %s] RSS=%.0f MB", label, rss_kb / 1024)
                        return
        else:
            import os
            # Windows: rough estimate via private working set
            import ctypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                log.info("  [MEM %s] RSS=%.0f MB", label, pmc.WorkingSetSize / (1024 * 1024))
                return
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
    hook_lead_in: float = 0.10,
    hook_scan: bool = True,
):
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    pipeline_t0 = time.monotonic()

    # Log video metadata for debugging server issues
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    log.info("=== PIPELINE START ===")
    log.info("  Video: %s (%.1f MB)", video_path.name, video_size_mb)
    log.info("  Settings: clips=%d duration=%.0fs format=%s captions=%s hook_lead_in=%.0f%%",
             num_clips, clip_duration, format_method, captions_enabled, hook_lead_in * 100)
    log.info("  Whisper: model=%s device=%s", whisper_model, device)
    log.info("  LLM: model=%s batch=%d timeout=%ds strict=%s",
             llm_model, llm_batch_size, llm_timeout_s, strict_llm)
    _log_memory("pipeline_start")
    _flush_log_handlers()

    log.info(">>> Step 1: about to extract audio from %s", video_path.name)
    _flush_log_handlers()
    console.print("[bold]1) Extracting audio[/bold]")
    t0 = time.monotonic()
    audio_path = out_dir / (video_path.stem + "_audio.wav")
    extract_audio(video_path, audio_path)
    log.info("<<< Step 1 (extract_audio) took %.1fs", time.monotonic() - t0)
    _log_memory("after_extract_audio")
    _flush_log_handlers()

    log.info(">>> Step 2: about to transcribe %s (model=%s, device=%s)", audio_path.name, whisper_model, device)
    _flush_log_handlers()
    console.print("[bold]2) Transcribing[/bold]")
    t0 = time.monotonic()
    segments, info = transcribe(audio_path, model_name=whisper_model, device=device)
    log.info("<<< Step 2 (transcribe) took %.1fs", time.monotonic() - t0)
    _log_memory("after_transcribe")
    _flush_log_handlers()

    log.info(">>> Step 2.1: quality gate check (%d segments)", len(segments) if segments else 0)
    _flush_log_handlers()
    if not segments:
        console.print("[red]No transcript segments found[/red]")
        return

    # Quality gate: require a minimum amount of speech to produce useful clips
    total_words = sum(len(seg.get("text", "").split()) for seg in segments)
    if total_words < 10:
        console.print(f"[red]Transcript too short ({total_words} words, need ≥10) — skipping[/red]")
        log.warning("Transcript has only %d words — below minimum threshold of 10, skipping", total_words)
        return

    console.print(f"Transcript segments: {len(segments)}")
    log.info("=== TRANSCRIPT SEGMENTS (%d total, %d words) ===", len(segments), total_words)
    for seg in segments:
        log.info("  [%6.1fs - %6.1fs] %s", seg["start"], seg["end"], seg["text"])
    full_transcript_path = out_dir / "transcript_full.txt"
    _write_full_transcript(segments, full_transcript_path)
    log.info("Wrote full transcript: %s", full_transcript_path)
    _flush_log_handlers()

    log.info(">>> Step 3: about to score motion (fps=%.1f)", motion_fps)
    _flush_log_handlers()
    console.print("[bold]3) Motion scoring[/bold]")
    t0 = time.monotonic()
    motion_scores = compute_motion_scores(video_path, sample_fps=motion_fps)
    log.info("<<< Step 3 (motion_scoring) took %.1fs — %d samples", time.monotonic() - t0, len(motion_scores))
    _log_memory("after_motion")
    _flush_log_handlers()

    log.info(">>> Step 4: about to rank segments (llm=%s)", llm_model)
    _flush_log_handlers()
    console.print("[bold]4) Ranking segments[/bold]")
    t0 = time.monotonic()

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

        # Hook-first: place peak near the START of the clip (lead_in_sec into it)
        lead_in_sec = clip_duration * hook_lead_in
        raw_start = seg["start"] - lead_in_sec
        start = max(0, raw_start)
        end = min(_video_duration, start + clip_duration)
        # Re-adjust start if end was clamped
        start = max(0, end - clip_duration)

        log.info("  Considering seg id=%d [%.1fs-%.1fs] score=%.3f",
                 seg["id"], seg["start"], seg["end"], seg["combined_score"])
        log.info("    Hook-first: peak at %.1fs, lead-in=%.1fs (%.0f%% of %.0fs clip)",
                 seg["start"], lead_in_sec, hook_lead_in * 100, clip_duration)
        log.info("    Ideal window: %.1fs - %.1fs", raw_start, raw_start + clip_duration)
        log.info("    Clamped window: %.1fs - %.1fs", start, end)

        # Two-pass hook scan: check if the opening seconds are weak
        # and shift forward to a stronger hook if one exists nearby
        if not hook_scan:
            log.info("    Hook scan: disabled")
        else:
            early_segs = [s for s in segments
                          if s["end"] > start and s["start"] < start + 12]
            if early_segs:
                best_hook = max(early_segs, key=lambda s: s["text_score"])
                if best_hook["start"] > start + 3:
                    old_start = start
                    start = max(0, best_hook["start"] - 2.0)
                    end = min(_video_duration, start + clip_duration)
                    start = max(0, end - clip_duration)
                    log.info("    Hook scan: trimmed filler, shifted start %.1fs -> %.1fs "
                             "(best opener seg id=%d at %.1fs, text_score=%.3f)",
                             old_start, start, best_hook["id"], best_hook["start"],
                             best_hook["text_score"])
                else:
                    log.info("    Hook scan: best opener in first 12s is seg id=%d at %.1fs (no shift needed)",
                             best_hook["id"], best_hook["start"])
            else:
                log.info("    Hook scan: no segments in first 12s")

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

    log.info("Step 4 (ranking + clip selection) took %.1fs", time.monotonic() - t0)
    log.info("Pipeline elapsed so far: %.1fs", time.monotonic() - pipeline_t0)
    _flush_log_handlers()

    log.info(">>> Step 5: extracting %d clip(s)", len(selected))
    _flush_log_handlers()
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

        log.info("Clip %d: >>> clip_video %.1fs-%.1fs", idx, seg["start"], seg["end"])
        _flush_log_handlers()
        t0 = time.monotonic()
        clip_video(video_path, seg["start"], seg["end"], raw_path)
        log.info("Clip %d: <<< clip_video took %.1fs", idx, time.monotonic() - t0)
        _flush_log_handlers()

        t0 = time.monotonic()
        log.info("Clip %d: >>> format_%s (raw=%s)", idx, format_method, raw_path.name)
        _log_memory("before_format")
        _flush_log_handlers()
        if format_method == "smart-crop":
            from tiktok_agent_v2.reframe import format_tiktok_smart_crop
            format_tiktok_smart_crop(raw_path, final_path, out_width, out_height)
        elif format_method == "center-crop":
            format_tiktok_center_crop(raw_path, final_path, out_width, out_height)
        else:
            format_tiktok_blur(raw_path, final_path, out_width, out_height)
        log.info("Clip %d: <<< format_%s took %.1fs", idx, format_method, time.monotonic() - t0)
        _log_memory("after_format")
        _flush_log_handlers()

        if captions_enabled:
            log.info("Clip %d: >>> caption flow starting", idx)
            _flush_log_handlers()

            t0 = time.monotonic()
            has_ass = create_clip_captions_ass(
                segments=segments,
                clip_start=seg["start"],
                clip_end=seg["end"],
                out_path=captions_ass_path,
                max_words=captions_max_words,
                max_chars=captions_max_chars,
            )
            log.info("Clip %d: ASS generation took %.1fs (ok=%s)", idx, time.monotonic() - t0, has_ass)

            if has_ass:
                log.info("Clip %d: >>> burn_in_captions", idx)
                _flush_log_handlers()
                t0 = time.monotonic()
                burn_in_captions(final_path, captions_ass_path, captioned_path)
                log.info("Clip %d: <<< burn_in_captions took %.1fs", idx, time.monotonic() - t0)
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

    total_elapsed = time.monotonic() - pipeline_t0
    log.info("=== PIPELINE COMPLETE === Total elapsed: %.1fs", total_elapsed)
    _flush_log_handlers()
    console.print(f"[green]Done ({total_elapsed:.0f}s total)[/green]")
