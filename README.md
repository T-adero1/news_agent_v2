# TikTok Agent V2 — Viral Clip Extractor

Automatically extracts short-form viral clips (TikTok/Shorts/Reels) from longer videos. Uses speech transcription + LLM ranking + motion analysis to identify the most engaging moments, then cuts and formats them for 9:16 vertical video.

## How It Works

The pipeline runs 5 sequential steps:

1. **Extract audio** — Pulls a 16kHz mono WAV from the input video via FFmpeg.
2. **Transcribe** — Runs `faster-whisper` (local, offline) to produce timestamped transcript segments with word-level timestamps.
3. **Motion scoring** — Samples video frames via OpenCV and computes mean absolute frame difference per time window.
4. **Rank segments** — Each transcript segment gets a combined score from:
   - **Text score** (weight: `--text-weight`, default 0.6): LLM virality rating (0-1) via OpenAI API, or heuristic word-rarity fallback if no API key.
   - **Motion score** (weight: `--motion-weight`, default 0.4): Normalized motion energy for that segment's time window.
   - **Time decay** (optional, `--prefer-early`): Exponential decay favoring earlier content.
5. **Extract & format clips** — Cuts the top-scoring non-overlapping windows, snaps boundaries to sentence edges (within 5s tolerance), and formats to 9:16 vertical.

## Prerequisites

- **Python 3.9+**
- **FFmpeg** installed and on PATH (`brew install ffmpeg` / `apt install ffmpeg`)
- **OPENAI_API_KEY** environment variable (for LLM scoring; optional — falls back to heuristic without it)

## Installation

```bash
cd tiktok_agent_v2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Dependencies (requirements.txt)

```
opencv-python
numpy
faster-whisper
ffmpeg-python
rich
requests
python-dotenv
```

## Usage

### Basic Command

```bash
python main.py --video /path/to/video.mp4
```

This will produce 3 clips of 18s each in `./outputs/`.

### Full Command with All Options

```bash
python main.py \
  --video /path/to/video.mp4 \
  --out outputs \
  --num-clips 3 \
  --clip-duration 45 \
  --min-gap 6 \
  --prefer-early \
  --early-half-life 45 \
  --whisper-model small \
  --device auto \
  --motion-fps 2.0 \
  --motion-weight 0.4 \
  --text-weight 0.6 \
  --format blur \
  --out-width 1080 \
  --out-height 1920 \
  --llm-model gpt-5-mini \
  --llm-batch-size 40 \
  --llm-timeout 60
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--video` | str | **required** | Path to the input video file (.mp4) |
| `--out` | str | `outputs` | Output directory for clips and logs |
| `--num-clips` | int | `3` | Number of clips to extract |
| `--clip-duration` | float | `18.0` | Target clip duration in seconds (may extend slightly to avoid cutting mid-sentence) |
| `--min-gap` | float | `6.0` | Minimum seconds between selected clips (prevents overlap) |
| `--prefer-early` | flag | off | Apply exponential time decay to prefer earlier content |
| `--early-half-life` | float | `45.0` | Half-life in seconds for time decay (only used with `--prefer-early`) |
| `--whisper-model` | str | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--device` | str | `auto` | Whisper compute device: `auto`, `cpu`, `cuda` |
| `--motion-fps` | float | `2.0` | Frames per second sampled for motion scoring |
| `--motion-weight` | float | `0.4` | Weight for motion score in combined ranking (0.0-1.0) |
| `--text-weight` | float | `0.6` | Weight for transcript score in combined ranking (0.0-1.0) |
| `--format` | str | `blur` | Output format: `blur` (fit with blurred bg) or `center-crop` (crop to fill) |
| `--out-width` | int | `1080` | Output video width in pixels |
| `--out-height` | int | `1920` | Output video height in pixels |
| `--llm-model` | str | `gpt-5-mini` | OpenAI model for transcript ranking |
| `--no-llm` | flag | off | Disable LLM ranking, use heuristic word-rarity scoring only |
| `--llm-batch-size` | int | `40` | Max transcript segments per LLM API call |
| `--llm-timeout` | int | `60` | LLM API request timeout in seconds |

## Output

For each run, the output directory contains:

- `clip_01_<start>s_<end>s.mp4` — Final formatted 9:16 clips
- `<video_stem>_audio.wav` — Extracted audio (intermediate, used for transcription)
- `pipeline.log` — Detailed decision log with transcript, LLM queries/responses, scores, ranking, and clip selection reasoning

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | No | Enables LLM-based transcript scoring. Without it, falls back to heuristic word-rarity scoring. Set in `.env` file or shell environment. |

The pipeline loads `.env` automatically via `python-dotenv`.

## Project Structure

```
tiktok_agent_v2/
├── main.py                          # CLI entry point, argument parsing, logging setup
├── pyproject.toml                   # Package metadata (pip install -e .)
├── requirements.txt                 # Python dependencies
├── .env                             # OPENAI_API_KEY (not committed)
├── outputs/                         # Default output directory
│   ├── clip_01_*.mp4
│   ├── clip_02_*.mp4
│   └── pipeline.log
└── src/tiktok_agent_v2/
    ├── __init__.py
    ├── pipeline.py                  # Main orchestrator — runs all 5 steps
    ├── transcript.py                # Whisper transcription with word-level timestamps
    ├── llm_ranker.py                # LLM-based segment scoring via OpenAI Responses API
    ├── ranker.py                    # Heuristic text scoring (fallback), time decay, normalization
    ├── motion.py                    # Frame-difference motion scoring via OpenCV
    └── ffmpeg_utils.py              # Audio extraction, video clipping, 9:16 formatting
```

## Module Details

### pipeline.py — Orchestrator

`run_pipeline(...)` is the single entry point. It:
1. Extracts audio → transcribes → computes motion scores → ranks → selects → cuts clips.
2. Clip windows are centered on the midpoint of the highest-scoring transcript segment, expanded to `clip_duration`.
3. Boundaries snap to sentence edges (within 5s tolerance) so clips never cut mid-sentence. Clips may be slightly longer than `clip_duration` as a result.
4. Selected clips must be separated by at least `min_gap` seconds. If a candidate overlaps an already-selected clip, it is skipped.

### llm_ranker.py — LLM Scoring

- Calls the **OpenAI Responses API** (`POST /v1/responses`) with `gpt-5-mini` by default.
- Sends transcript segments (with word-level timestamps) in batches.
- System prompt asks the LLM to score each segment 0-1 for viral potential (hooks, emotional peaks, punchlines, surprises, humor, quotable lines).
- Expects JSON response: either `[{id, score}, ...]` or `{id: score, ...}`.
- Uses `max_output_tokens: 4096` (gpt-5-mini is a reasoning model that needs headroom for internal reasoning tokens).
- Falls back gracefully — if the API call fails or returns empty, the pipeline uses heuristic scoring instead.

### ranker.py — Heuristic Fallback

Used when `--no-llm` is set or `OPENAI_API_KEY` is missing:
- Scores segments by word rarity (inverse frequency) with a length boost.
- `normalize(values)` — min-max normalization to [0, 1].
- `apply_time_decay(score, time_sec, half_life)` — exponential decay: `score * 0.5^(time/half_life)`.

### transcript.py — Whisper Transcription

- Uses `faster-whisper` with VAD filter and word-level timestamps.
- Returns list of segments: `{start, end, text, words: [{word, start, end}, ...]}`.
- On first run, the model weights are downloaded automatically (~500MB for `small`).

### motion.py — Motion Scoring

- Samples frames at `motion_fps` (default 2.0) and computes mean absolute pixel difference between consecutive grayscale frames.
- Returns `[(time_sec, motion_score), ...]`.
- `aggregate_motion_for_window(scores, start, end)` averages motion scores within a time range.

### ffmpeg_utils.py — Video Processing

- `extract_audio(video, wav)` — Extracts 16kHz mono WAV.
- `clip_video(video, start, end, out)` — Cuts a time range with H.264/AAC encoding.
- `format_tiktok_blur(video, out, w, h)` — Fits video inside 9:16 frame with blurred background fill. Uses `split` filter for dual-stream processing, SAR normalization, `yuv420p` pixel format.
- `format_tiktok_center_crop(video, out, w, h)` — Crops to 9:16 center (may cut content). SAR normalization, `yuv420p` pixel format.

## Programmatic Usage

```python
from pathlib import Path
from tiktok_agent_v2.pipeline import run_pipeline

run_pipeline(
    video_path=Path("/path/to/video.mp4"),
    out_dir=Path("./outputs"),
    num_clips=3,
    clip_duration=45.0,
    min_gap=6.0,
    prefer_early=True,
    early_half_life=45.0,
    whisper_model="small",
    device="auto",
    motion_fps=2.0,
    motion_weight=0.4,
    text_weight=0.6,
    format_method="blur",
    out_width=1080,
    out_height=1920,
    llm_model="gpt-5-mini",
    llm_batch_size=40,
    llm_timeout_s=60,
)
```

Set `llm_model=None` to disable LLM scoring and use heuristic fallback only.

## Scoring Formula

For each transcript segment `i`:

```
text_norm[i] = min-max normalize(text_scores)
motion_norm[i] = min-max normalize(motion_scores)
combined[i] = (text_weight * text_norm[i]) + (motion_weight * motion_norm[i])
if prefer_early:
    combined[i] *= 0.5 ^ (segment_start_time / early_half_life)
```

Segments are ranked by `combined` score descending. The top segment's midpoint becomes the center of a clip window of length `clip_duration`.

## Logging

All decision-making is logged to `<out_dir>/pipeline.log`:
- Full transcript with timestamps
- LLM system/user prompts and raw API responses
- Parsed LLM scores per segment
- Complete ranking table (text score, motion score, combined)
- Clip selection: midpoint calculation, window clamping, sentence snapping, overlap checks

## Constraints and Limits

- Video must have an audio track (needed for transcription).
- Clips may slightly exceed `clip_duration` (up to 5s) due to sentence boundary snapping.
- If the video is too short for the requested number of non-overlapping clips, fewer clips are returned.
- `gpt-5-mini` requires `max_output_tokens >= 4096` because it consumes ~1200 tokens for internal reasoning before generating the JSON response.
- On first run, Whisper model weights are downloaded (~500MB for `small`, ~3GB for `large-v3`).
