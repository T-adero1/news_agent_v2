# TikTok Agent V2 — Fixed-Config Clip Extractor

This project extracts short viral-style clips from a long video using:
- transcription (`faster-whisper`)
- LLM ranking (`gpt-5-mini`)
- motion scoring (OpenCV)
- 9:16 formatting + burned-in captions

## Usage

Run with one required argument:

```bash
python main.py --video /path/to/video.mp4
```

## Hardcoded Behavior

- Outputs to `outputs/`
- Extracts 3 clips
- Uses 45-second clip windows (plus sentence-boundary snapping)
- Uses LLM model `gpt-5-mini` (requires `OPENAI_API_KEY`)
- Captions are always enabled
- Full transcript is always written
- Per-clip transcript+score files are always written

## Required Environment Variable

- `OPENAI_API_KEY` (required)

## Output Files

- `outputs/clip_01_<start>s_<end>s.mp4`
- `outputs/clip_02_<start>s_<end>s.mp4`
- `outputs/clip_03_<start>s_<end>s.mp4`
- `outputs/transcript_full.txt`
- `outputs/clip_01_<start>s_<end>s_transcript.txt`
- `outputs/clip_02_<start>s_<end>s_transcript.txt`
- `outputs/clip_03_<start>s_<end>s_transcript.txt`
- `outputs/pipeline.log`
