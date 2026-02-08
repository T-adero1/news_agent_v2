import argparse
import logging
import os
import shutil
from pathlib import Path
from tiktok_agent_v2.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 45s clips with LLM ranking and captions")
    parser.add_argument("--video", required=True, help="Path to input video")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in out_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set. This script is hardcoded to use gpt-5-mini.")

    # Set up logging — file gets everything, console gets a summary
    log_path = out_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("tiktok_agent_v2").setLevel(logging.INFO)
    print(f"Detailed logs -> {log_path}")

    run_pipeline(
        video_path=video_path,
        out_dir=out_dir,
        num_clips=3,
        clip_duration=45.0,
        min_gap=6.0,
        prefer_early=False,
        early_half_life=45.0,
        whisper_model="small",
        device="cpu",
        motion_fps=2.0,
        motion_weight=0.4,
        text_weight=0.6,
        format_method="blur",
        out_width=1080,
        out_height=1920,
        llm_model="gpt-5-mini",
        llm_batch_size=40,
        llm_timeout_s=60,
        captions_enabled=True,
        captions_max_words=4,
        captions_max_chars=28,
        strict_llm=True,
    )


if __name__ == "__main__":
    main()
