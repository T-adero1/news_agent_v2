import argparse
import logging
from pathlib import Path
from tiktok_agent_v2.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Transcript + motion based clip finder")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--num-clips", type=int, default=3, help="Number of clips to extract")
    parser.add_argument("--clip-duration", type=float, default=18.0, help="Clip duration in seconds")
    parser.add_argument("--min-gap", type=float, default=6.0, help="Minimum gap between clips")
    parser.add_argument("--prefer-early", action="store_true", help="Prefer earlier segments")
    parser.add_argument("--early-half-life", type=float, default=45.0, help="Seconds for time decay half-life")
    parser.add_argument("--whisper-model", default="small", help="Whisper model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--device", default="auto", help="Whisper device: auto, cpu, cuda")
    parser.add_argument("--motion-fps", type=float, default=2.0, help="FPS to sample for motion scoring")
    parser.add_argument("--motion-weight", type=float, default=0.4, help="Weight for motion score")
    parser.add_argument("--text-weight", type=float, default=0.6, help="Weight for transcript score")
    parser.add_argument(
        "--format",
        default="blur",
        choices=["blur", "center-crop"],
        help="TikTok format method",
    )
    parser.add_argument("--out-width", type=int, default=1080, help="Output width")
    parser.add_argument("--out-height", type=int, default=1920, help="Output height")
    parser.add_argument("--llm-model", default="gpt-5-mini", help="LLM model for transcript ranking")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM transcript ranking")
    parser.add_argument("--llm-batch-size", type=int, default=40, help="Segments per LLM request")
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM request timeout (s)")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Detailed logs → {log_path}")

    run_pipeline(
        video_path=video_path,
        out_dir=out_dir,
        num_clips=args.num_clips,
        clip_duration=args.clip_duration,
        min_gap=args.min_gap,
        prefer_early=args.prefer_early,
        early_half_life=args.early_half_life,
        whisper_model=args.whisper_model,
        device=args.device,
        motion_fps=args.motion_fps,
        motion_weight=args.motion_weight,
        text_weight=args.text_weight,
        format_method=args.format,
        out_width=args.out_width,
        out_height=args.out_height,
        llm_model=None if args.no_llm else args.llm_model,
        llm_batch_size=args.llm_batch_size,
        llm_timeout_s=args.llm_timeout,
    )


if __name__ == "__main__":
    main()
