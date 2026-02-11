import argparse
import atexit
import faulthandler
import logging
import os
import signal
import sys
import threading
from pathlib import Path

from tiktok_agent_v2.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 45s clips with LLM ranking and captions")
    parser.add_argument("--video", required=True, help="Path to input video")
    return parser.parse_args()


def _flush_all_log_handlers():
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _install_runtime_logging_guards(log_path: Path, fault_path: Path):
    log = logging.getLogger("tiktok_agent_v2.main")

    try:
        fault_file = open(fault_path, "a", encoding="utf-8")
        faulthandler.enable(file=fault_file, all_threads=True)
    except Exception:
        log.exception("Failed to enable faulthandler")

    def _log_uncaught(exc_type, exc_value, exc_traceback):
        log.critical("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_value, exc_traceback))
        _flush_all_log_handlers()

    def _log_thread_exception(args):
        log.critical(
            "UNCAUGHT THREAD EXCEPTION in %s",
            getattr(args.thread, "name", "<unknown>"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        _flush_all_log_handlers()

    def _log_unraisable(unraisable):
        log.critical(
            "UNRAISABLE EXCEPTION: %s",
            getattr(unraisable, "err_msg", "<no message>"),
            exc_info=(type(unraisable.exc_value), unraisable.exc_value, unraisable.exc_traceback),
        )
        _flush_all_log_handlers()

    def _signal_handler(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = "UNKNOWN"
        log.critical("RECEIVED SIGNAL %s (%s) - terminating", signum, signame)
        _flush_all_log_handlers()
        raise SystemExit(128 + signum)

    sys.excepthook = _log_uncaught
    threading.excepthook = _log_thread_exception
    sys.unraisablehook = _log_unraisable

    for signame in ("SIGTERM", "SIGINT", "SIGBREAK"):
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), _signal_handler)

    atexit.register(_flush_all_log_handlers)
    log.info("Runtime logging guards enabled")
    log.info("Main log: %s", log_path)
    log.info("Fault log: %s", fault_path)


def main():
    args = parse_args()
    video_path = Path(args.video).expanduser()
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set. This script is hardcoded to use gpt-5-mini.")

    log_path = out_dir / "pipeline.log"
    fault_path = out_dir / "fault.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("tiktok_agent_v2").setLevel(logging.INFO)
    print(f"Detailed logs -> {log_path}")
    _install_runtime_logging_guards(log_path=log_path, fault_path=fault_path)

    run_pipeline(
        video_path=video_path,
        out_dir=out_dir,
        num_clips=1,
        clip_duration=45.0,
        min_gap=6.0,
        prefer_early=False,
        early_half_life=45.0,
        whisper_model="small",
        device="cpu",
        motion_fps=2.0,
        motion_weight=0.4,
        text_weight=0.6,
        format_method="smart-crop",
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
