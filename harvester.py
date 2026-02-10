"""
harvester.py — Automated news video harvesting pipeline.

Searches Bing News RSS for trending stories, finds embedded videos,
runs the TikTok clip pipeline, and delivers finished clips to Telegram.

Run:  python harvester.py

Required env vars in .env:
    OPENAI_API_KEY
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
"""

import atexit
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure ffmpeg.exe in project root is found by subprocess calls
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _project_root + os.pathsep + os.environ.get("PATH", "")

# Fix OpenMP DLL conflict between faster-whisper and numpy on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tiktok_agent_v2.pipeline import run_pipeline

log = logging.getLogger("harvester")

LOCK_FILE = Path("harvester.lock")
HISTORY_FILE = Path("harvester_history.json")
TMP_DIR = Path("harvester_tmp")

# ── Hardcoded settings ───────────────────────────────────────────────────────
NUM_STORIES = 25          # max stories to consider from Bing News
NUM_VIDEOS = 3            # stop after successfully processing this many
NUM_CLIPS_PER_VIDEO = 1   # clips to extract from each video
TOPICS = (
    "Houston car accident,"
    "Houston shooting,"
    "Houston hit and run,"
    "Houston crash injury,"
    "Houston fatal crash,"
    "Houston pedestrian struck,"
    "Houston crime,"
    "Houston police chase,"
    "Houston fire,"
    "Houston stabbing"
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    log_path = TMP_DIR / "harvester.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [harvester] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.getLogger("tiktok_agent_v2").setLevel(logging.INFO)
    log.info("Logging to %s", log_path)


def check_env():
    """Fail fast if required env vars are missing."""
    missing = []
    for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)


# ---------------------------------------------------------------------------
# File Lock
# ---------------------------------------------------------------------------

def acquire_lock():
    """Prevent overlapping cron runs. Steals stale locks from dead PIDs."""
    if LOCK_FILE.exists():
        try:
            old_pid = int(LOCK_FILE.read_text().strip())
            if sys.platform == "win32":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {old_pid}", "/NH"],
                    capture_output=True, text=True,
                )
                alive = str(old_pid) in result.stdout
            else:
                try:
                    os.kill(old_pid, 0)
                    alive = True
                except OSError:
                    alive = False

            if alive:
                log.info("Another harvester is running (PID %d). Exiting.", old_pid)
                sys.exit(0)
            else:
                log.info("Stale lock (PID %d dead). Stealing.", old_pid)
        except (ValueError, OSError):
            log.info("Corrupt lock file. Overwriting.")

    LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(release_lock)
    log.info("Lock acquired (PID %d)", os.getpid())


def release_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Deduplication History
# ---------------------------------------------------------------------------

def load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {"version": 1, "processed": {}}
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        if "processed" not in data:
            data["processed"] = {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not load history (%s), starting fresh", exc)
        return {"version": 1, "processed": {}}


def save_history(history: dict):
    """Atomic write via tmp + rename so crashes don't corrupt history."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(TMP_DIR), suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        Path(tmp_path).rename(HISTORY_FILE)
    except OSError:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def record_story(history: dict, url: str, title: str, clips_produced: int, sent_to_telegram: bool):
    history["processed"][url] = {
        "title": title,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "clips_produced": clips_produced,
        "sent_to_telegram": sent_to_telegram,
    }
    save_history(history)


# ---------------------------------------------------------------------------
# Bing News RSS Search (free, no API key)
# ---------------------------------------------------------------------------

def search_news_stories() -> list:
    """Search Bing News RSS for recent stories matching our topics.

    Queries each topic separately and merges results, deduped by URL.
    Returns list of dicts: [{title, url, summary}, ...]
    """
    topics = [t.strip() for t in TOPICS.split(",")]
    seen_urls = set()
    all_stories = []

    for topic in topics:
        query = quote(topic)
        rss_url = f"https://www.bing.com/news/search?q={query}&format=rss"
        log.info("Fetching Bing News RSS: %s", topic)

        try:
            resp = requests.get(rss_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except Exception as exc:
            log.warning("RSS fetch failed for '%s': %s", topic, exc)
            continue

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            log.warning("RSS parse failed for '%s': %s", topic, exc)
            continue

        for item in root.iter("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            if title_el is None or link_el is None:
                continue

            title = title_el.text or ""
            raw_url = link_el.text or ""
            if not raw_url:
                continue

            # Bing wraps real URLs in a redirect — extract from ?url= param
            url = _extract_bing_url(raw_url)
            if not url or url in seen_urls:
                continue

            seen_urls.add(url)
            all_stories.append({"title": title, "url": url, "summary": ""})

    log.info("Bing News returned %d unique stories from RSS", len(all_stories))

    # Also search YouTube for Houston TV news clips
    yt_stories = _search_youtube_news()
    for yt in yt_stories:
        if yt["url"] not in seen_urls:
            seen_urls.add(yt["url"])
            all_stories.append(yt)

    log.info("Total stories: %d (RSS + YouTube)", len(all_stories))

    # Prioritize: YouTube first (guaranteed video), then TV news stations
    tv_domains = ("click2houston.com", "khou.com", "abc13.com", "fox26houston.com",
                  "cw39.com", "kprc.com", "houstonchronicle.com", "msn.com")
    def _priority(story):
        domain = urlparse(story["url"]).netloc.lower()
        if "youtube.com" in domain or "youtu.be" in domain:
            return 0
        if any(d in domain for d in tv_domains):
            return 1
        return 2
    all_stories.sort(key=_priority)

    for i, s in enumerate(all_stories[:NUM_STORIES]):
        log.info("  [%d] %s — %s", i + 1, s["title"][:80], s["url"][:80])

    return all_stories[:NUM_STORIES]


def _search_youtube_news() -> list:
    """Search YouTube for recent Houston news clips from local TV stations."""
    queries = [
        "Houston car accident crash news",
        "Houston local news today",
        "Houston trending news"
    ]
    results = []
    seen = set()

    for query in queries:
        log.info("Searching YouTube: %s", query)
        cmd = [
            *YTDLP_CMD, f"ytsearch5:{query}",
            "--dump-json", "--no-download", "--flat-playlist",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

        if result.returncode != 0:
            continue

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                info = json.loads(line)
            except json.JSONDecodeError:
                continue

            vid_id = info.get("id", "")
            if not vid_id or vid_id in seen:
                continue
            seen.add(vid_id)

            duration = info.get("duration") or 0
            if not (60 <= duration <= 900):
                continue

            url = info.get("url") or f"https://www.youtube.com/watch?v={vid_id}"
            title = info.get("title") or "Untitled"
            results.append({"title": title, "url": url, "summary": "YouTube"})
            log.info("  YT: %s (%.0fs)", title[:80], duration)

    log.info("YouTube search found %d suitable videos", len(results))
    return results


def _extract_bing_url(bing_url: str) -> str:
    """Extract the real article URL from a Bing News redirect link."""
    try:
        parsed = urlparse(bing_url)
        params = parse_qs(parsed.query)
        real = params.get("url", [""])[0]
        if real:
            return real
    except Exception:
        pass
    return bing_url


# ---------------------------------------------------------------------------
# yt-dlp helper
# ---------------------------------------------------------------------------

YTDLP_CMD = [sys.executable, "-m", "yt_dlp"]


# ---------------------------------------------------------------------------
# Video Discovery (yt-dlp probe)
# ---------------------------------------------------------------------------

def probe_page_for_videos(url: str) -> list:
    """Run yt-dlp --dump-json to find embedded videos on a page.

    Returns list of dicts: [{url, title, duration}, ...] filtered to 1-15 min.
    """
    log.info("Probing for videos: %s", url)

    cmd = [*YTDLP_CMD, "--dump-json", "--no-download", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp probe timed out for %s", url)
        return []
    except FileNotFoundError:
        log.error("yt-dlp not found. Install it: pip install yt-dlp")
        return []

    if result.returncode != 0:
        stderr_short = (result.stderr or "")[:300]
        log.info("No videos at %s (rc=%d): %s", url, result.returncode, stderr_short)
        return []

    videos = []
    skipped = 0
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
        except json.JSONDecodeError:
            continue

        duration = info.get("duration") or 0
        vid_url = info.get("webpage_url") or info.get("url") or url
        vid_title = info.get("title") or "Untitled"

        if 60 <= duration <= 900:
            videos.append({"url": vid_url, "title": vid_title, "duration": duration})
            log.info("  Found: %s (%.0fs)", vid_title[:80], duration)
        else:
            skipped += 1
            if skipped <= 3:
                log.info("  Skip (%.0fs, need 60-900s): %s", duration, vid_title[:80])

    if skipped > 3:
        log.info("  ... and %d more skipped (wrong duration)", skipped - 3)

    if not videos:
        log.info("  No suitable videos (need 1-15 min)")

    return videos


# ---------------------------------------------------------------------------
# Video Download
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Path) -> Path | None:
    """Download video via yt-dlp. Returns path to MP4 or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading: %s", url)

    cmd = [
        *YTDLP_CMD,
        "-f", "bestvideo[ext=mp4]+bestaudio/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--restrict-filenames",
        "--print", "after_move:filepath",
        "-o", str(output_dir / "%(title).80s_%(id)s.%(ext)s"),
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log.error("Download timed out (10 min) for %s", url)
        return None
    except FileNotFoundError:
        log.error("yt-dlp not found")
        return None

    if result.returncode != 0:
        log.error("Download failed (rc=%d): %s", result.returncode, (result.stderr or "")[:500])
        return None

    filepath = result.stdout.strip().split("\n")[-1].strip()
    if not filepath or not Path(filepath).exists():
        mp4s = list(output_dir.glob("*.mp4"))
        if mp4s:
            filepath = str(mp4s[0])
        else:
            log.error("Download produced no MP4 file")
            return None

    downloaded = Path(filepath)
    log.info("Downloaded: %s (%.1f MB)", downloaded.name, downloaded.stat().st_size / (1024 * 1024))
    return downloaded


# ---------------------------------------------------------------------------
# Pipeline Integration
# ---------------------------------------------------------------------------

def process_video(video_path: Path, clip_dir: Path) -> list:
    """Run the clip pipeline. Returns list of final clip Paths."""
    clip_dir.mkdir(parents=True, exist_ok=True)
    log.info("Running pipeline: %s -> %s", video_path.name, clip_dir)

    try:
        run_pipeline(
            video_path=video_path,
            out_dir=clip_dir,
            num_clips=NUM_CLIPS_PER_VIDEO,
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
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        return []

    clips = []
    for mp4 in sorted(clip_dir.glob("clip_*.mp4")):
        if not mp4.stem.endswith("_raw") and not mp4.stem.endswith("_captioned"):
            clips.append(mp4)

    log.info("Pipeline produced %d clip(s)", len(clips))
    return clips


# ---------------------------------------------------------------------------
# Telegram Delivery
# ---------------------------------------------------------------------------

def send_to_telegram(clip_path: Path, story_title: str, story_url: str) -> bool:
    """Send a video clip to Telegram."""
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    domain = urlparse(story_url).netloc or story_url
    caption = f"<b>{story_title}</b>\n\nSource: {domain}"
    if len(caption) > 1024:
        caption = caption[:1021] + "..."

    file_size = clip_path.stat().st_size
    log.info("Sending to Telegram: %s (%.1f MB)", clip_path.name, file_size / (1024 * 1024))

    if file_size < 45 * 1024 * 1024:
        endpoint = f"https://api.telegram.org/bot{token}/sendVideo"
        field = "video"
    else:
        endpoint = f"https://api.telegram.org/bot{token}/sendDocument"
        field = "document"
        log.info("File > 45MB, using sendDocument fallback")

    try:
        with open(clip_path, "rb") as f:
            resp = requests.post(
                endpoint,
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                files={field: (clip_path.name, f, "video/mp4")},
                timeout=120,
            )
        resp.raise_for_status()
        result = resp.json()
        if result.get("ok"):
            log.info("Telegram send succeeded")
            return True
        else:
            log.error("Telegram error: %s", result.get("description", "unknown"))
            return False
    except Exception as exc:
        log.error("Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_harvester():
    """Find 3 stories, download 3 videos, make 3 clips, send to Telegram."""
    t0 = time.monotonic()
    history = load_history()

    # Search
    stories = search_news_stories()
    if not stories:
        log.info("No stories found. Done.")
        return 0

    # Dedup
    new_stories = []
    for story in stories:
        url = story.get("url", "")
        if url in history["processed"]:
            log.info("Already processed: %s", url)
        else:
            new_stories.append(story)

    if not new_stories:
        log.info("All stories already processed. Done.")
        return 0

    log.info("%d new stories available, will process up to %d", len(new_stories), NUM_VIDEOS)

    stats = {"processed": 0, "failed": 0, "sent": 0}

    for story_num, story in enumerate(new_stories, 1):
        if stats["processed"] >= NUM_VIDEOS:
            break

        url = story.get("url", "")
        title = story.get("title", "Untitled")
        log.info("=== STORY %d: %s ===", story_num, title)
        log.info("  URL: %s", url)

        try:
            # Probe for embedded video
            videos = probe_page_for_videos(url)
            if not videos:
                log.info("No video found, skipping.")
                record_story(history, url, title, clips_produced=0, sent_to_telegram=False)
                continue

            # Download first suitable video
            story_dir = TMP_DIR / f"story_{story_num}"
            downloaded = download_video(videos[0]["url"], story_dir)
            if not downloaded:
                log.error("Download failed, skipping.")
                record_story(history, url, title, clips_produced=0, sent_to_telegram=False)
                stats["failed"] += 1
                continue

            # Run clip pipeline
            clip_dir = story_dir / "clips"
            clips = process_video(downloaded, clip_dir)
            if not clips:
                log.error("Pipeline produced no clips, skipping.")
                record_story(history, url, title, clips_produced=0, sent_to_telegram=False)
                stats["failed"] += 1
                continue

            # Send to Telegram
            sent = False
            for clip in clips:
                if send_to_telegram(clip, title, url):
                    sent = True
                    stats["sent"] += 1

            # Record and cleanup
            record_story(history, url, title, clips_produced=len(clips), sent_to_telegram=sent)
            stats["processed"] += 1

            if downloaded.exists():
                downloaded.unlink()

        except Exception as exc:
            log.error("Story failed: %s", exc, exc_info=True)
            record_story(history, url, title, clips_produced=0, sent_to_telegram=False)
            stats["failed"] += 1

    elapsed = time.monotonic() - t0
    log.info("=== DONE (%.0fs) === Processed: %d | Failed: %d | Sent: %d",
             elapsed, stats["processed"], stats["failed"], stats["sent"])

    if stats["processed"] == 0 and stats["failed"] > 0:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    log.info("=== HARVESTER START ===")
    check_env()
    acquire_lock()
    sys.exit(run_harvester())


if __name__ == "__main__":
    main()
