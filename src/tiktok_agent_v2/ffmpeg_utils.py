from pathlib import Path
import logging
import subprocess
import ffmpeg

log = logging.getLogger("tiktok_agent_v2.ffmpeg_utils")


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    (
        ffmpeg
        .input(str(video_path))
        .output(str(wav_path), ac=1, ar=16000)
        .overwrite_output()
        .run(quiet=True)
    )
    return wav_path


def clip_video(video_path: Path, start: float, end: float, out_path: Path) -> Path:
    (
        ffmpeg
        .input(str(video_path), ss=start, t=max(0.1, end - start))
        .output(str(out_path), vcodec="copy", acodec="copy")
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


def format_tiktok_blur(video_path: Path, out_path: Path, width: int = 1080, height: int = 1920) -> Path:
    """
    Fit original video inside 9:16 and fill background with a blurred copy.
    Keeps the original audio track.
    """
    inp = ffmpeg.input(str(video_path))
    audio = inp.audio

    # Split video so we can use it for both bg and fg
    split = inp.video.filter("setsar", 1).filter_multi_output("split")

    # Background: fill frame, crop to exact size, then blur
    bg = (
        split[0]
        .filter("scale", width, height, force_original_aspect_ratio="increase")
        .filter("crop", width, height)
        .filter("boxblur", luma_radius=20, luma_power=1)
    )
    # Foreground: fit inside frame without cropping
    fg = (
        split[1]
        .filter("scale", width, height, force_original_aspect_ratio="decrease")
        .filter("setsar", 1)
    )

    composed = ffmpeg.overlay(bg, fg, x="(W-w)/2", y="(H-h)/2")

    (
        ffmpeg
        .output(
            composed,
            audio,
            str(out_path),
            vcodec="libx264",
            preset="ultrafast",
            acodec="aac",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


def format_tiktok_center_crop(video_path: Path, out_path: Path, width: int = 1080, height: int = 1920) -> Path:
    """
    Center-crop to 9:16 (may cut content).
    """
    inp = ffmpeg.input(str(video_path))
    video = (
        inp.video
        .filter("scale", width, height, force_original_aspect_ratio="increase")
        .filter("crop", width, height)
        .filter("setsar", 1)
    )
    (
        ffmpeg
        .output(
            video,
            inp.audio,
            str(out_path),
            vcodec="libx264",
            preset="ultrafast",
            acodec="aac",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


def _ffmpeg_safe_path(path: Path) -> str:
    """Return a path string safe for FFmpeg filter arguments.

    FFmpeg filter syntax treats colons, backslashes and single-quotes specially.
    We resolve to an absolute path, convert to forward slashes, and escape
    colons and single-quotes so the path is safe *without* additional quoting.
    """
    posix = str(path.resolve()).replace("\\", "/")
    posix = posix.replace(":", "\\:")
    posix = posix.replace("'", "\\'")
    return posix


def burn_in_captions(video_path: Path, captions_path: Path, out_path: Path) -> Path:
    """Burn subtitle captions into video using subprocess.

    Uses subprocess instead of ffmpeg-python because the latter over-escapes
    Windows paths (colons, backslashes) in filter arguments, causing libass
    to silently fail to open the file.

    For .ass files the ``ass`` filter is used so embedded styles (karaoke
    tags, colours, positioning) are preserved.  For .srt files the
    ``subtitles`` filter is used with a forced TikTok-style look.
    """
    if not captions_path.exists():
        raise RuntimeError(f"Caption file does not exist: {captions_path}")
    if captions_path.stat().st_size == 0:
        raise RuntimeError(f"Caption file is empty: {captions_path}")

    log.info(
        "Burning captions: video=%s captions=%s out=%s",
        video_path,
        captions_path,
        out_path,
    )

    safe = _ffmpeg_safe_path(captions_path)

    # Try to locate a fonts directory for libass.
    # If a bundled fonts/ dir exists next to the package, use it.
    # Otherwise try the system fonts directory.
    fonts_dir = None
    _bundled_fonts = Path(__file__).resolve().parent / "fonts"
    if _bundled_fonts.is_dir():
        fonts_dir = _bundled_fonts
    else:
        for p in [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]:
            if p.is_dir():
                fonts_dir = p
                break
    if fonts_dir:
        log.info("Using fontsdir: %s", fonts_dir)

    if captions_path.suffix.lower() == ".ass":
        log.info("Caption renderer mode: ASS")
        if fonts_dir:
            safe_fonts = _ffmpeg_safe_path(fonts_dir)
            vf = f"ass={safe}:fontsdir={safe_fonts}"
        else:
            vf = f"ass={safe}"
    else:
        log.info("Caption renderer mode: SRT/subtitles")
        style = (
            "FontName=Arial,"
            "FontSize=14,"
            "PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,"
            "BorderStyle=1,"
            "Outline=3,"
            "Shadow=0,"
            "Bold=1,"
            "Alignment=2,"
            "MarginV=220"
        )
        vf = f"subtitles={safe}:force_style='{style}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path.resolve()),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(out_path.resolve()),
    ]
    log.info("FFmpeg command: %s", cmd)

    input_size = video_path.stat().st_size

    result = subprocess.run(cmd, capture_output=True)
    stderr_text = result.stderr.decode("utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg caption burn failed (rc={result.returncode}): {stderr_text}")

    # Log stderr even on success — libass warnings (font not found, etc.) appear here
    if stderr_text:
        lines = stderr_text.strip().split("\n")
        # Always log warnings about fonts/libass (they appear early, before progress)
        for line in lines:
            ll = line.lower()
            if any(kw in ll for kw in ["libass", "fontconfig", "font", "glyph", "fontsdir"]):
                log.warning("  ffmpeg (font): %s", line.rstrip())
        # Also log the last 15 lines (encoding summary)
        for line in lines[-15:]:
            log.info("  ffmpeg: %s", line.rstrip())

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg reported success but output missing/empty: {out_path}")

    output_size = out_path.stat().st_size
    log.info("Caption burn complete: %s (%d bytes, input was %d bytes)", out_path, output_size, input_size)

    # Warn if output is suspiciously close to input size (captions may not have rendered)
    if input_size > 0 and abs(output_size - input_size) / input_size < 0.01:
        log.warning("Caption output size ~= input size — captions may not have rendered!")

    return out_path

