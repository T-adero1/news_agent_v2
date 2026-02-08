from pathlib import Path
import ffmpeg


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
        .output(str(out_path), vcodec="libx264", acodec="aac")
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
            acodec="aac",
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run(quiet=True)
    )
    return out_path


def _ffmpeg_safe_path(path: Path) -> str:
    """Return a path string safe for FFmpeg filter arguments on Windows.

    FFmpeg filter syntax treats colons and backslashes specially.
    Convert to forward slashes and escape colons (e.g. C: -> C\\:).
    """
    posix = str(path).replace("\\", "/")
    posix = posix.replace(":", "\\:")
    return posix


def burn_in_captions(video_path: Path, captions_path: Path, out_path: Path) -> Path:
    """Burn subtitle captions into video.

    For .ass files the ``ass`` filter is used so embedded styles (karaoke
    tags, colours, positioning) are preserved.  For .srt files the
    ``subtitles`` filter is used with a forced TikTok-style look.
    """
    if not captions_path.exists():
        raise RuntimeError(f"Caption file does not exist: {captions_path}")
    if captions_path.stat().st_size == 0:
        raise RuntimeError(f"Caption file is empty: {captions_path}")

    inp = ffmpeg.input(str(video_path))
    safe = _ffmpeg_safe_path(captions_path)

    if captions_path.suffix.lower() == ".ass":
        video = inp.video.filter("ass", safe)
    else:
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
        video = inp.video.filter("subtitles", safe, force_style=style)

    try:
        (
            ffmpeg
            .output(
                video,
                inp.audio,
                str(out_path),
                vcodec="libx264",
                acodec="aac",
                pix_fmt="yuv420p",
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else str(exc)
        raise RuntimeError(f"FFmpeg caption burn failed for {captions_path}: {stderr}") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg reported success but output missing/empty: {out_path}")
    return out_path

