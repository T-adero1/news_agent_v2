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

