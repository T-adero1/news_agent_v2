"""Smart face-tracking reframe for TikTok (9:16) output.

Two-pass approach:
  Pass 1 — Sample frames at low FPS, run MediaPipe face detection,
           detect scene cuts, build raw crop-centre trajectory.
  Pass 2 — Smooth trajectory with EMA, read every frame with OpenCV,
           crop + resize, pipe raw pixels to FFmpeg for H.264 encoding.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("tiktok_agent_v2.reframe")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_tiktok_smart_crop(
    video_path,
    out_path,
    width=1080,
    height=1920,
    *,
    analysis_fps: float = 8.0,
    smoothing_alpha: float = 0.08,
    scene_cut_threshold: float = 35.0,
    min_detection_confidence: float = 0.5,
    face_y_target: float = 0.33,
    fallback: str = "hold",
) -> Path:
    """Reframe a landscape video to 9:16 by tracking faces.

    Parameters
    ----------
    video_path : path-like
        Input video (any resolution / aspect).
    out_path : path-like
        Destination for the reframed MP4.
    width, height : int
        Output dimensions (default 1080x1920).
    analysis_fps : float
        How many frames per second to sample for face detection.
    smoothing_alpha : float
        EMA alpha for temporal smoothing (lower = smoother pan).
    scene_cut_threshold : float
        Mean-absolute-difference above which a scene cut is declared.
    min_detection_confidence : float
        MediaPipe face detection confidence threshold.
    face_y_target : float
        Target vertical position for face centre (0 = top, 1 = bottom).
    fallback : str
        What to do when no face is detected: ``"hold"`` last position
        or ``"center"`` the crop window.

    Returns
    -------
    Path
        The output file path.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    log.info("Smart-crop: %s  src=%dx%d  fps=%.1f  frames=%d",
             video_path.name, src_w, src_h, src_fps, total_frames)

    # Compute crop window size that fills width x height at source resolution
    out_aspect = width / height  # 0.5625 for 9:16
    crop_h = src_h
    crop_w = int(round(crop_h * out_aspect))
    if crop_w > src_w:
        # Source is narrower than target aspect — crop vertically instead
        crop_w = src_w
        crop_h = int(round(crop_w / out_aspect))

    log.info("Smart-crop: crop window = %dx%d (out=%dx%d)", crop_w, crop_h, width, height)

    # --- Pass 1: analyse keyframes for face positions -----------------------
    keyframe_positions = _analyse_faces(
        video_path,
        src_w, src_h,
        crop_w, crop_h,
        src_fps,
        total_frames,
        analysis_fps=analysis_fps,
        scene_cut_threshold=scene_cut_threshold,
        min_detection_confidence=min_detection_confidence,
        face_y_target=face_y_target,
        fallback=fallback,
    )

    # --- Smooth and interpolate to every frame ------------------------------
    frame_positions = _smooth_and_interpolate(
        keyframe_positions,
        total_frames,
        smoothing_alpha=smoothing_alpha,
        scene_cut_threshold=scene_cut_threshold,
    )

    # --- Pass 2: render -----------------------------------------------------
    _render(
        video_path, out_path,
        frame_positions,
        src_w, src_h,
        crop_w, crop_h,
        width, height,
        src_fps,
        total_frames,
    )

    log.info("Smart-crop: done → %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Pass 1 — face analysis
# ---------------------------------------------------------------------------

def _get_model_path() -> Path:
    """Return the path to the bundled face detection .tflite model."""
    return Path(__file__).resolve().parent / "blaze_face_short_range.tflite"


def _analyse_faces(
    video_path,
    src_w, src_h,
    crop_w, crop_h,
    src_fps,
    total_frames,
    *,
    analysis_fps,
    scene_cut_threshold,
    min_detection_confidence,
    face_y_target,
    fallback,
):
    """Return list of (frame_idx, cx, cy, is_scene_cut) for sampled keyframes."""
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceDetector,
            FaceDetectorOptions,
            RunningMode,
        )
    except ImportError:
        log.error(
            "mediapipe is not installed.  Install it with:  pip install mediapipe\n"
            "Falling back to centre-crop (no face tracking)."
        )
        interval = max(1, int(round(src_fps / analysis_fps)))
        positions = []
        for idx in range(0, total_frames, interval):
            positions.append((idx, src_w / 2, src_h / 2, False))
        return positions

    model_path = _get_model_path()
    if not model_path.exists():
        log.error("Face detection model not found: %s — falling back to centre crop", model_path)
        interval = max(1, int(round(src_fps / analysis_fps)))
        return [(idx, src_w / 2, src_h / 2, False) for idx in range(0, total_frames, interval)]

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        min_detection_confidence=min_detection_confidence,
        running_mode=RunningMode.IMAGE,
    )

    cap = cv2.VideoCapture(str(video_path))
    interval = max(1, int(round(src_fps / analysis_fps)))

    positions = []       # (frame_idx, cx, cy, is_scene_cut)
    prev_gray = None
    last_cx = src_w / 2
    last_cy = src_h / 2
    detections_found = 0
    scene_cuts = 0

    with FaceDetector.create_from_options(options) as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % interval != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Scene cut detection
            is_scene_cut = False
            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
                if diff > scene_cut_threshold:
                    is_scene_cut = True
                    scene_cuts += 1
            prev_gray = gray

            # Face detection — new Tasks API
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            cx, cy = _pick_face_centre(result, src_w, src_h)

            if cx is not None:
                detections_found += 1
                last_cx, last_cy = cx, cy
            else:
                if fallback == "center":
                    cx, cy = src_w / 2, src_h / 2
                else:
                    cx, cy = last_cx, last_cy

            # Shift cy toward face_y_target:
            # We want the face to sit at face_y_target fraction of the crop window.
            # crop centre-y = cy + crop_h*(0.5 - face_y_target)
            cy_adjusted = cy + crop_h * (0.5 - face_y_target)

            # Clamp to valid crop region
            cx_clamped = _clamp(cx, crop_w / 2, src_w - crop_w / 2)
            cy_clamped = _clamp(cy_adjusted, crop_h / 2, src_h - crop_h / 2)

            positions.append((frame_idx, cx_clamped, cy_clamped, is_scene_cut))
            frame_idx += 1

    cap.release()

    total_keyframes = len(positions)
    det_pct = (detections_found / total_keyframes * 100) if total_keyframes else 0
    log.info("Smart-crop pass 1: %d keyframes, %d faces (%.0f%%), %d scene cuts",
             total_keyframes, detections_found, det_pct, scene_cuts)

    if detections_found == 0:
        log.warning("Smart-crop: no faces detected in entire clip — using centre crop")

    return positions


def _pick_face_centre(result, src_w, src_h):
    """Pick the best face from MediaPipe Tasks result. Returns (cx, cy) or (None, None)."""
    if not result or not result.detections:
        return None, None

    faces = []
    for det in result.detections:
        bb = det.bounding_box
        # Tasks API returns pixel coordinates directly
        fw = bb.width
        fh = bb.height
        fcx = bb.origin_x + fw / 2
        fcy = bb.origin_y + fh / 2
        area = fw * fh
        faces.append((area, fcx, fcy))

    if not faces:
        return None, None

    faces.sort(key=lambda f: f[0], reverse=True)

    if len(faces) >= 2:
        a1, cx1, cy1 = faces[0]
        a2, cx2, cy2 = faces[1]
        # Two similar-sized faces close together → centre between them
        if a2 > a1 * 0.6:
            dist = abs(cx1 - cx2)
            if dist < src_w * 0.4:
                return (cx1 + cx2) / 2, (cy1 + cy2) / 2

    # Largest face wins
    return faces[0][1], faces[0][2]


# ---------------------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------------------

def _smooth_and_interpolate(keyframe_positions, total_frames, *, smoothing_alpha, scene_cut_threshold):
    """EMA-smooth keyframe positions and interpolate to every frame.

    Returns array of shape (total_frames, 2) with (cx, cy) per frame.
    """
    if not keyframe_positions:
        return np.zeros((total_frames, 2))

    kf_indices = np.array([p[0] for p in keyframe_positions], dtype=np.float64)
    kf_cx = np.array([p[1] for p in keyframe_positions], dtype=np.float64)
    kf_cy = np.array([p[2] for p in keyframe_positions], dtype=np.float64)
    kf_cuts = [p[3] for p in keyframe_positions]

    # EMA smooth (reset on scene cuts)
    smooth_cx = np.empty_like(kf_cx)
    smooth_cy = np.empty_like(kf_cy)
    smooth_cx[0] = kf_cx[0]
    smooth_cy[0] = kf_cy[0]

    for i in range(1, len(kf_cx)):
        if kf_cuts[i]:
            # Scene cut — snap instantly
            smooth_cx[i] = kf_cx[i]
            smooth_cy[i] = kf_cy[i]
        else:
            smooth_cx[i] = smoothing_alpha * kf_cx[i] + (1 - smoothing_alpha) * smooth_cx[i - 1]
            smooth_cy[i] = smoothing_alpha * kf_cy[i] + (1 - smoothing_alpha) * smooth_cy[i - 1]

    # Interpolate to every frame
    all_frames = np.arange(total_frames, dtype=np.float64)
    interp_cx = np.interp(all_frames, kf_indices, smooth_cx)
    interp_cy = np.interp(all_frames, kf_indices, smooth_cy)

    return np.column_stack([interp_cx, interp_cy])


# ---------------------------------------------------------------------------
# Pass 2 — render
# ---------------------------------------------------------------------------

def _render(
    video_path, out_path,
    frame_positions,
    src_w, src_h,
    crop_w, crop_h,
    out_w, out_h,
    src_fps,
    total_frames,
):
    """Read every frame, crop, resize, pipe to FFmpeg.

    Stderr is drained in a background thread to prevent the classic
    subprocess deadlock where FFmpeg blocks writing progress info to a
    full pipe buffer while we block writing pixels to stdin.
    """
    import threading
    import time
    t0 = time.monotonic()

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-loglevel", "warning",
        # Raw video from stdin
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", str(src_fps),
        "-i", "pipe:0",
        # Audio from original file (optional — won't fail if absent)
        "-i", str(video_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(out_path),
    ]

    log.info("Smart-crop pass 2: rendering %d frames → %s", total_frames, out_path.name)

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Drain stderr in a background thread so FFmpeg never blocks
    stderr_chunks = []

    def _drain_stderr():
        try:
            for line in proc.stderr:
                stderr_chunks.append(line)
        except Exception:
            pass

    drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
    drain_thread.start()

    cap = cv2.VideoCapture(str(video_path))
    frames_written = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(frame_positions):
            cx, cy = frame_positions[frame_idx]
        else:
            cx, cy = src_w / 2, src_h / 2

        # Compute top-left of crop window
        x1 = int(round(cx - crop_w / 2))
        y1 = int(round(cy - crop_h / 2))

        # Clamp (shouldn't be needed after upstream clamping, but be safe)
        x1 = max(0, min(x1, src_w - crop_w))
        y1 = max(0, min(y1, src_h - crop_h))

        cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]

        # Resize to output dimensions
        if cropped.shape[1] != out_w or cropped.shape[0] != out_h:
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cropped

        try:
            proc.stdin.write(resized.tobytes())
            frames_written += 1
        except (BrokenPipeError, OSError):
            log.error("Smart-crop: FFmpeg pipe broke at frame %d", frame_idx)
            break

    cap.release()

    try:
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    proc.wait()
    drain_thread.join(timeout=5)

    if proc.returncode != 0:
        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="ignore")
        for line in stderr_text.strip().split("\n")[-10:]:
            log.error("  ffmpeg: %s", line.rstrip())
        raise RuntimeError(f"Smart-crop FFmpeg render failed (rc={proc.returncode})")

    elapsed = time.monotonic() - t0
    log.info("Smart-crop pass 2: wrote %d frames in %.1fs (%.1f fps)",
             frames_written, elapsed, frames_written / max(elapsed, 0.001))

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Smart-crop produced empty output: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(val, lo, hi):
    """Clamp val to [lo, hi], handling degenerate case where lo > hi."""
    if lo > hi:
        return (lo + hi) / 2
    return max(lo, min(val, hi))
