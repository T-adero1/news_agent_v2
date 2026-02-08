from pathlib import Path
import cv2
import numpy as np


def compute_motion_scores(video_path: Path, sample_fps: float = 2.0):
    """
    Returns a list of (time_sec, motion_score) sampled across the video.
    Motion score is mean absolute frame difference.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(fps / sample_fps)))

    scores = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(np.mean(diff))
            time_sec = float(frame_idx / fps)
            scores.append((time_sec, score))
        prev_gray = gray
        frame_idx += 1

    cap.release()
    return scores


def aggregate_motion_for_window(motion_scores, start: float, end: float):
    window = [s for t, s in motion_scores if start <= t <= end]
    if not window:
        return 0.0
    return float(np.mean(window))
