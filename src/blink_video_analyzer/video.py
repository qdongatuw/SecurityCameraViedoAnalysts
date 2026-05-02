from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

from .models import VideoMetadata


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov"}


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    videos = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: path.stat().st_mtime)


def read_metadata(path: Path) -> VideoMetadata:
    import cv2

    modified_time = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or None
        frame_count_value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_count_value) if frame_count_value and frame_count_value > 0 else None
        duration = (frame_count / fps) if frame_count and fps else None
    finally:
        capture.release()

    return VideoMetadata(
        path=path,
        video_name=path.name,
        modified_time=modified_time,
        duration_seconds=duration,
        frame_count=frame_count,
        fps=fps,
    )


def extract_frames(path: Path, output_dir: Path, frame_count: int) -> list[Path]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return []

        if frame_count <= 1:
            indices = [total_frames // 2]
        else:
            step = max((total_frames - 1) / (frame_count - 1), 1)
            indices = sorted({min(total_frames - 1, round(i * step)) for i in range(frame_count)})

        frame_paths: list[Path] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            target = output_dir / f"{path.stem}_frame_{index:06d}.jpg"
            cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            frame_paths.append(target)
        return frame_paths
    finally:
        capture.release()


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
