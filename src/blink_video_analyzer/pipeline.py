from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .analyzer import VisionAnalyzer
from .local_yolo import LocalYoloAnalyzer
from .models import VideoAnalysis
from .output import sidecar_exists, write_sidecars, write_summary
from .video import extract_frames, find_videos, read_metadata


ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class AnalyzeOptions:
    input_dir: Path
    output_dir: Path
    engine: str = "local-yolo"
    recursive: bool = True
    frame_count: int = 5
    limit: int | None = None
    model: str = "gpt-4.1-mini"
    local_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.25
    device: str = "auto"
    overwrite: bool = False


def analyze_directory(
    options: AnalyzeOptions,
    progress_callback: ProgressCallback | None = None,
) -> list[VideoAnalysis]:
    videos = find_videos(options.input_dir, options.recursive)
    if options.limit:
        videos = videos[: options.limit]

    options.output_dir.mkdir(parents=True, exist_ok=True)
    frames_root = options.output_dir / "frames"
    if progress_callback:
        progress_callback(0, len(videos), "Loading analyzer")
    analyzer = _build_analyzer(options)
    results: list[VideoAnalysis] = []

    total = len(videos)
    for index, video_path in enumerate(videos, start=1):
        if progress_callback:
            progress_callback(index - 1, total, f"Analyzing {video_path.name}")

        metadata = read_metadata(video_path)
        if sidecar_exists(metadata.video_name, options.output_dir) and not options.overwrite:
            if progress_callback:
                progress_callback(index, total, f"Skipped {video_path.name}")
            continue

        frame_dir = frames_root / video_path.stem
        try:
            frame_paths = extract_frames(video_path, frame_dir, options.frame_count)
            result = analyzer.analyze(metadata, frame_paths)
        except Exception as exc:
            result = VideoAnalysis(
                video_name=metadata.video_name,
                video_path=str(metadata.path),
                modified_time=metadata.modified_time,
                duration_seconds=metadata.duration_seconds,
                description="分析失败。",
                confidence=0.0,
                error=str(exc),
            )

        write_sidecars(result, options.output_dir)
        results.append(result)
        if progress_callback:
            progress_callback(index, total, f"Finished {video_path.name}")

    write_summary(results, options.output_dir)
    return results


def _build_analyzer(options: AnalyzeOptions):
    if options.engine == "cloud-gpt":
        return VisionAnalyzer(options.model)
    if options.engine == "local-yolo":
        return LocalYoloAnalyzer(
            model=options.local_model,
            confidence_threshold=options.confidence_threshold,
            device=options.device,
        )
    raise ValueError(f"Unknown analysis engine: {options.engine}")
