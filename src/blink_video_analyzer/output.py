from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import VideoAnalysis


def analysis_stem(video_name: str) -> str:
    return Path(video_name).stem


def write_sidecars(result: VideoAnalysis, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = analysis_stem(result.video_name)
    json_path = output_dir / f"{stem}.analysis.json"
    md_path = output_dir / f"{stem}.analysis.md"

    json_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )

    md_path.write_text(_markdown(result), encoding="utf-8")
    return json_path, md_path


def write_summary(results: list[VideoAnalysis], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    jsonl_path = output_dir / "summary.jsonl"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_name",
                "modified_time",
                "duration_seconds",
                "motion_objects",
                "animal_species",
                "scene",
                "description",
                "confidence",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "video_name": result.video_name,
                    "modified_time": result.modified_time.isoformat(),
                    "duration_seconds": result.duration_seconds,
                    "motion_objects": "; ".join(result.motion_objects),
                    "animal_species": "; ".join(result.animal_species),
                    "scene": result.scene,
                    "description": result.description,
                    "confidence": result.confidence,
                    "error": result.error,
                }
            )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(result.model_dump_json() + "\n")


def sidecar_exists(video_name: str, output_dir: Path) -> bool:
    return (output_dir / f"{analysis_stem(video_name)}.analysis.json").exists()


def _markdown(result: VideoAnalysis) -> str:
    objects = ", ".join(result.motion_objects) if result.motion_objects else "unknown"
    animals = ", ".join(result.animal_species) if result.animal_species else "none"
    return (
        f"# {result.video_name}\n\n"
        f"- Video path: `{result.video_path}`\n"
        f"- Date/time: {result.modified_time.isoformat()}\n"
        f"- Duration: {result.duration_seconds or 'unknown'} seconds\n"
        f"- Motion objects: {objects}\n"
        f"- Animal species: {animals}\n"
        f"- Scene: {result.scene or 'unknown'}\n"
        f"- Confidence: {result.confidence:.2f}\n\n"
        f"{result.description}\n"
    )

