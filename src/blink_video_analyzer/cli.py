from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import AnalyzeOptions, analyze_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch analyze Blink motion MP4 clips.")
    parser.add_argument("input_dir", type=Path, help="Directory containing video files.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    parser.add_argument(
        "--engine",
        choices=["local-yolo", "cloud-gpt"],
        default="local-yolo",
        help="Analysis engine.",
    )
    parser.add_argument("--frames", type=int, default=5, help="Sampled frames per video.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N videos.")
    parser.add_argument(
        "--model",
        default=os.getenv("BLINK_ANALYZER_MODEL", "gpt-4.1-mini"),
        help="Cloud GPT vision model name.",
    )
    parser.add_argument(
        "--local-model",
        default="yolov8n.pt",
        help="Local YOLO model path or model name.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Local YOLO confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Local inference device: auto, cpu, 0, 1, etc.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sidecars.")
    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser()
    output_dir = (args.output.expanduser() if args.output else input_dir / "analysis")

    def progress(done: int, total: int, video_name: str) -> None:
        print(f"[{done}/{total}] {video_name}")

    results = analyze_directory(
        AnalyzeOptions(
            input_dir=input_dir,
            output_dir=output_dir,
            engine=args.engine,
            recursive=args.recursive,
            frame_count=args.frames,
            limit=args.limit,
            model=args.model,
            local_model=args.local_model,
            confidence_threshold=args.confidence,
            device=args.device,
            overwrite=args.overwrite,
        ),
        progress_callback=progress,
    )
    print(f"Done. Analyzed {len(results)} videos. Output: {output_dir}")


if __name__ == "__main__":
    main()
