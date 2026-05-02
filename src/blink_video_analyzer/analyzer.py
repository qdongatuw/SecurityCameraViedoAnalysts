from __future__ import annotations

import json
import os
from pathlib import Path

from .models import VideoAnalysis, VideoMetadata
from .video import image_to_data_url


SYSTEM_PROMPT = """You analyze short Blink security-camera motion clips from sampled frames.
Return only compact JSON. Identify what likely caused the motion: person, vehicle, animal,
tree/plant movement, lighting/shadow, precipitation, camera artifact, or unknown. If an animal is
visible, infer species as specifically as the image supports. Be honest about uncertainty."""


def build_user_prompt(metadata: VideoMetadata) -> str:
    duration = f"{metadata.duration_seconds:.1f}s" if metadata.duration_seconds else "unknown"
    return (
        "Analyze these sampled frames from one motion-triggered security clip.\n"
        f"Video file: {metadata.video_name}\n"
        f"File modified time used as clip time: {metadata.modified_time.isoformat()}\n"
        f"Approx duration: {duration}\n\n"
        "Return JSON with keys: motion_objects (array of strings), animal_species (array of strings), "
        "scene, description, confidence (0 to 1). Description should be one concise Chinese sentence."
    )


class VisionAnalyzer:
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = self._build_client() if os.getenv("OPENAI_API_KEY") else None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OPENAI_API_KEY is set, but the openai package is not installed. "
                "Run: pip install -e ."
            ) from exc
        return OpenAI()

    def analyze(self, metadata: VideoMetadata, frame_paths: list[Path]) -> VideoAnalysis:
        if not self.client:
            return VideoAnalysis(
                video_name=metadata.video_name,
                video_path=str(metadata.path),
                modified_time=metadata.modified_time,
                duration_seconds=metadata.duration_seconds,
                description="未设置 OPENAI_API_KEY；仅记录视频元数据，未进行视觉识别。",
                confidence=0.0,
                evidence_frames=[str(path) for path in frame_paths],
            )

        content: list[dict[str, object]] = [
            {"type": "input_text", "text": build_user_prompt(metadata)}
        ]
        for frame_path in frame_paths:
            content.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(frame_path),
                    "detail": "low",
                }
            )

        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
            temperature=0.1,
        )
        raw = response.output_text
        parsed = _parse_json(raw)

        return VideoAnalysis(
            video_name=metadata.video_name,
            video_path=str(metadata.path),
            modified_time=metadata.modified_time,
            duration_seconds=metadata.duration_seconds,
            motion_objects=_string_list(parsed.get("motion_objects")),
            animal_species=_string_list(parsed.get("animal_species")),
            scene=str(parsed.get("scene") or ""),
            description=str(parsed.get("description") or ""),
            confidence=_confidence(parsed.get("confidence")),
            evidence_frames=[str(path) for path in frame_paths],
            raw_model_output=raw,
        )


def _parse_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(line for line in lines if not line.strip().startswith("```"))
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"description": text, "confidence": 0.2}
    return value if isinstance(value, dict) else {"description": text, "confidence": 0.2}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
