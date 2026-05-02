from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .models import VideoAnalysis, VideoMetadata


VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "boat"}
ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}
IMPORTANT_CLASSES = {"person"} | VEHICLE_CLASSES | ANIMAL_CLASSES

CN_LABELS = {
    "person": "人",
    "bicycle": "自行车",
    "car": "汽车",
    "motorcycle": "摩托车",
    "bus": "公交车",
    "train": "火车",
    "truck": "卡车",
    "boat": "船",
    "bird": "鸟",
    "cat": "猫",
    "dog": "狗",
    "horse": "马",
    "sheep": "羊",
    "cow": "牛",
    "elephant": "象",
    "bear": "熊",
    "zebra": "斑马",
    "giraffe": "长颈鹿",
    "tree_or_light_motion": "树枝/光影变化",
    "unknown_motion": "未知运动",
}


class LocalYoloAnalyzer:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        device: str = "auto",
    ) -> None:
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.device = _resolve_device(device)
        self.model = self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Local YOLO mode requires ultralytics. Run: pip install -e ."
            ) from exc
        return YOLO(self.model_name)

    def analyze(self, metadata: VideoMetadata, frame_paths: list[Path]) -> VideoAnalysis:
        detections: list[dict[str, object]] = []
        class_counts: Counter[str] = Counter()
        best_confidence = 0.0

        for frame_path in frame_paths:
            results = self.model.predict(
                source=str(frame_path),
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )
            if not results:
                continue
            result = results[0]
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = str(names.get(class_id, class_id))
                if label not in IMPORTANT_CLASSES:
                    continue
                class_counts[label] += 1
                best_confidence = max(best_confidence, confidence)
                detections.append(
                    {
                        "frame": str(frame_path),
                        "label": label,
                        "confidence": round(confidence, 4),
                    }
                )

        labels = [label for label, _ in class_counts.most_common()]
        motion_objects = [_label_to_cn(label) for label in labels]
        animal_species = [_label_to_cn(label) for label in labels if label in ANIMAL_CLASSES]

        if not motion_objects:
            fallback = _motion_fallback(frame_paths)
            motion_objects = [_label_to_cn(fallback)]
            best_confidence = 0.35 if fallback == "tree_or_light_motion" else 0.15

        scene = _build_scene(labels)
        description = _build_description(metadata, motion_objects, animal_species, best_confidence)

        return VideoAnalysis(
            video_name=metadata.video_name,
            video_path=str(metadata.path),
            modified_time=metadata.modified_time,
            duration_seconds=metadata.duration_seconds,
            motion_objects=motion_objects,
            animal_species=animal_species,
            scene=scene,
            description=description,
            confidence=round(best_confidence, 3),
            evidence_frames=[str(path) for path in frame_paths],
            raw_model_output=json.dumps(
                {
                    "engine": "local-yolo",
                    "model": self.model_name,
                    "device": self.device,
                    "detections": detections,
                },
                ensure_ascii=False,
            ),
        )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "0" if torch.cuda.is_available() else "cpu"


def describe_compute_device(device: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed; local YOLO cannot run yet."

    if device == "cpu":
        return "CPU selected."

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if device == "auto":
            return f"CUDA available: {name}. Auto will use GPU 0."
        return f"CUDA available: using GPU {device}."

    if device == "auto":
        return "CUDA is not available in this Python environment. Auto will use CPU."
    return f"GPU {device} was selected, but CUDA is not available in this Python environment."


def _label_to_cn(label: str) -> str:
    return CN_LABELS.get(label, label)


def _build_scene(labels: list[str]) -> str:
    if "person" in labels:
        return "检测到人员活动"
    if any(label in VEHICLE_CLASSES for label in labels):
        return "检测到车辆活动"
    if any(label in ANIMAL_CLASSES for label in labels):
        return "检测到动物活动"
    return "未检测到明确目标，可能是环境运动"


def _build_description(
    metadata: VideoMetadata,
    motion_objects: list[str],
    animal_species: list[str],
    confidence: float,
) -> str:
    objects = "、".join(motion_objects)
    animal_note = f"，疑似动物种类：{'、'.join(animal_species)}" if animal_species else ""
    return (
        f"{metadata.modified_time:%Y-%m-%d %H:%M:%S}，视频中检测到的运动目标为："
        f"{objects}{animal_note}，置信度约 {confidence:.2f}。"
    )


def _motion_fallback(frame_paths: list[Path]) -> str:
    if len(frame_paths) < 2:
        return "unknown_motion"

    try:
        import cv2
    except ImportError:
        return "unknown_motion"

    previous = None
    changed_ratios: list[float] = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (160, 90))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        if previous is not None:
            diff = cv2.absdiff(previous, image)
            _, threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            changed = cv2.countNonZero(threshold) / float(threshold.size)
            changed_ratios.append(changed)
        previous = image

    if not changed_ratios:
        return "unknown_motion"
    average_change = sum(changed_ratios) / len(changed_ratios)
    return "tree_or_light_motion" if average_change >= 0.015 else "unknown_motion"
