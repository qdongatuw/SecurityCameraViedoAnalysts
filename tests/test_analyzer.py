from blink_video_analyzer.analyzer import _confidence, _parse_json, _string_list
from blink_video_analyzer.local_yolo import _label_to_cn


def test_parse_json_strips_code_fences() -> None:
    parsed = _parse_json('```json\n{"motion_objects":["car"],"confidence":0.8}\n```')

    assert parsed["motion_objects"] == ["car"]
    assert parsed["confidence"] == 0.8


def test_confidence_is_clamped() -> None:
    assert _confidence(1.5) == 1.0
    assert _confidence(-0.5) == 0.0
    assert _confidence("0.4") == 0.4


def test_string_list_filters_blank_items() -> None:
    assert _string_list(["cat", "", "  person  "]) == ["cat", "person"]


def test_local_yolo_label_translation() -> None:
    assert _label_to_cn("person") == "人"
    assert _label_to_cn("car") == "汽车"
