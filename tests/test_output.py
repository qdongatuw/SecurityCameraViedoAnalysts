from blink_video_analyzer.output import analysis_stem


def test_analysis_stem_uses_video_name_without_extension() -> None:
    assert analysis_stem("front-door.motion.mp4") == "front-door.motion"

