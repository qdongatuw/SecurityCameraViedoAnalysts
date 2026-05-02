import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from blink_video_analyzer.pipeline import AnalyzeOptions, analyze_directory


load_dotenv()

st.set_page_config(page_title="Blink Video Analyzer", layout="wide")

st.title("Blink Video Analyzer")
st.caption("批量分析 Blink motion active MP4，使用文件修改时间作为视频时间。")

with st.sidebar:
    input_dir = st.text_input("视频目录", value="")
    output_dir = st.text_input("输出目录", value="")
    engine = st.selectbox("分析引擎", options=["local-yolo", "cloud-gpt"], index=0)
    recursive = st.checkbox("递归扫描子目录", value=True)
    overwrite = st.checkbox("覆盖已有分析", value=False)
    frame_count = st.slider("每个视频抽取帧数", min_value=1, max_value=12, value=5)
    limit = st.number_input("最多处理视频数（0 表示不限）", min_value=0, value=0, step=1)
    model = st.text_input("视觉模型", value=os.getenv("BLINK_ANALYZER_MODEL", "gpt-4.1-mini"))
    local_model = st.text_input("本地 YOLO 模型", value="yolov8n.pt")
    device = st.selectbox("本地推理设备", options=["auto", "0", "cpu"], index=0)
    confidence = st.slider("YOLO 置信度阈值", min_value=0.05, max_value=0.8, value=0.25)
    run = st.button("开始分析", type="primary")

if run:
    if not input_dir.strip():
        st.error("请填写视频目录。")
        st.stop()

    source = Path(input_dir).expanduser()
    if not source.exists():
        st.error(f"视频目录不存在：{source}")
        st.stop()

    destination = Path(output_dir).expanduser() if output_dir.strip() else source / "analysis"
    progress = st.progress(0)
    status = st.empty()
    rows_area = st.empty()
    rows = []

    def on_progress(done: int, total: int, video_name: str) -> None:
        progress.progress(done / total if total else 1.0)
        status.write(f"正在处理 {done}/{total}: {video_name}")

    options = AnalyzeOptions(
        input_dir=source,
        output_dir=destination,
        engine=engine,
        recursive=recursive,
        frame_count=frame_count,
        limit=limit or None,
        model=model,
        local_model=local_model,
        confidence_threshold=confidence,
        device=device,
        overwrite=overwrite,
    )

    try:
        results = analyze_directory(options, progress_callback=on_progress)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    for result in results:
        rows.append(
            {
                "video": result.video_name,
                "modified_time": result.modified_time.isoformat(),
                "motion_objects": ", ".join(result.motion_objects),
                "animal_species": ", ".join(result.animal_species),
                "confidence": result.confidence,
                "description": result.description,
            }
        )

    rows_area.dataframe(rows, use_container_width=True)
    st.success(f"完成：{len(results)} 个视频。输出目录：{destination}")
else:
    st.info("在左侧填写视频目录后开始分析。")
