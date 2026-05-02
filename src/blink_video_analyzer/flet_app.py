from __future__ import annotations

import threading
from pathlib import Path
from tkinter import Tk, filedialog

import flet as ft
from dotenv import load_dotenv

from .local_yolo import describe_compute_device
from .pipeline import AnalyzeOptions, analyze_directory


def main() -> None:
    load_dotenv()
    ft.app(target=_build_app)


def _build_app(page: ft.Page) -> None:
    page.title = "Blink Video Analyzer"
    page.window_width = 1060
    page.window_height = 760
    page.padding = 20
    page.theme_mode = ft.ThemeMode.LIGHT

    input_dir = ft.TextField(label="视频目录", expand=True)
    output_dir = ft.TextField(label="输出目录", expand=True)
    engine = ft.Dropdown(
        label="分析引擎",
        value="local-yolo",
        width=180,
        options=[
            ft.dropdown.Option("local-yolo", "本地 YOLO"),
            ft.dropdown.Option("cloud-gpt", "云端 GPT"),
        ],
    )
    local_model = ft.TextField(label="YOLO 模型", value="yolov8n.pt", width=180)
    cloud_model = ft.TextField(label="GPT 模型", value="gpt-4.1-mini", width=180)
    device = ft.Dropdown(
        label="设备",
        value="auto",
        width=140,
        options=[
            ft.dropdown.Option("auto", "自动"),
            ft.dropdown.Option("0", "GPU 0"),
            ft.dropdown.Option("cpu", "CPU"),
        ],
    )
    frame_count = ft.Slider(min=1, max=12, divisions=11, value=5, label="{value} 帧", expand=True)
    confidence = ft.Slider(min=0.05, max=0.8, divisions=15, value=0.25, label="{value}", expand=True)
    recursive = ft.Checkbox(label="递归扫描子目录", value=True)
    overwrite = ft.Checkbox(label="覆盖已有分析", value=False)
    limit = ft.TextField(label="最多处理数量，空为不限", width=180)
    progress = ft.ProgressBar(value=0)
    status = ft.Text("等待开始")
    device_status = ft.Text(describe_compute_device("auto"), color=ft.Colors.BLUE_GREY_700)
    results = ft.ListView(expand=True, spacing=8, auto_scroll=True)
    run_button = ft.FilledButton("开始分析", icon=ft.Icons.PLAY_ARROW)

    def pick_input(_: ft.ControlEvent) -> None:
        selected = _choose_directory("选择 Blink 视频目录")
        if selected:
            input_dir.value = selected
            if not output_dir.value:
                output_dir.value = str(Path(selected) / "analysis")
            page.update()

    def pick_output(_: ft.ControlEvent) -> None:
        selected = _choose_directory("选择分析输出目录")
        if selected:
            output_dir.value = selected
            page.update()

    def refresh_device_status(_: ft.ControlEvent | None = None) -> None:
        if engine.value == "local-yolo":
            device_status.value = describe_compute_device(device.value or "auto")
        else:
            device_status.value = "Cloud GPT mode uploads sampled frames to OpenAI."
        page.update()

    def set_running(is_running: bool) -> None:
        run_button.disabled = is_running
        run_button.text = "分析中" if is_running else "开始分析"
        page.update()

    def append_log(message: str) -> None:
        results.controls.append(ft.Text(message, selectable=True))
        page.update()

    def on_progress(done: int, total: int, video_name: str) -> None:
        progress.value = done / total if total else 1
        status.value = f"{done}/{total}  {video_name}"
        page.update()

    def run_analysis(_: ft.ControlEvent) -> None:
        source_text = input_dir.value.strip()
        if not source_text:
            page.snack_bar = ft.SnackBar(ft.Text("请先选择视频目录。"))
            page.snack_bar.open = True
            page.update()
            return

        def worker() -> None:
            set_running(True)
            results.controls.clear()
            progress.value = 0
            status.value = "正在准备..."
            page.update()

            try:
                source = Path(source_text).expanduser()
                destination = Path(output_dir.value.strip()).expanduser() if output_dir.value.strip() else source / "analysis"
                parsed_limit = int(limit.value) if limit.value.strip() else None
                options = AnalyzeOptions(
                    input_dir=source,
                    output_dir=destination,
                    engine=engine.value or "local-yolo",
                    recursive=bool(recursive.value),
                    frame_count=int(frame_count.value),
                    limit=parsed_limit,
                    model=cloud_model.value.strip() or "gpt-4.1-mini",
                    local_model=local_model.value.strip() or "yolov8n.pt",
                    confidence_threshold=float(confidence.value),
                    device=device.value or "auto",
                    overwrite=bool(overwrite.value),
                )
                device_status.value = (
                    describe_compute_device(options.device)
                    if options.engine == "local-yolo"
                    else "Cloud GPT mode uploads sampled frames to OpenAI."
                )
                page.update()
                analyzed = analyze_directory(options, progress_callback=on_progress)
                for item in analyzed:
                    append_log(
                        f"{item.video_name} | {item.modified_time:%Y-%m-%d %H:%M:%S} | "
                        f"{', '.join(item.motion_objects) or 'unknown'} | {item.description}"
                    )
                status.value = f"完成：{len(analyzed)} 个视频。输出：{destination}"
                progress.value = 1
                page.update()
            except Exception as exc:
                append_log(f"错误：{exc}")
                status.value = "分析失败"
                page.update()
            finally:
                set_running(False)

        threading.Thread(target=worker, daemon=True).start()

    run_button.on_click = run_analysis
    engine.on_change = refresh_device_status
    device.on_change = refresh_device_status

    page.add(
        ft.Text("Blink Video Analyzer", size=26, weight=ft.FontWeight.BOLD),
        ft.Text("本地 YOLO 可使用 3060 显卡；云端 GPT 会上传抽取帧。"),
        ft.Row(
            [
                input_dir,
                ft.IconButton(icon=ft.Icons.FOLDER_OPEN, tooltip="选择视频目录", on_click=pick_input),
            ]
        ),
        ft.Row(
            [
                output_dir,
                ft.IconButton(icon=ft.Icons.FOLDER_OPEN, tooltip="选择输出目录", on_click=pick_output),
            ]
        ),
        ft.Row([engine, local_model, cloud_model, device, limit], spacing=12),
        ft.Row(
            [
                ft.Text("抽帧数", width=70),
                frame_count,
                ft.Text("YOLO 置信度", width=100),
                confidence,
            ]
        ),
        device_status,
        ft.Row([recursive, overwrite, run_button], spacing=16),
        progress,
        status,
        ft.Divider(),
        results,
    )

def _choose_directory(title: str) -> str | None:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title=title)
    finally:
        root.destroy()
    return selected or None


if __name__ == "__main__":
    main()
