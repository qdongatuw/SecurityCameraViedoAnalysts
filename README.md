# Blink Video Analyzer

批量分析 Blink 摄像头 motion active 短视频。程序会扫描 MP4 文件，使用视频文件修改时间作为拍摄时间，抽取代表性帧，识别触发运动的内容，并输出每个视频的描述文件和汇总表。

## 两种分析模式

- `local-yolo`：本地 YOLO 推理，不上传画面。适合隐私优先和大量视频。RTX 3060 可以用 GPU 跑。
- `cloud-gpt`：抽帧后发送到 OpenAI 视觉模型。描述能力更强，但会消耗 API 额度，并且抽取帧会上传到云端。

本地 YOLO 对人、汽车、卡车、公交车、自行车、猫、狗、鸟等常见类别效果较好。更细的动物种类需要后续接入专门模型或二次分类。

## 安装

本地 YOLO/GPU 模式建议使用 Python 3.12。PyTorch 官方 Windows 安装说明目前支持 Python 3.9 到 3.12；如果你用 Windows Store 的 Python 3.13，可能会出现 `ModuleNotFoundError: No module named 'torch'` 或无法安装 CUDA 版 PyTorch。

```powershell
cd C:\Github\SecurityCameraViedoAnalysts
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

如果想让 3060 跑 CUDA，请先安装与你电脑 CUDA/驱动匹配的 PyTorch，再安装本项目。例如常见 CUDA 12.1：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

如果 PyTorch 官网推荐的是 CUDA 12.8，也可以用：

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

验证 GPU 是否可用：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出是 `False CPU`，说明当前 Python 环境里的 PyTorch 不能使用 CUDA。任务管理器默认的 `3D` 曲线不适合判断深度学习推理；可以用 `nvidia-smi`，或者在任务管理器图表下拉选择 `CUDA` / `Compute`。

## Flet 桌面界面

```powershell
blink-flet
```

或者：

```powershell
python -m blink_video_analyzer.flet_app
```

## 命令行

本地 YOLO：

```powershell
blink-analyze "D:\BlinkVideos" --output "D:\BlinkVideos\analysis" --engine local-yolo --device auto --frames 5
```

云端 GPT：

```powershell
$env:OPENAI_API_KEY="你的 API key"
blink-analyze "D:\BlinkVideos" --output "D:\BlinkVideos\analysis" --engine cloud-gpt --frames 3
```

常用参数：

- `--engine local-yolo`：本地识别，不上传图片
- `--engine cloud-gpt`：云端 GPT 识别
- `--device auto`：自动选择 GPU 或 CPU
- `--device 0`：强制使用第 1 块 GPU
- `--local-model yolov8n.pt`：默认轻量 YOLO 模型，首次运行会自动下载权重
- `--confidence 0.25`：YOLO 检测置信度阈值
- `--frames 5`：每个视频抽取几帧
- `--recursive`：递归扫描子目录
- `--limit 20`：只分析前 20 个视频，便于试跑
- `--overwrite`：覆盖已存在的单视频分析文件

## 输出

每个视频会生成：

- `视频名.analysis.json`
- `视频名.analysis.md`

输出目录还会生成：

- `summary.csv`
- `summary.jsonl`
