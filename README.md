# Whisper 字幕工具

这是一个面向 Windows 的本地字幕工具，基于 OpenAI Whisper，提供图形界面，支持：

- 视频提取音频并转换为 `mp3`
- 音频或视频批量生成 `srt` 字幕
- 原文字幕生成
- 英文字幕生成
- 日语字幕生成
- 接入 OpenAI 兼容翻译 Provider 做字幕翻译

当前项目已经固定使用 Conda 环境 `whisper-env`。程序启动、Whisper 调用、`ffmpeg` 调用都会优先使用这个环境。

## 1. 环境安装

### 1.1 安装 Conda

如果电脑还没有 Conda，请先安装：

- 官网参考：https://anaconda.org/anaconda/conda

推荐使用 Miniconda 或 Anaconda，安装完成后确保 `conda` 和 `conda.bat` 可以在终端中使用。

### 1.2 创建项目环境

在终端执行：

```bash
conda create -n "whisper-env" python=3.11
conda activate "whisper-env"
```

### 1.3 安装 PyTorch

如果你使用 NVIDIA 显卡并希望走 `cuda`，可安装 CUDA 版 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你不使用 CUDA，也可以安装 CPU 版 PyTorch。  
当前软件默认设备是 `cuda`，如果环境里没有可用 CUDA，请在界面里改成 `cpu`。

### 1.4 安装 ffmpeg

`mp4 -> mp3` 转换依赖 `ffmpeg`。请在 `whisper-env` 环境中安装：

```bash
conda install ffmpeg -c conda-forge
```

安装完成后，程序会优先使用：

```text
C:\Users\29c\miniconda3\envs\whisper-env\Library\bin\ffmpeg.exe
```

### 1.5 安装 Whisper

在 `whisper-env` 环境中安装：

```bash
pip install -U openai-whisper
```

如果网络较慢，也可以使用镜像：

```bash
pip install -U openai-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.6 可选：检查安装是否成功

在 `whisper-env` 环境中执行：

```bash
python -c "import whisper, torch; print('whisper ok'); print('cuda:', torch.cuda.is_available())"
ffmpeg -version
```

如果以上命令都能正常执行，说明环境已经基本可用。

## 2. 软件介绍

### 2.1 软件定位

这个工具用于本地批量处理媒体文件，核心流程如下：

1. 如果输入是视频，先从视频中提取音频并转成 `mp3`
2. 调用 Whisper 识别语音并生成字幕
3. 根据目标语言，决定直接输出原文、使用 Whisper 原生英文翻译，或调用外部翻译 Provider
4. 将最终字幕保存到输出目录

### 2.2 主要功能

- 批量添加视频或音频文件
- 一键把视频转成 `mp3`
- 一键生成字幕
- 支持 `transcribe` 与 `translate-to-English` 两种任务模式
- 支持 OpenAI 兼容接口作为字幕翻译 Provider
- 自动把输出文件保存到项目内 `output` 目录

### 2.3 当前默认设置

当前程序内置的推荐参数更偏向本地桌面显卡环境：

- `model`: `turbo`
- `device`: `cuda`
- `fp16`: 开启
- `beam_size`: `5`
- `temperature`: `0`
- `threads`: `4`

如果你更看重准确率，可以把模型改成 `medium`；如果显存较小，建议继续使用 `turbo`。

### 2.4 输出目录说明

默认输出位置：

- 字幕输出：`output`
- 视频提取后的音频：`output\mp3`
- 处理中间文件：`output\.subtitle_gui_work`

## 3. 使用教程

### 3.1 启动软件

环境安装完成后，直接双击：

```text
启动字幕工具.bat
```

这个启动脚本会先执行：

```bash
conda activate "whisper-env"
```

然后再启动图形界面。

### 3.2 只做视频转 MP3

适合先把视频里的音频提取出来单独使用。

步骤：

1. 点击“添加文件”
2. 选择一个或多个视频文件
3. 确认输出目录
4. 点击“转换为 MP3”

处理完成后，生成的音频默认保存在：

```text
output\mp3
```

### 3.3 生成原文字幕

适合保留原始语种字幕。

推荐设置：

- `Task Mode`: `transcribe`
- `Target Subtitle`: `原文`

步骤：

1. 添加视频或音频文件
2. 选择输出目录
3. 选择 Whisper 模型
4. 点击“生成字幕”

### 3.4 生成英文字幕

有两种方式：

方式一：使用 Whisper 原生英文翻译

- `Task Mode`: `translate-to-English`
- `Target Subtitle`: `英语`

这种方式不需要外部翻译接口。

方式二：先转原文，再走翻译 Provider

- `Task Mode`: `transcribe`
- `Target Subtitle`: `英语`
- 需要填写 `Base URL`、`API Key`、`Model`

### 3.5 生成日语字幕

日语字幕走的是“先识别、再翻译”的流程，因此需要翻译 Provider。

推荐设置：

- `Task Mode`: `transcribe`
- `Target Subtitle`: `日语`
- 填写 `Base URL`、`API Key`、`Model`

### 3.6 翻译 Provider 的填写方式

软件支持 OpenAI 兼容接口。你需要在界面中填写：

- `Base URL`
- `API Key`
- `Model`
- `System Prompt`

例如：

- `Base URL`: 兼容 `/v1/chat/completions` 的接口地址
- `Model`: 你实际可用的对话模型名

如果目标语言是英语或日语，且任务模式为 `transcribe`，没有填写 Provider 会直接报错。

### 3.7 处理完成后的文件命名

输出字幕文件会按原文件名生成，例如：

- 原文字幕：`文件名.srt`
- 英文字幕：`文件名.en.srt`
- 日语字幕：`文件名.ja.srt`

如果目标文件已存在，程序会自动避免直接覆盖。

## 4. 软件界面介绍

主界面大致分为四个区域。

### 4.1 控制面板

位于窗口上方，主要用于设置任务参数，包括：

- 输出目录
- Whisper 模型
- 音频语言
- 任务模式
- 目标字幕语言
- 运行设备 `cuda / cpu`
- `fp16`
- `beam_size`
- `temperature`
- `CPU threads`
- `ffmpeg.exe` 路径

说明：

- 当前版本中，`ffmpeg` 会优先固定使用 `whisper-env` 环境里的版本
- 这个输入框主要用于显示当前实际使用的路径

### 4.2 操作按钮区

控制面板下方的按钮包含：

- “添加文件”：加入待处理媒体文件
- “清空列表”：清空当前任务列表
- “转换为 MP3”：只提取音频，不生成字幕
- “生成字幕”：执行完整字幕流程

### 4.3 任务列表

中间区域用于显示每个文件的处理状态，通常会看到：

- 文件名
- 文件类型
- 当前状态
- 最新输出路径

适合查看批量任务的执行进度。

### 4.4 Provider 配置区

下方区域用于填写翻译接口参数：

- `Base URL`
- `Model`
- `API Key`
- `System Prompt`

如果你只做原文字幕，或者用 Whisper 原生英文翻译，可以不填这一块。

### 4.5 日志区

日志区会显示处理过程中的关键输出，例如：

- `ffmpeg` 转换命令
- Whisper 调用命令
- CUDA 检测结果
- 当前文件处理进度
- 报错信息

遇到问题时，优先查看这里的最后几行。

## 常见问题

### 1. 提示缺少 whisper

先确认当前环境是：

```bash
conda activate "whisper-env"
```

然后执行：

```bash
pip install -U openai-whisper
```

### 2. 提示未找到 ffmpeg

请在 `whisper-env` 中执行：

```bash
conda install ffmpeg -c conda-forge
```

### 3. 提示当前设置为 cuda，但没有检测到 CUDA

说明当前环境中的 PyTorch 不是可用的 CUDA 版本，或者显卡驱动没有准备好。可以先执行：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出是 `False`，请检查驱动或重新安装对应版本的 PyTorch。

### 4. 日文或特殊字符文件报编码错误

当前版本已经针对 Whisper 控制台编码问题做了兼容处理。  
如果仍然报错，优先把日志末尾几行完整复制出来排查。

## 命令行示例

如果你想在 `whisper-env` 中手动测试 Whisper，可执行：

```bash
conda activate "whisper-env"
python -m whisper input.mp3 --model turbo --device cuda --fp16 True --beam_size 5 --temperature 0 --threads 4 --task transcribe --output_format srt
```
