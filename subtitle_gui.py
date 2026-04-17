#!/usr/bin/env python3
"""Batch media-to-subtitle GUI for local Whisper workflows."""

from __future__ import annotations

import hashlib
import json
import locale
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / ".subtitle_gui_config.json"
DEFAULT_OUTPUT_DIR = APP_DIR / "output"
DEFAULT_CONDA_ENV_NAME = "whisper-env"
DEFAULT_SYSTEM_PROMPT = (
    "You are a subtitle translator. Translate the given subtitle text into the "
    "requested language. Return only the translated subtitle text. Keep line "
    "breaks when they help readability, and do not add explanations."
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".rmvb"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
APP_FFMPEG_CANDIDATES = [
    APP_DIR / "ffmpeg.exe",
    APP_DIR / "ffmpeg" / "bin" / "ffmpeg.exe",
    APP_DIR / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
]

LANGUAGE_OPTIONS = [
    ("自动检测", ""),
    ("中文", "Chinese"),
    ("英语", "English"),
    ("日语", "Japanese"),
    ("韩语", "Korean"),
    ("法语", "French"),
    ("德语", "German"),
    ("西班牙语", "Spanish"),
]
LANGUAGE_CODES = {
    "原文": "",
    "英语": "en",
    "日语": "ja",
}
MODEL_OPTIONS = ["turbo", "tiny", "base", "small", "medium", "large"]
DEVICE_OPTIONS = ["cuda", "cpu"]
TARGET_OPTIONS = ["原文", "英语", "日语"]
TASK_OPTIONS = ["transcribe", "translate-to-English"]
PROFILE_HINT = "推荐默认值: turbo + cuda + fp16 + beam_size 5 + temperature 0 + threads 4"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _conda_prefix_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if env_prefix:
        candidates.append(Path(env_prefix).expanduser())

    if sys.executable:
        python_path = Path(sys.executable).resolve()
        candidates.append(python_path.parent)
        if python_path.parent.name.lower() == "scripts":
            candidates.append(python_path.parent.parent)
        for parent in python_path.parents:
            if parent.name.lower() in {"miniconda3", "anaconda3", "miniforge3", "mambaforge"}:
                candidates.append(parent / "envs" / DEFAULT_CONDA_ENV_NAME)
                break

    home_dir = Path.home()
    for base_name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        candidates.append(home_dir / base_name / "envs" / DEFAULT_CONDA_ENV_NAME)

    return _dedupe_paths(candidates)


def _detect_conda_python_path() -> Path | None:
    env_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if env_prefix:
        python_path = Path(env_prefix).expanduser() / "python.exe"
        if python_path.exists() and python_path.is_file():
            return python_path

    candidates: list[Path] = []
    if sys.executable:
        python_path = Path(sys.executable).resolve()
        for parent in python_path.parents:
            if parent.name.lower() in {"miniconda3", "anaconda3", "miniforge3", "mambaforge"}:
                candidates.append(parent / "envs" / DEFAULT_CONDA_ENV_NAME / "python.exe")
                break

    home_dir = Path.home()
    for base_name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        candidates.append(home_dir / base_name / "envs" / DEFAULT_CONDA_ENV_NAME / "python.exe")

    for python_path in _dedupe_paths(candidates):
        if python_path.exists() and python_path.is_file():
            return python_path
    return None


def _detect_conda_ffmpeg_path() -> Path | None:
    env_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    candidates: list[Path] = []
    if env_prefix:
        prefix = Path(env_prefix).expanduser()
        candidates.extend(
            [
                prefix / "Library" / "bin" / "ffmpeg.exe",
                prefix / "Scripts" / "ffmpeg.exe",
                prefix / "bin" / "ffmpeg",
            ]
        )

    for prefix in _conda_prefix_candidates():
        candidates.extend(
            [
                prefix / "Library" / "bin" / "ffmpeg.exe",
                prefix / "Scripts" / "ffmpeg.exe",
                prefix / "bin" / "ffmpeg",
            ]
        )

    for candidate in _dedupe_paths(candidates):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _default_ffmpeg_candidates() -> list[Path]:
    candidates: list[Path] = []
    for prefix in _conda_prefix_candidates():
        candidates.extend(
            [
                prefix / "Library" / "bin" / "ffmpeg.exe",
                prefix / "Scripts" / "ffmpeg.exe",
                prefix / "bin" / "ffmpeg",
            ]
        )
    candidates.extend(APP_FFMPEG_CANDIDATES)
    return _dedupe_paths(candidates)


def _detect_default_ffmpeg_path() -> Path | None:
    for candidate in _default_ffmpeg_candidates():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


@dataclass
class SubtitleSegment:
    index: int
    timing: str
    text: str


@dataclass
class JobSettings:
    output_dir: Path
    model: str
    language: str
    task_mode: str
    target_language: str
    device: str
    fp16: bool
    temperature: float
    beam_size: int
    threads: int
    ffmpeg_path: Path | None
    provider_base_url: str
    provider_api_key: str
    provider_model: str
    provider_system_prompt: str


class ConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def save(self, data: dict[str, Any]) -> None:
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class TranslationProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str,
        logger,
    ) -> None:
        self.base_url = base_url.strip()
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.logger = logger

    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def translate_text(self, text: str, target_language: str) -> str:
        if not text.strip():
            return text
        endpoint = self._build_endpoint(self.base_url)
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Target language: {target_language}\n"
                        "Translate the subtitle text below. Return only the "
                        "translated subtitle text.\n\n"
                        f"{text}"
                    ),
                },
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Translation API HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Translation API connection failed: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected translation response: {body}") from exc

    @staticmethod
    def _build_endpoint(base_url: str) -> str:
        parsed = urllib.parse.urlparse(base_url)
        if not parsed.scheme or not parsed.netloc:
            raise RuntimeError("Base URL is invalid. Example: https://api.deepseek.com/v1")
        if parsed.path.endswith("/chat/completions"):
            return base_url
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = f"{path}/chat/completions"
        elif not path:
            path = "/v1/chat/completions"
        else:
            path = f"{path}/chat/completions"
        return urllib.parse.urlunparse(parsed._replace(path=path))


class SubtitleProcessor:
    def __init__(self, logger) -> None:
        self.logger = logger
        self.python_command = self._resolve_python_command()

    def ensure_dependencies(self, settings: JobSettings) -> str:
        ffmpeg_command = self._resolve_ffmpeg_command(settings.ffmpeg_path)
        if ffmpeg_command is None:
            raise RuntimeError(
                "未找到 whisper-env 环境里的 ffmpeg。\n"
                "请先运行: conda activate \"whisper-env\"\n"
                "然后安装: conda install ffmpeg -c conda-forge"
            )
        if ffmpeg_command is None:
            raise RuntimeError(
                "未找到 ffmpeg。可选做法:\n"
                "1. 把 ffmpeg.exe 加到 PATH\n"
                "2. 在界面里填写 ffmpeg.exe 路径\n"
                "3. 把 ffmpeg.exe 放到 tools\\ffmpeg\\bin\\ffmpeg.exe"
            )
        result = subprocess.run(
            [self.python_command, "-c", "import whisper"],
            capture_output=True,
            check=False,
        )
        stdout = self._safe_decode(result.stdout)
        stderr = self._safe_decode(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                "当前 Python 环境缺少 whisper。\n"
                "请先运行 安装依赖_cuda.bat\n"
                f"Python: {self.python_command}\n"
                f"{stderr.strip() or stdout.strip()}"
            )
        if settings.device == "cuda":
            self._ensure_cuda_available()
        return ffmpeg_command

    def convert_to_mp3(self, source_path: Path, output_dir: Path, ffmpeg_command: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        mp3_path = output_dir / f"{source_path.stem}.mp3"
        if mp3_path.exists():
            mp3_path = self._ensure_unique_path(mp3_path)
        command = [
            ffmpeg_command,
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(mp3_path),
        ]
        self.logger(f"[ffmpeg] {self._format_command(command)}")
        self._run_command(command, f"转换 {source_path.name} 为 MP3 失败")
        return mp3_path

    def run_whisper(
        self,
        audio_path: Path,
        output_dir: Path,
        settings: JobSettings,
        task: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_command,
            "-m",
            "whisper",
            str(audio_path),
            "--model",
            settings.model,
            "--device",
            settings.device,
            "--fp16",
            "True" if settings.fp16 else "False",
            "--temperature",
            str(settings.temperature),
            "--beam_size",
            str(settings.beam_size),
            "--threads",
            str(settings.threads),
            "--output_dir",
            str(output_dir),
            "--output_format",
            "srt",
            "--task",
            task,
            "--verbose",
            "False",
        ]
        if settings.language:
            command.extend(["--language", settings.language])
        self.logger(f"[whisper] {self._format_command(command)}")
        self._run_command(command, f"Whisper 处理 {audio_path.name} 失败")
        srt_path = output_dir / f"{audio_path.stem}.srt"
        if not srt_path.exists():
            raise RuntimeError(f"Whisper 运行完成，但未找到输出文件: {srt_path}")
        return srt_path

    def translate_srt(
        self,
        source_srt: Path,
        target_srt: Path,
        target_language: str,
        provider: TranslationProvider,
    ) -> Path:
        segments = parse_srt(source_srt)
        if not segments:
            raise RuntimeError(f"No subtitle segments found in {source_srt.name}")
        if target_srt.exists():
            target_srt = self._ensure_unique_path(target_srt)
        translated_segments = []
        total = len(segments)
        for idx, segment in enumerate(segments, start=1):
            self.logger(f"Translating subtitle {idx}/{total} -> {target_language}")
            translated_text = provider.translate_text(segment.text, target_language)
            translated_segments.append(
                SubtitleSegment(
                    index=segment.index,
                    timing=segment.timing,
                    text=translated_text,
                )
            )
        write_srt(target_srt, translated_segments)
        return target_srt

    def finalize_output(self, source_srt: Path, destination_srt: Path) -> Path:
        destination_srt.parent.mkdir(parents=True, exist_ok=True)
        if destination_srt.exists():
            destination_srt = self._ensure_unique_path(destination_srt)
        shutil.copy2(source_srt, destination_srt)
        return destination_srt

    def build_destination_path(
        self,
        output_dir: Path,
        original_file: Path,
        target_language: str,
        whisper_language: str,
    ) -> Path:
        suffix = LANGUAGE_CODES.get(target_language, "")
        if target_language == "原文" and whisper_language:
            suffix = whisper_language.lower()
        filename = original_file.stem if not suffix else f"{original_file.stem}.{suffix}"
        return output_dir / f"{filename}.srt"

    @staticmethod
    def make_work_dir(output_dir: Path, source_path: Path) -> Path:
        digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
        work_dir = output_dir / ".subtitle_gui_work" / f"{source_path.stem}_{digest}"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    @staticmethod
    def _resolve_python_command() -> str:
        conda_python = _detect_conda_python_path()
        if conda_python is not None:
            return str(conda_python)
        if sys.executable:
            return sys.executable
        return shutil.which("python") or "python"

    @staticmethod
    def _resolve_ffmpeg_command(ffmpeg_path: Path | None) -> str | None:
        conda_ffmpeg = _detect_conda_ffmpeg_path()
        if conda_ffmpeg is not None:
            return str(conda_ffmpeg)
        return None

    def _ensure_cuda_available(self) -> None:
        code = (
            "import sys\n"
            "try:\n"
            "    import torch\n"
            "except Exception as exc:\n"
            "    print(f'IMPORT_ERROR:{exc}')\n"
            "    raise SystemExit(1)\n"
            "if not torch.cuda.is_available():\n"
            "    print('CUDA_UNAVAILABLE')\n"
            "    raise SystemExit(2)\n"
            "print(torch.cuda.get_device_name(0))\n"
        )
        result = subprocess.run(
            [self.python_command, "-c", code],
            capture_output=True,
            check=False,
        )
        stdout = self._safe_decode(result.stdout).strip()
        stderr = self._safe_decode(result.stderr).strip()
        if result.returncode == 0:
            self.logger(f"[CUDA] 已检测到显卡: {stdout}")
            return
        if "IMPORT_ERROR:" in stdout:
            raise RuntimeError(
                "当前 Python 环境未正确安装 PyTorch。\n"
                "请先运行 安装依赖_cuda.bat，然后再试。"
            )
        raise RuntimeError(
            "当前设置为 cuda，但未检测到可用的 NVIDIA CUDA 环境。\n"
            "请先确认显卡驱动正常，且安装的是支持 CUDA 的 PyTorch。\n"
            f"{stderr or stdout}"
        )

    @staticmethod
    def _ensure_unique_path(path: Path) -> Path:
        counter = 2
        while True:
            candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _run_command(self, command: list[str], error_prefix: str) -> None:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        assert process.stdout is not None

        collected_chunks: list[str] = []
        line_buffer = ""

        while True:
            chunk = process.stdout.read(1)
            if not chunk:
                break
            text = self._safe_decode(chunk)
            collected_chunks.append(text)
            if text in {"\n", "\r"}:
                if line_buffer.strip():
                    self.logger(line_buffer.strip())
                line_buffer = ""
                continue
            line_buffer += text

        return_code = process.wait()
        if line_buffer.strip():
            self.logger(line_buffer.strip())

        output_text = "".join(collected_chunks).strip()
        if return_code != 0:
            raise RuntimeError(f"{error_prefix}: {output_text or 'Unknown error'}")

    @staticmethod
    def _safe_decode(payload: bytes | str | None) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        encodings = ["utf-8", locale.getpreferredencoding(False), "gbk", "cp936", "cp1252"]
        tried: set[str] = set()
        for encoding in encodings:
            if not encoding or encoding in tried:
                continue
            tried.add(encoding)
            try:
                return payload.decode(encoding)
            except UnicodeDecodeError:
                continue
        return payload.decode("utf-8", errors="replace")

    @staticmethod
    def _format_command(command: list[str]) -> str:
        return subprocess.list2cmdline(command)


class SubtitleGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Whisper 字幕工具")
        self.root.geometry("1260x860")

        self.config_store = ConfigStore(CONFIG_PATH)
        self.config = self.config_store.load()
        self.processor = SubtitleProcessor(self.log)
        self.selected_files: list[Path] = []
        self.item_by_path: dict[Path, str] = {}
        self.log_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.output_dir_var = tk.StringVar(value=self._initial_output_dir())
        self.model_var = tk.StringVar(value=self.config.get("model", "turbo"))
        self.source_language_var = tk.StringVar(value=self.config.get("source_language", "自动检测"))
        self.task_mode_var = tk.StringVar(value=self.config.get("task_mode", "transcribe"))
        self.target_language_var = tk.StringVar(value=self.config.get("target_language", "原文"))
        self.device_var = tk.StringVar(value=self.config.get("device", "cuda"))
        self.fp16_var = tk.BooleanVar(value=bool(self.config.get("fp16", True)))
        self.temperature_var = tk.StringVar(value=str(self.config.get("temperature", "0")))
        self.beam_size_var = tk.StringVar(value=str(self.config.get("beam_size", "5")))
        self.threads_var = tk.StringVar(value=str(self.config.get("threads", "4")))
        self.ffmpeg_path_var = tk.StringVar(value=self._initial_ffmpeg_path())
        self.base_url_var = tk.StringVar(value=self.config.get("base_url", ""))
        self.api_key_var = tk.StringVar(value=self.config.get("api_key", ""))
        self.provider_model_var = tk.StringVar(value=self.config.get("provider_model", ""))
        self.status_var = tk.StringVar(value="就绪")

        self._build_layout()
        self._refresh_task_mode_hint()
        self.root.after(150, self._drain_log_queue)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        self.root.rowconfigure(3, weight=1)

        controls = ttk.LabelFrame(self.root, text="控制面板", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))
        for column in range(4):
            controls.columnconfigure(column, weight=1)

        ttk.Label(controls, text="输出目录").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.output_dir_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(0, 8)
        )
        ttk.Button(controls, text="浏览", command=self.choose_output_dir).grid(
            row=0, column=3, sticky="ew"
        )

        ttk.Label(controls, text="Whisper Model").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=MODEL_OPTIONS,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(10, 0))

        ttk.Label(controls, text="音频语言").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.source_language_var,
            values=[label for label, _ in LANGUAGE_OPTIONS],
            state="readonly",
        ).grid(row=1, column=3, sticky="ew", pady=(10, 0))

        ttk.Label(controls, text="任务模式").grid(row=2, column=0, sticky="w", pady=(10, 0))
        task_box = ttk.Combobox(
            controls,
            textvariable=self.task_mode_var,
            values=TASK_OPTIONS,
            state="readonly",
        )
        task_box.grid(row=2, column=1, sticky="ew", padx=(0, 8), pady=(10, 0))
        task_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_task_mode_hint())

        ttk.Label(controls, text="目标字幕").grid(row=2, column=2, sticky="w", pady=(10, 0))
        target_box = ttk.Combobox(
            controls,
            textvariable=self.target_language_var,
            values=TARGET_OPTIONS,
            state="readonly",
        )
        target_box.grid(row=2, column=3, sticky="ew", pady=(10, 0))
        target_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_task_mode_hint())

        ttk.Label(controls, text="运行设备").grid(row=3, column=0, sticky="w", pady=(10, 0))
        ttk.Combobox(
            controls,
            textvariable=self.device_var,
            values=DEVICE_OPTIONS,
            state="readonly",
        ).grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(10, 0))

        ttk.Checkbutton(
            controls,
            text="启用 fp16",
            variable=self.fp16_var,
            onvalue=True,
            offvalue=False,
        ).grid(row=3, column=2, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="beam_size").grid(row=4, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.beam_size_var).grid(
            row=4, column=1, sticky="ew", padx=(0, 8), pady=(10, 0)
        )

        ttk.Label(controls, text="temperature").grid(row=4, column=2, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.temperature_var).grid(
            row=4, column=3, sticky="ew", pady=(10, 0)
        )

        ttk.Label(controls, text="CPU threads").grid(row=5, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.threads_var).grid(
            row=5, column=1, sticky="ew", padx=(0, 8), pady=(10, 0)
        )

        ttk.Label(controls, text="ffmpeg.exe 路径").grid(row=6, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.ffmpeg_path_var).grid(
            row=6, column=1, columnspan=2, sticky="ew", padx=(0, 8), pady=(10, 0)
        )
        ttk.Button(controls, text="浏览", command=self.choose_ffmpeg_file).grid(
            row=6, column=3, sticky="ew", pady=(10, 0)
        )

        self.profile_hint_label = ttk.Label(controls, text=PROFILE_HINT, foreground="#0b6b3a")
        self.profile_hint_label.grid(row=7, column=0, columnspan=4, sticky="w", pady=(8, 0))

        self.task_hint_label = ttk.Label(controls, text="", foreground="#8a5a00")
        self.task_hint_label.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))

        actions = ttk.Frame(controls)
        actions.grid(row=9, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        actions.columnconfigure(3, weight=1)

        ttk.Button(actions, text="添加文件", command=self.add_files).grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(actions, text="清空列表", command=self.clear_files).grid(
            row=0, column=1, sticky="ew", padx=(0, 8)
        )
        self.convert_button = ttk.Button(
            actions, text="转换为 MP3", command=self.start_convert_only
        )
        self.convert_button.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        self.subtitle_button = ttk.Button(
            actions, text="生成字幕", command=self.start_generate_subtitles
        )
        self.subtitle_button.grid(row=0, column=3, sticky="ew")

        file_frame = ttk.LabelFrame(self.root, text="任务列表", padding=12)
        file_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        file_frame.columnconfigure(0, weight=1)
        file_frame.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(
            file_frame,
            columns=("file", "type", "status", "output"),
            show="headings",
            height=12,
        )
        self.tree.heading("file", text="文件")
        self.tree.heading("type", text="类型")
        self.tree.heading("status", text="状态")
        self.tree.heading("output", text="最新输出")
        self.tree.column("file", width=360, anchor="w")
        self.tree.column("type", width=120, anchor="center")
        self.tree.column("status", width=160, anchor="center")
        self.tree.column("output", width=470, anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(file_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        provider_frame = ttk.LabelFrame(self.root, text="OpenAI 兼容翻译 Provider", padding=12)
        provider_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(8, 12))
        provider_frame.columnconfigure(1, weight=1)
        provider_frame.columnconfigure(3, weight=1)
        provider_frame.rowconfigure(4, weight=1)

        ttk.Label(provider_frame, text="Base URL").grid(row=0, column=0, sticky="w")
        ttk.Entry(provider_frame, textvariable=self.base_url_var).grid(
            row=0, column=1, sticky="ew", padx=(0, 12)
        )
        ttk.Label(provider_frame, text="Model").grid(row=0, column=2, sticky="w")
        ttk.Entry(provider_frame, textvariable=self.provider_model_var).grid(
            row=0, column=3, sticky="ew"
        )

        ttk.Label(provider_frame, text="API Key").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(provider_frame, textvariable=self.api_key_var, show="*").grid(
            row=1, column=1, columnspan=3, sticky="ew", pady=(10, 0)
        )

        ttk.Label(provider_frame, text="System Prompt").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        self.system_prompt_text = tk.Text(provider_frame, height=5, wrap="word")
        self.system_prompt_text.grid(row=2, column=1, columnspan=3, sticky="nsew", pady=(10, 0))
        self.system_prompt_text.insert(
            "1.0",
            self.config.get("provider_system_prompt", DEFAULT_SYSTEM_PROMPT),
        )

        log_frame = ttk.LabelFrame(provider_frame, text="日志", padding=8)
        log_frame.grid(row=4, column=0, columnspan=4, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

    def _initial_output_dir(self) -> str:
        saved = str(self.config.get("output_dir", "")).strip()
        if not saved:
            return str(DEFAULT_OUTPUT_DIR)
        saved_path = Path(saved).expanduser()
        if saved.startswith("/Users/") and not saved_path.exists():
            return str(DEFAULT_OUTPUT_DIR)
        return str(saved_path)

    def _initial_ffmpeg_path(self) -> str:
        conda_ffmpeg = _detect_conda_ffmpeg_path()
        if conda_ffmpeg is not None:
            return str(conda_ffmpeg)
        default_path = _detect_default_ffmpeg_path()
        return str(default_path) if default_path is not None else ""

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_dir_var.get() or str(DEFAULT_OUTPUT_DIR))
        if path:
            self.output_dir_var.set(path)
            self.save_config()

    def choose_ffmpeg_file(self) -> None:
        initial_dir = str(APP_DIR)
        current = self.ffmpeg_path_var.get().strip()
        if current:
            initial_dir = str(Path(current).expanduser().parent)
        path = filedialog.askopenfilename(
            title="选择 ffmpeg.exe",
            initialdir=initial_dir,
            filetypes=[("ffmpeg executable", "ffmpeg.exe"), ("Executable", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self.ffmpeg_path_var.set(path)
            self.save_config()

    def add_files(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="选择媒体文件",
            filetypes=[("媒体文件", "*.mp4 *.mov *.mkv *.avi *.webm *.rmvb *.mp3 *.wav *.m4a *.aac *.flac *.ogg")],
        )
        if not filenames:
            return
        for name in filenames:
            path = Path(name)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                self.log(f"已跳过不支持的文件: {path}")
                continue
            if path in self.item_by_path:
                continue
            item_id = self.tree.insert(
                "",
                "end",
                values=(str(path), self.describe_file(path), "Queued", ""),
            )
            self.selected_files.append(path)
            self.item_by_path[path] = item_id
        self.save_config()

    def clear_files(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("任务进行中", "请等待当前任务完成。")
            return
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        self.selected_files.clear()
        self.item_by_path.clear()
        self.status_var.set("就绪")

    def start_convert_only(self) -> None:
        self._start_worker(mode="convert")

    def start_generate_subtitles(self) -> None:
        self._start_worker(mode="subtitle")

    def _start_worker(self, mode: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("任务进行中", "当前已有任务在运行。")
            return
        if not self.selected_files:
            messagebox.showwarning("未选择文件", "请先添加至少一个媒体文件。")
            return
        try:
            settings = self.collect_settings(mode)
        except RuntimeError as exc:
            messagebox.showerror("设置无效", str(exc))
            return

        self.save_config()
        self._set_running_state(True, f"正在执行{self._mode_label(mode)}...")
        self.log(f"开始执行{self._mode_label(mode)}，共 {len(self.selected_files)} 个文件。")
        self.worker_thread = threading.Thread(
            target=self._worker_main,
            args=(mode, settings, list(self.selected_files)),
            daemon=True,
        )
        self.worker_thread.start()

    def collect_settings(self, mode: str) -> JobSettings:
        output_dir_text = self.output_dir_var.get().strip()
        if not output_dir_text:
            raise RuntimeError("请选择输出目录。")
        output_dir = Path(output_dir_text).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"无法创建输出目录: {exc}") from exc

        device = self.device_var.get().strip().lower()
        if device not in DEVICE_OPTIONS:
            raise RuntimeError("运行设备只能是 cuda 或 cpu。")

        beam_size = self._parse_positive_int(self.beam_size_var.get(), "beam_size")
        threads = self._parse_positive_int(self.threads_var.get(), "CPU threads")
        temperature = self._parse_non_negative_float(self.temperature_var.get(), "temperature")

        ffmpeg_path = _detect_conda_ffmpeg_path()
        if ffmpeg_path is None:
            raise RuntimeError(
                "未找到 whisper-env 环境里的 ffmpeg。\n"
                "请先运行: conda activate \"whisper-env\"\n"
                "然后安装: conda install ffmpeg -c conda-forge"
            )
        self.ffmpeg_path_var.set(str(ffmpeg_path))

        source_language = self._language_value(self.source_language_var.get())
        task_mode = self.task_mode_var.get()
        target_language = self.target_language_var.get()
        provider_base_url = self.base_url_var.get().strip()
        provider_api_key = self.api_key_var.get().strip()
        provider_model = self.provider_model_var.get().strip()
        provider_system_prompt = self.system_prompt_text.get("1.0", "end").strip()

        if mode == "subtitle":
            if target_language == "原文" and task_mode != "transcribe":
                raise RuntimeError("生成原文字幕时，任务模式必须是 'transcribe'。")
            if target_language == "日语" and task_mode != "transcribe":
                raise RuntimeError("生成日语字幕时，任务模式必须是 'transcribe'。")
            if target_language in {"日语", "英语"} and task_mode == "transcribe":
                if not (provider_base_url and provider_api_key and provider_model):
                    raise RuntimeError(
                        "使用 Provider 翻译时，需要填写 Base URL、API Key 和 Model。"
                    )
        return JobSettings(
            output_dir=output_dir,
            model=self.model_var.get(),
            language=source_language,
            task_mode=task_mode,
            target_language=target_language,
            device=device,
            fp16=bool(self.fp16_var.get() and device == "cuda"),
            temperature=temperature,
            beam_size=beam_size,
            threads=threads,
            ffmpeg_path=ffmpeg_path,
            provider_base_url=provider_base_url,
            provider_api_key=provider_api_key,
            provider_model=provider_model,
            provider_system_prompt=provider_system_prompt,
        )

    def _worker_main(self, mode: str, settings: JobSettings, files: list[Path]) -> None:
        try:
            ffmpeg_command = self.processor.ensure_dependencies(settings)
            provider = TranslationProvider(
                base_url=settings.provider_base_url,
                api_key=settings.provider_api_key,
                model=settings.provider_model,
                system_prompt=settings.provider_system_prompt,
                logger=self.log,
            )
            success_count = 0
            for source_path in files:
                self._queue_status(source_path, "处理中", "")
                try:
                    latest_output = self._process_single_file(
                        source_path,
                        settings,
                        provider,
                        mode,
                        ffmpeg_command,
                    )
                    if mode == "convert":
                        self._queue_replacement(source_path, latest_output)
                    success_count += 1
                    self._queue_status(source_path, "完成", str(latest_output))
                except Exception as exc:  # noqa: BLE001
                    self.log(f"错误 [{source_path.name}] {exc}")
                    self._queue_status(source_path, "失败", str(exc))
            self.log(f"任务完成，成功 {success_count}/{len(files)} 个文件。")
            self.log_queue.put(("finished", f"已完成：成功 {success_count}/{len(files)}"))
        except Exception as exc:  # noqa: BLE001
            self.log(f"严重错误: {exc}")
            self.log_queue.put(("fatal", str(exc)))

    def _process_single_file(
        self,
        source_path: Path,
        settings: JobSettings,
        provider: TranslationProvider,
        mode: str,
        ffmpeg_command: str,
    ) -> Path:
        if not source_path.exists():
            raise RuntimeError(f"未找到源文件: {source_path}")

        work_dir = self.processor.make_work_dir(settings.output_dir, source_path)
        mp3_output_dir = settings.output_dir / "mp3"
        if source_path.suffix.lower() in VIDEO_EXTENSIONS:
            self.log(f"[{source_path.name}] 正在提取音频...")
            audio_path = self.processor.convert_to_mp3(source_path, mp3_output_dir, ffmpeg_command)
        else:
            self.log(f"[{source_path.name}] 直接使用音频文件。")
            audio_path = source_path

        if mode == "convert":
            return audio_path

        self.log(f"[{source_path.name}] 正在生成字幕...")
        final_srt = self._generate_subtitle_for_target(source_path, audio_path, settings, provider, work_dir)
        return final_srt

    def _generate_subtitle_for_target(
        self,
        source_path: Path,
        audio_path: Path,
        settings: JobSettings,
        provider: TranslationProvider,
        work_dir: Path,
    ) -> Path:
        if settings.target_language == "原文":
            raw_srt = self.processor.run_whisper(
                audio_path=audio_path,
                output_dir=work_dir,
                settings=settings,
                task="transcribe",
            )
            destination = self.processor.build_destination_path(
                settings.output_dir,
                source_path,
                "原文",
                settings.language,
            )
            self.log(f"[{source_path.name}] 正在保存原文字幕 -> {destination.name}")
            return self.processor.finalize_output(raw_srt, destination)

        if settings.target_language == "英语" and settings.task_mode == "translate-to-English":
            translated_srt = self.processor.run_whisper(
                audio_path=audio_path,
                output_dir=work_dir,
                settings=settings,
                task="translate",
            )
            destination = self.processor.build_destination_path(
                settings.output_dir,
                source_path,
                "英语",
                settings.language,
            )
            self.log(f"[{source_path.name}] 正在保存英文字幕 -> {destination.name}")
            return self.processor.finalize_output(translated_srt, destination)

        if not provider.is_configured():
            raise RuntimeError("当前目标语言需要已配置好的翻译 Provider。")

        source_srt = self.processor.run_whisper(
            audio_path=audio_path,
            output_dir=work_dir,
            settings=settings,
            task="transcribe",
        )
        destination = self.processor.build_destination_path(
            settings.output_dir,
            source_path,
            settings.target_language,
            settings.language,
        )
        self.log(
            f"[{source_path.name}] 正在翻译为 {settings.target_language} -> {destination.name}"
        )
        return self.processor.translate_srt(
            source_srt=source_srt,
            target_srt=destination,
            target_language=settings.target_language,
            provider=provider,
        )

    def describe_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return "Video"
        if suffix in AUDIO_EXTENSIONS:
            return "Audio"
        return "Unknown"

    def _queue_status(self, source_path: Path, status: str, output: str) -> None:
        self.log_queue.put(("status", source_path, status, output))

    def _queue_replacement(self, old_path: Path, new_path: Path) -> None:
        self.log_queue.put(("replace_file", old_path, new_path))

    def _drain_log_queue(self) -> None:
        while True:
            try:
                event = self.log_queue.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "log":
                self._append_log(event[1])
            elif kind == "status":
                _, source_path, status, output = event
                item_id = self.item_by_path.get(source_path)
                if item_id:
                    self.tree.item(
                        item_id,
                        values=(str(source_path), self.describe_file(source_path), status, output),
                    )
            elif kind == "replace_file":
                _, old_path, new_path = event
                self._replace_file_entry(old_path, new_path)
            elif kind == "finished":
                self._set_running_state(False, event[1])
            elif kind == "fatal":
                self._set_running_state(False, "任务失败")
                messagebox.showerror("严重错误", event[1])
        if self.worker_thread and not self.worker_thread.is_alive():
            if str(self.convert_button.cget("state")) == "disabled":
                if not self.status_var.get().startswith("正在执行"):
                    self._set_running_state(False, self.status_var.get() or "就绪")
            self.worker_thread = None
        self.root.after(150, self._drain_log_queue)

    def log(self, message: str) -> None:
        self.log_queue.put(("log", message))

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _replace_file_entry(self, old_path: Path, new_path: Path) -> None:
        if old_path == new_path:
            return
        item_id = self.item_by_path.pop(old_path, None)
        if not item_id:
            return
        self.item_by_path[new_path] = item_id
        self.selected_files = [new_path if path == old_path else path for path in self.selected_files]
        self.tree.item(
            item_id,
            values=(str(new_path), self.describe_file(new_path), "已替换为 MP3", str(new_path)),
        )
        self.log(f"[{old_path.name}] 已自动替换为 MP3，可直接继续生成字幕。")

    def _set_running_state(self, running: bool, status_text: str) -> None:
        self.status_var.set(status_text)
        state = "disabled" if running else "normal"
        self.convert_button.configure(state=state)
        self.subtitle_button.configure(state=state)

    def _refresh_task_mode_hint(self) -> None:
        task_mode = self.task_mode_var.get()
        target = self.target_language_var.get()
        if target == "原文":
            hint = "原文字幕始终使用 Whisper 的 transcribe。"
        elif target == "日语":
            hint = "日语字幕会先用 Whisper transcribe，再交给翻译 Provider。"
        elif task_mode == "translate-to-English":
            hint = "英文字幕会使用 Whisper 原生 translate 模式。"
        else:
            hint = "当任务模式为 transcribe 时，英文字幕会使用翻译 Provider。"
        self.task_hint_label.configure(text=hint)

    def _mode_label(self, mode: str) -> str:
        if mode == "convert":
            return "音频提取"
        if mode == "subtitle":
            return "字幕生成"
        return mode

    def _language_value(self, selected_label: str) -> str:
        for label, value in LANGUAGE_OPTIONS:
            if label == selected_label:
                return value
        return ""

    @staticmethod
    def _parse_positive_int(value: str, field_name: str) -> int:
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise RuntimeError(f"{field_name} 必须是正整数。") from exc
        if parsed <= 0:
            raise RuntimeError(f"{field_name} 必须大于 0。")
        return parsed

    @staticmethod
    def _parse_non_negative_float(value: str, field_name: str) -> float:
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise RuntimeError(f"{field_name} 必须是数字。") from exc
        if parsed < 0:
            raise RuntimeError(f"{field_name} 不能小于 0。")
        return parsed

    def save_config(self) -> None:
        self.config_store.save(
            {
                "output_dir": self.output_dir_var.get(),
                "model": self.model_var.get(),
                "source_language": self.source_language_var.get(),
                "task_mode": self.task_mode_var.get(),
                "target_language": self.target_language_var.get(),
                "device": self.device_var.get(),
                "fp16": bool(self.fp16_var.get()),
                "temperature": self.temperature_var.get(),
                "beam_size": self.beam_size_var.get(),
                "threads": self.threads_var.get(),
                "ffmpeg_path": self.ffmpeg_path_var.get(),
                "base_url": self.base_url_var.get(),
                "api_key": self.api_key_var.get(),
                "provider_model": self.provider_model_var.get(),
                "provider_system_prompt": self.system_prompt_text.get("1.0", "end").strip(),
            }
        )


def parse_srt(path: Path) -> list[SubtitleSegment]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", content.strip())
    segments: list[SubtitleSegment] = []
    for block in blocks:
        lines = [line.rstrip("\r") for line in block.splitlines()]
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        timing = lines[1].strip()
        text = "\n".join(lines[2:]).strip()
        segments.append(SubtitleSegment(index=index, timing=timing, text=text))
    return segments


def write_srt(path: Path, segments: list[SubtitleSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = []
    for segment in segments:
        blocks.append(f"{segment.index}\n{segment.timing}\n{segment.text}")
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = SubtitleGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, app))
    root.mainloop()


def on_close(root: tk.Tk, app: SubtitleGUI) -> None:
    if app.worker_thread and app.worker_thread.is_alive():
        if not messagebox.askyesno(
            "任务仍在运行",
            "后台任务还在运行，仍然要关闭窗口吗？",
        ):
            return
    app.save_config()
    root.destroy()


if __name__ == "__main__":
    main()
