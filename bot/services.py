"""Сценарии бота поверх бесплатных моделей + постобработка через ffmpeg."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid

from . import hf
from .config import config

log = logging.getLogger(__name__)

NEGATIVE = "worst quality, blurry, jittery, distorted, deformed face, watermark, text"
QUALITY_SUFFIX = ", cinematic, highly detailed, smooth motion, 4k"


def enhance_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    return prompt if len(prompt) > 200 else prompt + QUALITY_SUFFIX


async def text_to_video(prompt: str) -> str:
    job = hf.Job(
        prompt=enhance_prompt(prompt),
        negative_prompt=NEGATIVE,
        want="video",
        hints=("text_to_video", "t2v", "text", "generate", "infer"),
    )
    return await polish_video(await hf.run(config.text2video_spaces, job))


async def animate_photo(photo: str, prompt: str | None) -> str:
    job = hf.Job(
        prompt=enhance_prompt(prompt or "the person comes alive, gentle natural movement, subtle smile, camera slowly pushes in"),
        negative_prompt=NEGATIVE,
        target_image=photo,
        want="video",
        hints=("image_to_video", "i2v", "image", "animate", "generate"),
    )
    return await polish_video(await hf.run(config.img2video_spaces, job))


async def swap_face_video(face: str, video: str) -> str:
    job = hf.Job(face_image=face, target_video=video, want="video", hints=("swap", "video", "predict", "run", "process"))
    return await polish_video(await hf.run(config.faceswap_video_spaces, job))


async def swap_face_image(face: str, target: str) -> str:
    job = hf.Job(face_image=face, target_image=target, want="image", hints=("swap", "predict", "run", "process"))
    return await hf.run(config.faceswap_image_spaces, job)


def new_path(ext: str) -> str:
    os.makedirs(config.work_dir, exist_ok=True)
    return os.path.join(config.work_dir, f"{uuid.uuid4().hex}{ext}")


async def _ffmpeg(*args: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-loglevel", "error", *args,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        log.warning("ffmpeg: %s", err.decode(errors="ignore")[-500:])
    return proc.returncode == 0


async def _duration(path: str) -> float:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    out, _ = await proc.communicate()
    try:
        return float(out.decode().strip())
    except ValueError:
        return 0.0


async def polish_video(path: str) -> str:
    """Красивый финал: плавное появление/затухание, H.264 для Telegram, быстрый старт."""
    if not shutil.which("ffmpeg"):
        return path
    out = new_path(".mp4")
    fade = config.fade_seconds
    dur = await _duration(path) if fade > 0 else 0
    vf = ["scale=trunc(iw/2)*2:trunc(ih/2)*2", "format=yuv420p"]
    if fade > 0 and dur > fade * 3:
        vf.insert(0, f"fade=t=in:st=0:d={fade},fade=t=out:st={dur - fade:.2f}:d={fade}")
    ok = await _ffmpeg(
        "-i", path, "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out,
    )
    return out if ok else path


async def to_mp4(path: str) -> str:
    """Приводим входящее видео (кружочки, gif, webm) к mp4 для моделей."""
    if not shutil.which("ffmpeg") or path.lower().endswith(".mp4"):
        return path
    out = new_path(".mp4")
    ok = await _ffmpeg("-i", path, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", out)
    return out if ok else path
