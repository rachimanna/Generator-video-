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


def _say(text: str) -> None:
    live = hf.current_status.get()
    if live:
        live.text = text


async def swap_face_video(face: str, video: str) -> str:
    _say(f"Готовлю видео (до {config.max_video_seconds:g} с, {config.max_video_side}p)")
    video = await to_mp4(video)
    _say("Ищу лицо на фото")
    face = await crop_face(face)
    job = hf.Job(face_image=face, target_video=video, want="video", hints=("swap", "video", "predict", "run", "process"))
    return await polish_video(await hf.run(config.faceswap_video_spaces, job))


async def swap_face_image(face: str, target: str) -> str:
    _say("Ищу лицо на фото")
    face = await crop_face(face)
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
    """Готовим входящее видео для модели: mp4, не длиннее MAX_VIDEO_SECONDS, не больше MAX_VIDEO_SIDE.

    Замена лица идёт покадрово, поэтому длинное/тяжёлое видео на бесплатной модели
    может считаться десятки минут — режем заранее.
    """
    if not shutil.which("ffmpeg"):
        return path
    side = config.max_video_side
    out = new_path(".mp4")
    ok = await _ffmpeg(
        "-i", path, "-t", str(config.max_video_seconds),
        "-vf", f"scale='min({side},iw)':'min({side},ih)':force_original_aspect_ratio=decrease,"
               "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", out,
    )
    return out if ok else path


def _crop_face_sync(path: str) -> str:
    try:
        import cv2
    except ImportError:
        return path
    img = cv2.imread(path)
    if img is None:
        return path
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
    if len(faces) <= 1 and max(img.shape[:2]) <= 1600:
        return path  # одно лицо — фото и так подходит
    if len(faces) == 0:
        return path
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # самое крупное анфас
    h_img, w_img = img.shape[:2]
    pad = int(max(w, h) * 0.8)  # берём с запасом: волосы, подбородок, шея
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
    out = new_path(".jpg")
    cv2.imwrite(out, img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    log.info("Лиц на фото: %d, вырезал самое крупное %s", len(faces), (x0, y0, x1, y1))
    return out


async def crop_face(path: str) -> str:
    """Если на фото коллаж/несколько лиц — оставляем одно, самое крупное анфас.

    Модели замены лица берут первое найденное лицо, и на коллаже это может быть
    профиль или мелкий кадр — тогда результат получается плохим.
    """
    try:
        return await asyncio.to_thread(_crop_face_sync, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("crop_face: %s", exc)
        return path
