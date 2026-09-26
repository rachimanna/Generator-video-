"""Универсальный вызов бесплатных моделей на Hugging Face Spaces.

Spaces часто меняют названия эндпоинтов и параметров, поэтому здесь нет
жёстко прописанных сигнатур: мы читаем описание API спейса, сами находим
подходящий эндпоинт и раскладываем по его параметрам промпт / фото / видео.
Остальные параметры остаются со значениями по умолчанию из спейса.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

from gradio_client import Client, handle_file

from .config import config

log = logging.getLogger(__name__)

VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

SOURCE_WORDS = ("source", "src", "face", "swap", "reference", "ref", "input face", "your face")
TARGET_WORDS = ("target", "dest", "dst", "video", "original", "base")


class SpaceError(RuntimeError):
    pass


@dataclass
class Job:
    """Что нужно отдать модели и что ожидаем получить."""

    prompt: str | None = None
    negative_prompt: str | None = None
    face_image: str | None = None      # фото с лицом (источник)
    target_image: str | None = None    # фото, на котором меняем лицо / кадр для анимации
    target_video: str | None = None    # видео, в котором меняем лицо
    want: str = "video"                # "video" | "image"
    hints: tuple[str, ...] = field(default_factory=tuple)  # слова, по которым выбираем эндпоинт

    @property
    def images(self) -> list[str]:
        return [p for p in (self.face_image, self.target_image) if p]


_clients: dict[str, Client] = {}
_clients_lock = threading.Lock()


def _client(space: str) -> Client:
    with _clients_lock:
        if space not in _clients:
            log.info("Подключаюсь к спейсу %s", space)
            _clients[space] = Client(
                space,
                token=config.hf_token,
                verbose=False,
                download_files=os.path.join(config.work_dir, "hf"),
            )
        return _clients[space]


def _text(p: dict) -> str:
    return f"{p.get('parameter_name') or ''} {p.get('label') or ''}".lower()


def _is_video(p: dict) -> bool:
    return (p.get("component") or "").lower() == "video"


def _is_image(p: dict) -> bool:
    return (p.get("component") or "").lower() == "image"


def _is_text(p: dict) -> bool:
    return (p.get("component") or "").lower() in ("textbox", "text")


def _score(name: str, info: dict, job: Job) -> float | None:
    params = info.get("parameters", [])
    n_img = sum(_is_image(p) for p in params)
    n_vid = sum(_is_video(p) for p in params)
    n_txt = sum(_is_text(p) for p in params)

    if len(job.images) > n_img or (job.target_video and n_vid < 1) or (job.prompt and n_txt < 1 and not job.images):
        return None

    score = 0.0
    lname = name.lower()
    for hint in job.hints:
        if hint in lname:
            score += 5
    for bad in ("load", "example", "clear", "reset", "upload", "change", "update", "toggle", "lambda", "random", "seed"):
        if bad in lname:
            score -= 4
    # необязательные параметры без значения по умолчанию, которые мы не сможем заполнить
    required_img = sum(_is_image(p) and not p.get("parameter_has_default") for p in params)
    required_vid = sum(_is_video(p) and not p.get("parameter_has_default") for p in params)
    if required_img > len(job.images) or required_vid > (1 if job.target_video else 0):
        score -= 10
    if job.prompt and n_txt:
        score += 2
    if job.images and n_img:
        score += 2
    if job.target_video and n_vid:
        score += 3
    returns = [r.get("component", "").lower() for r in info.get("returns", [])]
    if job.want in returns:
        score += 3
    return score


def _pick_endpoint(client: Client, job: Job, api_name: str | None) -> tuple[str, list[dict]]:
    api = client.view_api(print_info=False, return_format="dict") or {}
    endpoints: dict = api.get("named_endpoints", {})
    if not endpoints:
        raise SpaceError("у спейса нет публичного API")
    if api_name:
        if api_name not in endpoints:
            raise SpaceError(f"эндпоинт {api_name} не найден, есть: {', '.join(endpoints)}")
        return api_name, endpoints[api_name].get("parameters", [])

    scored = [(s, n) for n, i in endpoints.items() if (s := _score(n, i, job)) is not None]
    if not scored:
        raise SpaceError(f"не нашёл подходящий эндпоинт среди: {', '.join(endpoints)}")
    scored.sort(reverse=True)
    name = scored[0][1]
    log.info("Выбран эндпоинт %s (кандидаты: %s)", name, scored[:4])
    return name, endpoints[name].get("parameters", [])


def _file_arg(path: str, p: dict) -> Any:
    wrapped = handle_file(path)
    ptype = str((p.get("python_type") or {}).get("type", "")).lower()
    if _is_video(p) and "dict" in ptype:
        return {"video": wrapped}
    return wrapped


def _build_kwargs(params: list[dict], job: Job) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    image_params = [p for p in params if _is_image(p)]
    images_left: dict[str, str] = {}
    if job.face_image:
        images_left["face"] = job.face_image
    if job.target_image:
        images_left["target"] = job.target_image

    # 1) раскладываем фото по смыслу названия параметра
    assigned: dict[int, str] = {}
    for idx, p in enumerate(image_params):
        t = _text(p)
        if "face" in images_left and any(w in t for w in SOURCE_WORDS) and not any(w in t for w in ("target", "dest")):
            assigned[idx] = images_left.pop("face")
        elif "target" in images_left and any(w in t for w in TARGET_WORDS):
            assigned[idx] = images_left.pop("target")
    # 2) остаток — по порядку: сначала лицо, потом цель
    order = [images_left[k] for k in ("face", "target") if k in images_left]
    for idx, _ in enumerate(image_params):
        if idx not in assigned and order:
            assigned[idx] = order.pop(0)

    prompt_used = False
    video_used = False
    for p in params:
        name = p.get("parameter_name")
        if not name:
            continue
        if _is_image(p):
            idx = image_params.index(p)
            if idx in assigned:
                kwargs[name] = _file_arg(assigned[idx], p)
        elif _is_video(p) and job.target_video and not video_used:
            kwargs[name] = _file_arg(job.target_video, p)
            video_used = True
        elif _is_text(p):
            t = _text(p)
            if "negative" in t:
                if job.negative_prompt:
                    kwargs[name] = job.negative_prompt
            elif job.prompt and not prompt_used:
                kwargs[name] = job.prompt
                prompt_used = True
        elif not p.get("parameter_has_default"):
            kwargs[name] = p.get("parameter_default")
    return kwargs


def _collect_files(obj: Any, out: list[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        if os.path.isfile(obj):
            out.append(obj)
    elif isinstance(obj, dict):
        for key in ("video", "path", "value", "name", "image"):
            if key in obj:
                _collect_files(obj[key], out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_files(item, out)


def _pick_output(result: Any, want: str) -> str:
    files: list[str] = []
    _collect_files(result, files)
    ext = VIDEO_EXT if want == "video" else IMAGE_EXT
    for f in files:
        if f.lower().endswith(ext):
            return f
    if files:
        return files[0]
    raise SpaceError(f"модель не вернула файл (ответ: {str(result)[:200]})")


def _run_space(spec: str, job: Job) -> str:
    space, _, api_name = spec.partition("@")
    client = _client(space)
    endpoint, params = _pick_endpoint(client, job, api_name or None)
    kwargs = _build_kwargs(params, job)
    log.info("%s%s ← %s", space, endpoint, list(kwargs))
    result = client.predict(api_name=endpoint, **kwargs)
    return _pick_output(result, job.want)


async def run(spaces: list[str], job: Job) -> str:
    """Пробует спейсы по очереди, возвращает путь к готовому файлу."""
    errors = []
    for spec in spaces:
        try:
            return await asyncio.to_thread(_run_space, spec, job)
        except Exception as exc:  # noqa: BLE001 — любой сбой спейса = пробуем следующий
            log.warning("Спейс %s не сработал: %s", spec, exc)
            with _clients_lock:
                _clients.pop(spec.partition("@")[0], None)
            errors.append(f"• {spec}: {str(exc)[:200]}")
    raise SpaceError("Все модели сейчас недоступны:\n" + "\n".join(errors))
