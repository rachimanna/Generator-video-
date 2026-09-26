"""Универсальный вызов бесплатных моделей на Hugging Face Spaces.

Spaces часто меняют названия эндпоинтов и параметров, поэтому здесь нет
жёстко прописанных сигнатур: мы читаем описание API спейса, сами находим
подходящий эндпоинт и раскладываем по его параметрам промпт / фото / видео.
Остальные параметры остаются со значениями по умолчанию из спейса.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError

from .config import config

log = logging.getLogger(__name__)

VIDEO_EXT = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

SOURCE_WORDS = ("source", "src", "face", "swap", "reference", "ref", "input face", "your face")
TARGET_WORDS = ("target", "dest", "dst", "video", "original", "base")


class SpaceError(RuntimeError):
    pass


@dataclass
class LiveStatus:
    """Реальный статус задачи на спейсе — его показывает прогресс-бар в чате."""

    text: str | None = None
    progress: float | None = None  # 0..1, если модель сообщает прогресс
    eta: float | None = None


current_status: contextvars.ContextVar[LiveStatus | None] = contextvars.ContextVar("current_status", default=None)


def _update_status(space: str, st: Any, live: LiveStatus | None) -> None:
    if live is None or st is None:
        return
    code = getattr(getattr(st, "code", None), "value", "")
    model = space.split("/")[-1]
    live.eta = getattr(st, "eta", None)
    if code in ("STARTING", "JOINING_QUEUE"):
        live.text = f"Подключаюсь к модели {model}"
    elif code == "IN_QUEUE":
        rank, size = getattr(st, "rank", None), getattr(st, "queue_size", None)
        live.text = f"Очередь на {model}: " + (f"{rank + 1} из {size}" if rank is not None and size else "ждём")
    elif code == "QUEUE_FULL":
        live.text = f"Очередь {model} переполнена"
    elif code == "SENDING_DATA":
        live.text = f"Отправляю файлы в {model}"
    elif code in ("PROCESSING", "ITERATING", "PROGRESS"):
        live.text = f"{model} обрабатывает"
        units = getattr(st, "progress_data", None) or []
        if units:
            u = units[-1]
            if u.progress is not None:
                live.progress = float(u.progress)
            elif u.index is not None and u.length:
                live.progress = u.index / u.length
            if u.desc:
                live.text = f"{model}: {u.desc}"


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


def _callable_names(client: Client) -> set[str] | None:
    """Эндпоинты, которые реально можно вызвать (view_api иногда показывает лишние)."""
    try:
        deps = client.config.get("dependencies", [])
    except Exception:  # noqa: BLE001
        return None
    names = set()
    for d in deps:
        name = d.get("api_name")
        if not name or d.get("api_visibility") == "private" or d.get("show_api") is False:
            continue
        names.add("/" + name)
    return names


def _pick_endpoints(client: Client, job: Job, api_name: str | None) -> list[tuple[str, list[dict]]]:
    """Возвращает подходящие эндпоинты, лучший первым."""
    api = client.view_api(print_info=False, return_format="dict") or {}
    endpoints: dict = api.get("named_endpoints", {})
    callable_names = _callable_names(client)
    if callable_names:
        hidden = [n for n in endpoints if n not in callable_names]
        if hidden:
            log.info("Пропускаю невызываемые эндпоинты: %s", hidden)
        endpoints = {n: i for n, i in endpoints.items() if n in callable_names}
    if not endpoints:
        raise SpaceError("у спейса нет публичного API")
    if api_name:
        if api_name not in endpoints:
            raise SpaceError(f"эндпоинт {api_name} не найден, есть: {', '.join(endpoints)}")
        return [(api_name, endpoints[api_name].get("parameters", []))]

    scored = [(s, n) for n, i in endpoints.items() if (s := _score(n, i, job)) is not None]
    if not scored:
        raise SpaceError(f"не нашёл подходящий эндпоинт среди: {', '.join(endpoints)}")
    scored.sort(reverse=True)
    log.info("Кандидаты: %s", scored[:4])
    return [(n, endpoints[n].get("parameters", [])) for _, n in scored[:3]]


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
    errors = []
    for endpoint, params in _pick_endpoints(client, job, api_name or None):
        kwargs = _build_kwargs(params, job)
        log.info("%s%s ← %s", space, endpoint, list(kwargs))
        try:
            result = _submit_and_wait(client, space, endpoint, kwargs)
        except AppError:
            raise  # сама модель упала (квота, ошибка внутри) — перебор эндпоинтов не поможет
        except (ValueError, TypeError) as exc:
            # эндпоинт не принял вызов (не тот API / не те параметры) — пробуем следующий
            log.warning("%s%s: %s", space, endpoint, exc)
            errors.append(f"{endpoint}: {exc}")
            continue
        return _pick_output(result, job.want)
    raise SpaceError("; ".join(errors))


def _submit_and_wait(client: Client, space: str, endpoint: str, kwargs: dict) -> Any:
    live = current_status.get()
    job = client.submit(api_name=endpoint, **kwargs)
    deadline = time.monotonic() + config.job_timeout
    while not job.done():
        if time.monotonic() > deadline:
            job.cancel()
            raise SpaceError(f"модель не ответила за {config.job_timeout} с (JOB_TIMEOUT)")
        try:
            _update_status(space, job.status(), live)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    return job.result()


async def run(spaces: list[str], job: Job) -> str:
    """Пробует спейсы по очереди, возвращает путь к готовому файлу."""
    errors = []
    live = current_status.get()
    for spec in spaces:
        if live:
            live.text, live.progress, live.eta = f"Подключаюсь к {spec.split('@')[0]}", None, None
        try:
            return await asyncio.to_thread(_run_space, spec, job)
        except Exception as exc:  # noqa: BLE001 — любой сбой спейса = пробуем следующий
            log.warning("Спейс %s не сработал: %s", spec, exc)
            with _clients_lock:
                _clients.pop(spec.partition("@")[0], None)
            errors.append(f"• {spec}: {str(exc)[:200]}")
    raise SpaceError("Все модели сейчас недоступны:\n" + "\n".join(errors))
