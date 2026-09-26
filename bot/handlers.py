from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from . import services
from .config import config
from .ui import Progress

log = logging.getLogger(__name__)
router = Router()

_jobs = asyncio.Semaphore(config.max_parallel_jobs)
_busy: set[int] = set()


@dataclass
class Session:
    face: str | None = None          # сохранённое лицо
    photo: str | None = None         # последнее фото, ждущее выбора действия
    photo_caption: str | None = None
    video: str | None = None         # видео, ждущее фото с лицом


_sessions: dict[int, Session] = {}
_groups: dict[str, list[Message]] = {}


def session(user_id: int) -> Session:
    return _sessions.setdefault(user_id, Session())


START_TEXT = (
    "🎬 <b>Видео-генератор на нейросетях</b>\n\n"
    "Что умею:\n"
    "✍️ <b>Текст → видео</b> — просто напиши, что снять.\n"
    "✨ <b>Оживить фото</b> — пришли фото (в подписи можно описать движение).\n"
    "🎭 <b>Замена лица в видео</b> — пришли фото с лицом и видео "
    "(одним альбомом или по очереди, порядок не важен).\n"
    "🖼 <b>Замена лица на фото</b> — альбом из двух фото: сначала лицо, потом куда его поставить.\n\n"
    "Команды: /video &lt;описание&gt;, /face — сбросить лицо, /cancel — очистить всё.\n\n"
    "⚠️ Используй только свои лица или лица людей, которые дали согласие. "
    "Не выдавай результат за реальную запись."
)


@router.message(CommandStart())
@router.message(Command("help"))
async def start(message: Message) -> None:
    await message.answer(START_TEXT)


@router.message(Command("cancel"))
async def cancel(message: Message) -> None:
    _sessions.pop(message.from_user.id, None)
    await message.answer("🧹 Всё очищено. Можно начинать заново.")


@router.message(Command("face"))
async def reset_face(message: Message) -> None:
    session(message.from_user.id).face = None
    await message.answer("🙂 Лицо сброшено. Пришли новое фото с лицом.")


@router.message(Command("video"))
async def video_cmd(message: Message, command: CommandObject) -> None:
    if not command.args:
        await message.answer("Напиши так: <code>/video кот в скафандре гуляет по Луне</code>")
        return
    await _generate_from_text(message, command.args)


# ---------- запуск задач ----------

async def run_job(
    message: Message,
    title: str,
    work: Callable[[], Awaitable[str]],
    as_photo: bool = False,
    expected: float = 120,
) -> None:
    user_id = message.chat.id
    if user_id in _busy:
        await message.answer("⏳ Подожди, предыдущая генерация ещё идёт.")
        return
    _busy.add(user_id)
    status = await message.answer(f"🕐 <b>{title}</b>\nВ очереди…")
    try:
        async with _jobs:
            async with Progress(status, title, expected):
                result = await work()
        caption = "✅ Готово! Хочешь ещё — просто пришли следующее задание."
        file = FSInputFile(result)
        if as_photo:
            await message.answer_photo(file, caption=caption)
        else:
            await message.answer_video(file, caption=caption, supports_streaming=True)
    except Exception as exc:  # noqa: BLE001
        log.exception("Ошибка генерации")
        await message.answer(
            "😔 Не получилось.\n\n"
            f"<code>{str(exc)[:900]}</code>\n\n"
            "Бесплатные модели иногда перегружены — попробуй через пару минут."
        )
    finally:
        _busy.discard(user_id)


async def _generate_from_text(message: Message, prompt: str) -> None:
    await run_job(message, "Генерирую видео по описанию", lambda: services.text_to_video(prompt), expected=90)


async def _swap_video(message: Message, face: str, video: str) -> None:
    await run_job(
        message, "Меняю лицо в видео",
        lambda: services.swap_face_video(face, video), expected=180,
    )


async def _swap_photo(message: Message, face: str, target: str) -> None:
    await run_job(message, "Меняю лицо на фото", lambda: services.swap_face_image(face, target), as_photo=True, expected=40)


async def _animate(message: Message, photo: str, prompt: str | None) -> None:
    await run_job(message, "Оживляю фото", lambda: services.animate_photo(photo, prompt), expected=120)


# ---------- скачивание ----------

async def download(bot: Bot, message: Message) -> tuple[str, str]:
    """Возвращает (kind, path), kind = photo | video."""
    if message.photo:
        file_id, ext, kind = message.photo[-1].file_id, ".jpg", "photo"
    elif message.video:
        file_id, ext, kind = message.video.file_id, ".mp4", "video"
    elif message.animation:
        file_id, ext, kind = message.animation.file_id, ".mp4", "video"
    elif message.video_note:
        file_id, ext, kind = message.video_note.file_id, ".mp4", "video"
    elif message.document:
        mime = message.document.mime_type or ""
        name = message.document.file_name or ""
        ext = os.path.splitext(name)[1] or (".mp4" if mime.startswith("video") else ".jpg")
        kind = "video" if mime.startswith("video") else "photo"
        file_id = message.document.file_id
    else:
        raise ValueError("неподдерживаемый тип файла")
    path = services.new_path(ext)
    await bot.download(file_id, destination=path)
    return kind, path


async def safe_download(bot: Bot, message: Message) -> tuple[str, str] | None:
    try:
        return await download(bot, message)
    except Exception as exc:  # noqa: BLE001
        log.warning("download failed: %s", exc)
        await message.answer("📦 Не смог скачать файл. Telegram отдаёт ботам файлы только до 20 МБ — пришли покороче/поменьше.")
        return None


# ---------- альбомы: фото + видео одновременно ----------

MEDIA = F.photo | F.video | F.animation | F.video_note | F.document


@router.message(F.media_group_id, MEDIA)
async def album_part(message: Message, bot: Bot) -> None:
    group_id = message.media_group_id
    first = group_id not in _groups
    _groups.setdefault(group_id, []).append(message)
    if not first:
        return
    await asyncio.sleep(1.5)  # ждём остальные части альбома
    parts = sorted(_groups.pop(group_id, []), key=lambda m: m.message_id)
    ack = await message.answer("📥 Получил альбом, скачиваю файлы…")

    files = [f for m in parts if (f := await safe_download(bot, m))]
    photos = [p for k, p in files if k == "photo"]
    videos = [p for k, p in files if k == "video"]
    caption = next((m.caption for m in parts if m.caption), None)
    sess = session(message.from_user.id)
    with contextlib.suppress(Exception):
        await ack.delete()

    if photos and videos:
        sess.face = photos[0]
        await _swap_video(message, photos[0], videos[0])
    elif len(photos) >= 2:
        sess.face = photos[0]
        await _swap_photo(message, photos[0], photos[1])
    elif videos and sess.face:
        await _swap_video(message, sess.face, videos[0])
    elif photos:
        sess.photo, sess.photo_caption = photos[0], caption
        await _ask_photo_action(message, sess)
    else:
        await message.answer("Не понял альбом 🤔 Пришли фото с лицом + видео.")


# ---------- одиночные фото и видео ----------

@router.message(F.video | F.animation | F.video_note | F.document.func(lambda d: (d.mime_type or "").startswith("video")))
async def single_video(message: Message, bot: Bot) -> None:
    got = await safe_download(bot, message)
    if not got:
        return
    sess = session(message.from_user.id)
    if sess.face:
        await _swap_video(message, sess.face, got[1])
    elif sess.photo:
        sess.face, sess.photo = sess.photo, None
        await _swap_video(message, sess.face, got[1])
    else:
        sess.video = got[1]
        await message.answer("🎥 Видео получил! Теперь пришли <b>фото с лицом</b>, которое поставить в это видео.")


@router.message(F.photo | F.document.func(lambda d: (d.mime_type or "").startswith("image")))
async def single_photo(message: Message, bot: Bot) -> None:
    got = await safe_download(bot, message)
    if not got:
        return
    sess = session(message.from_user.id)
    if sess.video:
        video, sess.video = sess.video, None
        sess.face = got[1]
        await _swap_video(message, got[1], video)
        return
    sess.photo, sess.photo_caption = got[1], message.caption
    await _ask_photo_action(message, sess)


async def _ask_photo_action(message: Message, sess: Session) -> None:
    rows = [
        [InlineKeyboardButton(text="🎭 Это лицо для замены", callback_data="photo:face")],
        [InlineKeyboardButton(text="✨ Оживить фото (анимация)", callback_data="photo:animate")],
    ]
    if sess.face and sess.face != sess.photo:
        rows.append([InlineKeyboardButton(text="🔁 Вставить сохранённое лицо сюда", callback_data="photo:swap")])
    await message.answer("📸 Что сделать с этим фото?", reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data.startswith("photo:"))
async def photo_action(call: CallbackQuery) -> None:
    sess = session(call.from_user.id)
    action = call.data.split(":", 1)[1]
    await call.answer()
    if not sess.photo:
        await call.message.edit_text("Фото устарело, пришли его ещё раз 🙏")
        return
    photo, sess.photo = sess.photo, None
    if action == "face":
        sess.face = photo
        await call.message.edit_text(
            "🎭 Лицо запомнил! Теперь пришли <b>видео</b> (или фото), куда его поставить.\n"
            "Сбросить лицо — /face"
        )
    elif action == "animate":
        await call.message.edit_text("✨ Оживляем!")
        await _animate(call.message, photo, sess.photo_caption)
    elif action == "swap" and sess.face:
        await call.message.edit_text("🔁 Ставлю сохранённое лицо на фото")
        await _swap_photo(call.message, sess.face, photo)


# ---------- текст ----------

@router.message(F.text & ~F.text.startswith("/"))
async def text_prompt(message: Message) -> None:
    await _generate_from_text(message, message.text)
