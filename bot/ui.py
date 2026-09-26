"""Анимированный статус «идёт генерация» в сообщении Telegram."""

from __future__ import annotations

import asyncio
import contextlib
import html
import time

from aiogram.types import Message

from .hf import LiveStatus, current_status

FRAMES = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
STAGES = [
    "Загружаю в нейросеть",
    "Нейросеть разглядывает кадры",
    "Рисую движение",
    "Шлифую детали",
    "Навожу красоту",
    "Почти готово",
]
BAR = 12


def _fmt(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60} мин {sec % 60:02d} с" if sec >= 60 else f"{sec} с"


class Progress:
    def __init__(self, message: Message, title: str, expected: float = 120):
        self.message = message
        self.title = title
        self.expected = expected
        self._task: asyncio.Task | None = None
        self._start = time.monotonic()
        self.live = LiveStatus()

    def _render(self, tick: int) -> str:
        elapsed = time.monotonic() - self._start
        live = self.live
        if live.progress is not None:
            ratio = max(0.0, min(live.progress, 1.0))  # настоящий прогресс от модели
        else:
            # модель прогресс не сообщает — плавно ползём, но не упираемся в «почти готово»
            ratio = 0.9 * (1 - 0.5 ** (elapsed / self.expected))
        filled = int(ratio * BAR)
        bar = "▰" * filled + "▱" * (BAR - filled)
        stage = html.escape(live.text) if live.text else STAGES[min(int(ratio * len(STAGES)), len(STAGES) - 1)]
        eta = f" · осталось ~{int(live.eta)} с" if live.eta else ""
        return (
            f"{FRAMES[tick % len(FRAMES)]} <b>{self.title}</b>\n\n"
            f"{bar} {int(ratio * 100)}%\n"
            f"<i>{stage}{'.' * (tick % 3 + 1)}</i>\n\n"
            f"⏱ {_fmt(elapsed)}{eta}"
        )

    async def _loop(self) -> None:
        tick = 0
        while True:
            with contextlib.suppress(Exception):
                await self.message.edit_text(self._render(tick))
            tick += 1
            await asyncio.sleep(4)

    async def __aenter__(self) -> "Progress":
        current_status.set(self.live)
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        with contextlib.suppress(Exception):
            await self.message.delete()
