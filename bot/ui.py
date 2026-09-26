"""Анимированный статус «идёт генерация» в сообщении Telegram."""

from __future__ import annotations

import asyncio
import contextlib
import time

from aiogram.types import Message

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


class Progress:
    def __init__(self, message: Message, title: str, expected: float = 120):
        self.message = message
        self.title = title
        self.expected = expected
        self._task: asyncio.Task | None = None
        self._start = time.monotonic()

    def _render(self, tick: int) -> str:
        elapsed = time.monotonic() - self._start
        ratio = min(elapsed / self.expected, 0.95)
        filled = int(ratio * BAR)
        bar = "▰" * filled + "▱" * (BAR - filled)
        stage = STAGES[min(int(ratio * len(STAGES)), len(STAGES) - 1)]
        return (
            f"{FRAMES[tick % len(FRAMES)]} <b>{self.title}</b>\n\n"
            f"{bar} {int(ratio * 100)}%\n"
            f"<i>{stage}{'.' * (tick % 3 + 1)}</i>\n\n"
            f"⏱ {int(elapsed)} с"
        )

    async def _loop(self) -> None:
        tick = 0
        while True:
            with contextlib.suppress(Exception):
                await self.message.edit_text(self._render(tick))
            tick += 1
            await asyncio.sleep(4)

    async def __aenter__(self) -> "Progress":
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        with contextlib.suppress(Exception):
            await self.message.delete()
