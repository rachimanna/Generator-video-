"""Точка входа.

Локально: long polling (WEBHOOK_BASE_URL пустой).
На Render: webhook на RENDER_EXTERNAL_URL + /health для проверки живости.
"""

import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

from bot.config import config
from bot.handlers import router

WEBHOOK_PATH = "/telegram/webhook"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def make() -> tuple[Bot, Dispatcher]:
    if not config.bot_token:
        sys.exit("BOT_TOKEN не задан — возьми токен у @BotFather и положи в .env")
    os.makedirs(config.work_dir, exist_ok=True)
    bot = Bot(config.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)
    return bot, dp


async def health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


def run_webhook(bot: Bot, dp: Dispatcher) -> None:
    async def on_startup() -> None:
        await bot.set_webhook(
            config.webhook_base_url + WEBHOOK_PATH,
            secret_token=config.webhook_secret or None,
            drop_pending_updates=True,
        )

    dp.startup.register(on_startup)
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    SimpleRequestHandler(dp, bot, secret_token=config.webhook_secret or None).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    web.run_app(app, host="0.0.0.0", port=config.port)


async def run_polling(bot: Bot, dp: Dispatcher) -> None:
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    bot, dp = make()
    if config.webhook_base_url:
        run_webhook(bot, dp)
    else:
        asyncio.run(run_polling(bot, dp))
