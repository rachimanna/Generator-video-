import hashlib
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _list(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Config:
    bot_token: str = os.getenv("BOT_TOKEN", "")
    hf_token: str | None = os.getenv("HF_TOKEN") or None

    text2video_spaces: list[str] = field(
        default_factory=lambda: _list("TEXT2VIDEO_SPACES", "Lightricks/ltx-video-distilled,Wan-AI/Wan2.1")
    )
    img2video_spaces: list[str] = field(
        default_factory=lambda: _list("IMG2VIDEO_SPACES", "Lightricks/ltx-video-distilled")
    )
    faceswap_video_spaces: list[str] = field(
        default_factory=lambda: _list("FACESWAP_VIDEO_SPACES", "tonyassi/video-face-swap")
    )
    faceswap_image_spaces: list[str] = field(
        default_factory=lambda: _list("FACESWAP_IMAGE_SPACES", "tonyassi/face-swap,felixrosberg/face-swap")
    )

    fade_seconds: float = float(os.getenv("FADE_SECONDS", "0.6"))
    job_timeout: int = int(os.getenv("JOB_TIMEOUT", "300"))  # сек. на одну модель
    auto_discover: bool = os.getenv("AUTO_DISCOVER", "1") not in ("0", "false", "no")
    max_model_tries: int = int(os.getenv("MAX_MODEL_TRIES", "4"))
    max_video_seconds: float = float(os.getenv("MAX_VIDEO_SECONDS", "6"))  # длиннее — обрежем
    max_video_side: int = int(os.getenv("MAX_VIDEO_SIDE", "540"))
    video_fps: int = int(os.getenv("VIDEO_FPS", "15"))
    max_parallel_jobs: int = int(os.getenv("MAX_PARALLEL_JOBS", "2"))

    webhook_base_url: str = (os.getenv("WEBHOOK_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")
    # секрет для проверки, что запросы на webhook идут от Telegram (только A-Z a-z 0-9 _ -)
    webhook_secret: str = os.getenv("WEBHOOK_SECRET") or hashlib.sha256(
        os.getenv("BOT_TOKEN", "").encode()
    ).hexdigest()[:32]
    port: int = int(os.getenv("PORT", "8080"))

    work_dir: str = os.getenv("WORK_DIR", "/tmp/videobot")


config = Config()
