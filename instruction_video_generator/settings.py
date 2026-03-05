from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True, frozen=True)
class Settings:
    openai_api_key: str | None
    anthropic_api_key: str | None
    gemini_api_key: str | None
    browser_use_api_key: str | None
    ffmpeg_bin: str
    ffprobe_bin: str

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        if env_file is None:
            load_dotenv(override=False)
        else:
            load_dotenv(dotenv_path=env_file, override=False)
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            browser_use_api_key=os.getenv("BROWSER_USE_API_KEY"),
            ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
            ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
        )
