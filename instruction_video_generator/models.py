from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class BrowserCloudMode(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass(slots=True)
class GenerationRequest:
    instructions: str
    output_dir: Path
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str | None = None
    cloud_mode: BrowserCloudMode = BrowserCloudMode.CLOUD
    cloud_proxy_country_code: str = "us"
    cloud_proxy_retry_on_challenge: bool = True
    cloud_proxy_retry_countries: tuple[str, ...] = ("us", "gb", "de", "fr")
    cloud_custom_proxy_url: str | None = None
    cloud_custom_proxy_username: str | None = None
    cloud_custom_proxy_password: str | None = None
    cloud_profile_id: str | None = None
    cloud_fallback_on_challenge: bool = True
    challenge_stuck_threshold: int = 3
    use_system_chrome: bool = False
    local_user_data_dir: Path | None = None
    local_profile_directory: str | None = None
    start_url: str | None = None
    max_steps: int = 40
    max_actions_per_step: int = 5
    step_timeout: int = 180
    headless: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    browser_video_speed: float = 2.0
    narration_voice: str = "alloy"
    narration_model: str = "gpt-4o-mini-tts"
    narration_speed: float = 1.0
    narration_min_speed: float = 1.0
    narration_max_speed: float = 2.5
    narration_max_chars: int = 420
    min_raw_video_seconds: float = 10.0
    auto_zoom_level: float = 1.8
    auto_zoom_dwell_threshold: float = 0.5
    auto_zoom_transition_duration: float = 0.4
    job_name: str | None = None

    def validate(self) -> None:
        if not self.instructions.strip():
            raise ValueError("instructions must not be empty")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_actions_per_step < 1:
            raise ValueError("max_actions_per_step must be >= 1")
        if self.challenge_stuck_threshold < 1:
            raise ValueError("challenge_stuck_threshold must be >= 1")
        proxy_country = self.cloud_proxy_country_code.strip().lower()
        if len(proxy_country) != 2 or not proxy_country.isalpha():
            raise ValueError("cloud_proxy_country_code must be a 2-letter country code")
        if not self.cloud_proxy_retry_countries:
            raise ValueError("cloud_proxy_retry_countries must not be empty")
        for value in self.cloud_proxy_retry_countries:
            code = value.strip().lower()
            if len(code) != 2 or not code.isalpha():
                raise ValueError("cloud_proxy_retry_countries must contain 2-letter country codes")
        if self.cloud_custom_proxy_username and not self.cloud_custom_proxy_url:
            raise ValueError("cloud_custom_proxy_url is required when cloud_custom_proxy_username is set")
        if self.cloud_custom_proxy_password and not self.cloud_custom_proxy_url:
            raise ValueError("cloud_custom_proxy_url is required when cloud_custom_proxy_password is set")
        if self.cloud_custom_proxy_url is not None and not self.cloud_custom_proxy_url.strip():
            raise ValueError("cloud_custom_proxy_url must not be empty when provided")
        if self.local_profile_directory is not None and not self.local_profile_directory.strip():
            raise ValueError("local_profile_directory must not be empty when provided")
        if self.viewport_width < 320 or self.viewport_height < 240:
            raise ValueError("viewport dimensions are too small")
        if self.browser_video_speed <= 0:
            raise ValueError("browser_video_speed must be > 0")
        if self.narration_speed <= 0:
            raise ValueError("narration_speed must be > 0")
        if self.narration_min_speed < 1.0:
            raise ValueError("narration_min_speed must be >= 1.0")
        if self.narration_max_speed <= 0:
            raise ValueError("narration_max_speed must be > 0")
        if self.narration_min_speed > self.narration_max_speed:
            raise ValueError("narration_min_speed must be <= narration_max_speed")
        if self.narration_max_chars < 120:
            raise ValueError("narration_max_chars must be >= 120")
        if self.min_raw_video_seconds < 0:
            raise ValueError("min_raw_video_seconds must be >= 0")
        if self.auto_zoom_level < 1.0:
            raise ValueError("auto_zoom_level must be >= 1.0")
        if self.auto_zoom_dwell_threshold < 0:
            raise ValueError("auto_zoom_dwell_threshold must be >= 0")
        if self.auto_zoom_transition_duration < 0:
            raise ValueError("auto_zoom_transition_duration must be >= 0")


@dataclass(slots=True)
class ActionEvent:
    step_number: int
    started_at: float | None
    ended_at: float | None
    url: str | None
    title: str | None
    action_names: list[str]
    actions: list[dict[str, Any]]
    success: bool | None
    extracted_content: list[str]
    errors: list[str]
    center_x: float | None = None
    center_y: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BrowserRunArtifacts:
    video_path: Path
    history_path: Path
    actions_path: Path
    events: list[ActionEvent]
    final_result: str | None
    run_mode: str


@dataclass(slots=True)
class NarrationArtifacts:
    script_path: Path
    audio_path: Path
    script_text: str
    speed_used: float = 1.0
    duration_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class VideoSegment:
    start: float
    end: float
    is_zoomed: bool
    phase: str = "normal"
    center_x: float = 0.5
    center_y: float = 0.5
    zoom_level: float = 1.0


@dataclass(slots=True)
class VideoEditArtifacts:
    final_video_path: Path
    segments: list[VideoSegment]
    duration_seconds: float
    ffmpeg_command: list[str]


@dataclass(slots=True)
class PipelineResult:
    job_dir: Path
    browser_artifacts: BrowserRunArtifacts
    narration_artifacts: NarrationArtifacts
    video_artifacts: VideoEditArtifacts
    manifest_path: Path
