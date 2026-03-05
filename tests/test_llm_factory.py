import pytest

from instruction_video_generator.llm_factory import LLMFactory
from instruction_video_generator.models import LLMProvider
from instruction_video_generator.settings import Settings


def test_openai_missing_key_raises():
    factory = LLMFactory(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        factory.create(LLMProvider.OPENAI, None)


def test_anthropic_missing_key_raises():
    factory = LLMFactory(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        factory.create(LLMProvider.ANTHROPIC, None)


def test_gemini_missing_key_raises():
    factory = LLMFactory(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        factory.create(LLMProvider.GEMINI, None)
