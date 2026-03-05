from __future__ import annotations

from browser_use.llm import ChatAnthropic, ChatGoogle, ChatOpenAI

from .models import LLMProvider
from .settings import Settings


class LLMFactory:
    def __init__(self, settings: Settings):
        self.settings = settings

    def create(self, provider: LLMProvider, model: str | None = None):
        if provider == LLMProvider.OPENAI:
            api_key = self.settings.openai_api_key
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for provider=openai")
            return ChatOpenAI(model=model or "gpt-4.1-mini", api_key=api_key)
        if provider == LLMProvider.ANTHROPIC:
            api_key = self.settings.anthropic_api_key
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for provider=anthropic")
            return ChatAnthropic(model=model or "claude-sonnet-4-6", api_key=api_key)
        if provider == LLMProvider.GEMINI:
            api_key = self.settings.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for provider=gemini")
            return ChatGoogle(model=model or "gemini-2.5-flash", api_key=api_key)
        raise ValueError(f"Unsupported provider: {provider}")
