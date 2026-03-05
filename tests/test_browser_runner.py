from pathlib import Path

import pytest

from instruction_video_generator.browser_runner import BrowserAutomationRunner
from instruction_video_generator.models import ActionEvent, BrowserCloudMode, GenerationRequest
from instruction_video_generator.settings import Settings
from instruction_video_generator.llm_factory import LLMFactory


def _request(instructions: str, cloud_mode: BrowserCloudMode = BrowserCloudMode.CLOUD) -> GenerationRequest:
    return GenerationRequest(
        instructions=instructions,
        output_dir=Path("."),
        cloud_mode=cloud_mode,
    )


def _runner(with_key: bool) -> BrowserAutomationRunner:
    settings = Settings(
        openai_api_key="test",
        anthropic_api_key=None,
        gemini_api_key=None,
        browser_use_api_key="bu_test" if with_key else None,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
    )
    return BrowserAutomationRunner(
        LLMFactory(settings),
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
        browser_use_api_key=settings.browser_use_api_key,
    )


def test_cloud_mode_uses_cloud_with_key():
    runner = _runner(with_key=True)
    request = _request("Open canva.com and create a slide", cloud_mode=BrowserCloudMode.CLOUD)
    assert runner._resolve_initial_mode(request) == "cloud"


def test_cloud_mode_requires_api_key():
    runner = _runner(with_key=False)
    request = _request("Open browser-use.com docs", cloud_mode=BrowserCloudMode.CLOUD)
    with pytest.raises(ValueError, match="BROWSER_USE_API_KEY is required when --cloud-mode cloud"):
        runner._resolve_initial_mode(request)


def test_local_mode_forced_even_with_key():
    runner = _runner(with_key=True)
    request = _request("Open canva.com", cloud_mode=BrowserCloudMode.LOCAL)
    assert runner._resolve_initial_mode(request) == "local"


def test_auto_mode_uses_cloud_for_canva_with_key():
    runner = _runner(with_key=True)
    request = _request("Open canva.com and create a slide", cloud_mode=BrowserCloudMode.AUTO)
    assert runner._resolve_initial_mode(request) == "cloud"


def test_auto_mode_uses_local_for_generic_site():
    runner = _runner(with_key=True)
    request = _request("Open example.com and click more info", cloud_mode=BrowserCloudMode.AUTO)
    assert runner._resolve_initial_mode(request) == "local"


def test_challenge_block_detection():
    runner = _runner(with_key=True)
    events = [
        ActionEvent(
            step_number=1,
            started_at=0.0,
            ended_at=1.0,
            url="https://www.canva.com/",
            title="Checking your browser before accessing",
            action_names=["navigate"],
            actions=[],
            success=False,
            extracted_content=[],
            errors=[],
        ),
        ActionEvent(
            step_number=2,
            started_at=1.0,
            ended_at=2.0,
            url="https://www.canva.com/",
            title="Attention Required! | Cloudflare",
            action_names=["wait"],
            actions=[],
            success=False,
            extracted_content=[],
            errors=[],
        ),
        ActionEvent(
            step_number=3,
            started_at=2.0,
            ended_at=3.0,
            url="https://www.canva.com/",
            title="Verify you are human",
            action_names=["navigate"],
            actions=[],
            success=False,
            extracted_content=[],
            errors=[],
        ),
    ]
    assert runner._has_challenge_block(events, threshold=3) is True


def test_build_task_how_to_prefers_in_app_demo():
    runner = _runner(with_key=True)
    request = _request("How do I change Google Doc layout?")
    task = runner._build_task(request)
    assert "how-to request" in task
    assert "Do not fulfill the request by only visiting help/support/documentation pages." in task
    assert "Show exact clicks/menus/settings directly in the product whenever possible." in task


def test_build_task_docs_request_does_not_block_docs_pages():
    runner = _runner(with_key=True)
    request = _request("Show me a Google Docs help page for page setup")
    task = runner._build_task(request)
    assert "Do not fulfill the request by only visiting help/support/documentation pages." not in task


def test_build_task_google_docs_layout_playbook():
    runner = _runner(with_key=True)
    request = _request("Show me how to change google doc layout")
    task = runner._build_task(request)
    assert "For Google Docs layout requests" in task
    assert "File > Page setup." in task
