import asyncio
import json
from pathlib import Path

from instruction_video_generator.models import (
    ActionEvent,
    BrowserRunArtifacts,
    GenerationRequest,
    LLMProvider,
    NarrationArtifacts,
    VideoEditArtifacts,
    VideoSegment,
)
from instruction_video_generator.pipeline import InstructionVideoPipeline
from instruction_video_generator.settings import Settings


class FakeBrowserRunner:
    async def run(self, request: GenerationRequest, job_dir: Path) -> BrowserRunArtifacts:
        raw_dir = job_dir / "raw_browser_video"
        artifacts_dir = job_dir / "artifacts"
        raw_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        video_path = raw_dir / "recording.mp4"
        video_path.write_bytes(b"fake-video")
        history_path = artifacts_dir / "agent_history.json"
        history_path.write_text("{}", encoding="utf-8")
        actions_path = artifacts_dir / "action_events.json"
        actions_path.write_text("[]", encoding="utf-8")
        events = [
            ActionEvent(
                step_number=1,
                started_at=0.0,
                ended_at=1.0,
                url="https://example.com",
                title="Example",
                action_names=["go_to_url", "click_element_by_index"],
                actions=[],
                success=True,
                extracted_content=[],
                errors=[],
            )
        ]
        return BrowserRunArtifacts(
            video_path=video_path,
            history_path=history_path,
            actions_path=actions_path,
            events=events,
            final_result="Finished",
            run_mode="local",
        )


class FakeNarrationService:
    def create_artifacts(
        self,
        request: GenerationRequest,
        events: list[ActionEvent],
        final_result: str | None,
        job_dir: Path,
        target_duration_seconds: float | None = None,
    ) -> NarrationArtifacts:
        narration_dir = job_dir / "narration"
        narration_dir.mkdir(parents=True, exist_ok=True)
        script_path = narration_dir / "narration_script.txt"
        script_path.write_text("Step one", encoding="utf-8")
        audio_path = narration_dir / "narration.mp3"
        audio_path.write_bytes(b"fake-audio")
        return NarrationArtifacts(
            script_path=script_path,
            audio_path=audio_path,
            script_text="Step one",
            speed_used=1.1,
            duration_seconds=0.9,
        )


class FakeVideoEditor:
    def probe_duration(self, video_path: Path) -> float:
        return 1.0

    def speed_adjust_video(self, video_path: Path, speed_factor: float, output_path: Path) -> Path:
        output_path.write_bytes(video_path.read_bytes())
        return output_path

    def render(
        self,
        request: GenerationRequest,
        browser_artifacts: BrowserRunArtifacts,
        narration_artifacts: NarrationArtifacts,
        job_dir: Path,
    ) -> VideoEditArtifacts:
        final_dir = job_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_video = final_dir / "instructional_video.mp4"
        final_video.write_bytes(b"final-video")
        return VideoEditArtifacts(
            final_video_path=final_video,
            segments=[VideoSegment(start=0.0, end=1.0, is_zoomed=False)],
            duration_seconds=1.0,
            ffmpeg_command=["ffmpeg", "-i", "in", "out"],
        )


def test_pipeline_generates_manifest(tmp_path: Path):
    request = GenerationRequest(
        instructions="Open the docs and click install",
        output_dir=tmp_path,
        llm_provider=LLMProvider.OPENAI,
        headless=True,
        job_name="demo-job",
    )
    pipeline = InstructionVideoPipeline(
        settings=Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        ),
        browser_runner=FakeBrowserRunner(),
        narration_service=FakeNarrationService(),
        video_editor=FakeVideoEditor(),
    )
    result = asyncio.run(pipeline.generate(request))
    assert result.video_artifacts.final_video_path.exists()
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["job_name"] == "demo-job"
    assert manifest["final_video_path"].endswith("instructional_video.mp4")
    assert manifest["llm_provider"] == "openai"
    assert manifest["run_mode"] == "local"
    assert manifest["browser_video_speed"] == 2.0
    assert manifest["browser_video_source_path"].endswith("recording.mp4")
    assert manifest["narration_speed_used"] == 1.1
    assert manifest["narration_audio_seconds"] == 0.9
    assert manifest["raw_video_seconds"] == 1.0
