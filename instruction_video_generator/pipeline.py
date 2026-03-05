from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .browser_runner import BrowserAutomationRunner
from .llm_factory import LLMFactory
from .models import GenerationRequest, PipelineResult
from .narration import NarrationService
from .settings import Settings
from .video_editor import VideoEditor


def sanitize_job_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "job"


class InstructionVideoPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        browser_runner: BrowserAutomationRunner | None = None,
        narration_service: NarrationService | None = None,
        video_editor: VideoEditor | None = None,
    ):
        self.settings = settings or Settings.from_env()
        llm_factory = LLMFactory(self.settings)
        self.browser_runner = browser_runner or BrowserAutomationRunner(
            llm_factory,
            ffmpeg_bin=self.settings.ffmpeg_bin,
            ffprobe_bin=self.settings.ffprobe_bin,
            browser_use_api_key=self.settings.browser_use_api_key,
        )
        self.narration_service = narration_service or NarrationService(self.settings)
        self.video_editor = video_editor or VideoEditor(self.settings)

    async def generate(self, request: GenerationRequest) -> PipelineResult:
        request.validate()
        output_dir = request.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        job_name = self._resolve_job_name(request)
        job_dir = output_dir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)

        browser_artifacts = await self.browser_runner.run(request, job_dir)
        raw_source_path = browser_artifacts.video_path
        if abs(request.browser_video_speed - 1.0) > 1e-6:
            sped_path = raw_source_path.with_name(f"{raw_source_path.stem}-x{request.browser_video_speed:.2f}.mp4")
            browser_artifacts.video_path = self.video_editor.speed_adjust_video(
                raw_source_path,
                request.browser_video_speed,
                sped_path,
            )
        raw_video_duration = self.video_editor.probe_duration(browser_artifacts.video_path)
        narration_artifacts = self.narration_service.create_artifacts(
            request=request,
            events=browser_artifacts.events,
            final_result=browser_artifacts.final_result,
            job_dir=job_dir,
            target_duration_seconds=raw_video_duration,
        )
        video_artifacts = self.video_editor.render(
            request=request,
            browser_artifacts=browser_artifacts,
            narration_artifacts=narration_artifacts,
            job_dir=job_dir,
        )
        manifest_path = job_dir / "manifest.json"
        manifest = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "job_name": job_name,
            "instructions": request.instructions,
            "llm_provider": request.llm_provider.value,
            "llm_model": request.llm_model,
            "cloud_mode": request.cloud_mode.value,
            "run_mode": browser_artifacts.run_mode,
            "start_url": request.start_url,
            "browser_video_speed": request.browser_video_speed,
            "browser_video_source_path": str(raw_source_path),
            "browser_video_path": str(browser_artifacts.video_path),
            "agent_history_path": str(browser_artifacts.history_path),
            "action_events_path": str(browser_artifacts.actions_path),
            "narration_script_path": str(narration_artifacts.script_path),
            "narration_audio_path": str(narration_artifacts.audio_path),
            "narration_speed_used": narration_artifacts.speed_used,
            "narration_audio_seconds": narration_artifacts.duration_seconds,
            "raw_video_seconds": raw_video_duration,
            "final_video_path": str(video_artifacts.final_video_path),
            "duration_seconds": video_artifacts.duration_seconds,
            "segments": [asdict(segment) for segment in video_artifacts.segments],
            "ffmpeg_command": video_artifacts.ffmpeg_command,
            "final_result": browser_artifacts.final_result,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return PipelineResult(
            job_dir=job_dir,
            browser_artifacts=browser_artifacts,
            narration_artifacts=narration_artifacts,
            video_artifacts=video_artifacts,
            manifest_path=manifest_path,
        )

    def _resolve_job_name(self, request: GenerationRequest) -> str:
        if request.job_name:
            return sanitize_job_name(request.job_name)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"instruction-video-{stamp}"

    def generate_sync(self, request: GenerationRequest) -> PipelineResult:
        return asyncio.run(self.generate(request))
