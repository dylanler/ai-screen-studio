from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .models import BrowserCloudMode, GenerationRequest, LLMProvider
from .pipeline import InstructionVideoPipeline, sanitize_job_name


@dataclass(slots=True)
class QueueStep:
    step_id: str
    label: str
    description: str
    status: str = "pending"  # pending | active | done | error
    detail: str = ""


@dataclass(slots=True)
class JobState:
    job_id: str
    prompt: str
    status: str = "queued"  # queued | running | completed | failed
    queue: list[QueueStep] = field(default_factory=list)
    current_step: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    error: str | None = None
    message: str = "Queued for generation"
    result: dict[str, Any] | None = None


QUEUE_TEMPLATE: list[tuple[str, str, str]] = [
    ("queued", "Queued", "Request accepted and prepared"),
    ("browser", "Browser Run", "Executing browser automation and capture"),
    ("narration", "Narration", "Writing and synthesizing the voiceover"),
    ("render", "Render", "Compositing and exporting final video"),
    ("done", "Completed", "Instructional video is ready"),
]

DEFAULT_CLOUD_PROFILE_ID = "536cd6ff-add0-4b96-a4e7-c8794254a4cc"

JOBS: dict[str, JobState] = {}
JOBS_LOCK = asyncio.Lock()


class CreateJobRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2000)
    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-sonnet-4-6")
    cloud_mode: str = Field(default="cloud")
    cloud_profile_id: str | None = Field(default=None)
    cloud_proxy_country_code: str = Field(default="us")
    browser_video_speed: float = Field(default=2.0, gt=0)
    narration_speed: float = Field(default=1.0, gt=0)
    narration_min_speed: float = Field(default=1.0, ge=1.0)
    narration_max_speed: float = Field(default=2.5, gt=0)
    output_dir: str = Field(default="outputs")
    max_steps: int = Field(default=32, ge=1, le=120)


app = FastAPI(title="Instruction Video Chat UI", version="0.1.0")


def _new_queue() -> list[QueueStep]:
    return [QueueStep(step_id=step_id, label=label, description=description) for step_id, label, description in QUEUE_TEMPLATE]


def _job_payload(job: JobState) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "prompt": job.prompt,
        "status": job.status,
        "current_step": job.current_step,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "message": job.message,
        "error": job.error,
        "queue": [asdict(step) for step in job.queue],
        "result": job.result,
    }


def _update_timestamp(job: JobState) -> None:
    job.updated_at = datetime.now().isoformat(timespec="seconds")


def _set_step_active(job: JobState, step_id: str, detail: str) -> None:
    for step in job.queue:
        if step.status == "active":
            step.status = "done"
        if step.step_id == step_id:
            step.status = "active"
            step.detail = detail
    job.current_step = step_id
    _update_timestamp(job)


def _set_step_done(job: JobState, step_id: str, detail: str) -> None:
    for step in job.queue:
        if step.step_id == step_id:
            step.status = "done"
            step.detail = detail
    _update_timestamp(job)


def _set_step_error(job: JobState, step_id: str, detail: str) -> None:
    for step in job.queue:
        if step.step_id == step_id:
            step.status = "error"
            step.detail = detail
    _update_timestamp(job)


def _default_cloud_profile_id() -> str | None:
    return (
        os.getenv("BROWSER_USE_DEFAULT_PROFILE_ID")
        or os.getenv("BROWSER_USE_PROFILE_ID")
        or os.getenv("CLOUD_PROFILE_ID")
        or DEFAULT_CLOUD_PROFILE_ID
    )


def _build_request(payload: CreateJobRequest, job_id: str) -> GenerationRequest:
    try:
        provider = LLMProvider(payload.provider)
    except ValueError as exc:
        raise ValueError(f"Unsupported provider: {payload.provider}") from exc
    try:
        cloud_mode = BrowserCloudMode(payload.cloud_mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported cloud mode: {payload.cloud_mode}") from exc

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = sanitize_job_name(f"chat-{stamp}-{job_id[:8]}")
    profile_id = payload.cloud_profile_id or _default_cloud_profile_id()

    request = GenerationRequest(
        instructions=payload.prompt,
        output_dir=Path(payload.output_dir),
        llm_provider=provider,
        llm_model=payload.model,
        cloud_mode=cloud_mode,
        cloud_profile_id=profile_id,
        cloud_proxy_country_code=payload.cloud_proxy_country_code,
        browser_video_speed=payload.browser_video_speed,
        narration_speed=payload.narration_speed,
        narration_min_speed=payload.narration_min_speed,
        narration_max_speed=payload.narration_max_speed,
        max_steps=payload.max_steps,
        job_name=job_name,
    )
    request.validate()
    return request


async def _run_job(job_id: str, payload: CreateJobRequest) -> None:
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return

    try:
        request = _build_request(payload, job_id)
        pipeline = InstructionVideoPipeline()

        output_dir = request.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        job_dir = output_dir / (request.job_name or f"chat-{job_id[:8]}")
        job_dir.mkdir(parents=True, exist_ok=True)

        async with JOBS_LOCK:
            job.status = "running"
            job.message = "Running browser automation"
            _set_step_done(job, "queued", "Prompt accepted")
            _set_step_active(job, "browser", "Launching browser agent")

        browser_artifacts = await pipeline.browser_runner.run(request, job_dir)
        raw_source_path = browser_artifacts.video_path
        if abs(request.browser_video_speed - 1.0) > 1e-6:
            sped_path = raw_source_path.with_name(f"{raw_source_path.stem}-x{request.browser_video_speed:.2f}.mp4")
            browser_artifacts.video_path = await asyncio.to_thread(
                pipeline.video_editor.speed_adjust_video,
                raw_source_path,
                request.browser_video_speed,
                sped_path,
            )
        raw_video_duration = await asyncio.to_thread(pipeline.video_editor.probe_duration, browser_artifacts.video_path)

        async with JOBS_LOCK:
            job.message = "Generating narration"
            _set_step_done(job, "browser", f"Captured browser run ({raw_video_duration:.1f}s)")
            _set_step_active(job, "narration", "Writing and synthesizing narration")

        narration_artifacts = await asyncio.to_thread(
            pipeline.narration_service.create_artifacts,
            request,
            browser_artifacts.events,
            browser_artifacts.final_result,
            job_dir,
            raw_video_duration,
        )

        async with JOBS_LOCK:
            job.message = "Rendering final video"
            _set_step_done(
                job,
                "narration",
                f"Narration generated ({(narration_artifacts.duration_seconds or 0):.1f}s)",
            )
            _set_step_active(job, "render", "Applying zoom, timing, and audio mix")

        video_artifacts = await asyncio.to_thread(
            pipeline.video_editor.render,
            request,
            browser_artifacts,
            narration_artifacts,
            job_dir,
        )

        manifest_path = job_dir / "manifest.json"
        manifest = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "job_name": request.job_name,
            "instructions": request.instructions,
            "llm_provider": request.llm_provider.value,
            "llm_model": request.llm_model,
            "cloud_mode": request.cloud_mode.value,
            "run_mode": browser_artifacts.run_mode,
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
        await asyncio.to_thread(
            manifest_path.write_text,
            json.dumps(manifest, indent=2),
            "utf-8",
        )

        async with JOBS_LOCK:
            job.status = "completed"
            job.message = "Video generation complete"
            _set_step_done(job, "render", "Final video composed")
            _set_step_done(job, "done", "Instructional video is ready")
            job.current_step = "done"
            job.result = {
                "job_dir": str(job_dir),
                "final_video_path": str(video_artifacts.final_video_path),
                "manifest_path": str(manifest_path),
                "narration_script_path": str(narration_artifacts.script_path),
                "narration_audio_path": str(narration_artifacts.audio_path),
                "browser_video_path": str(browser_artifacts.video_path),
            }
            _update_timestamp(job)
    except Exception as exc:
        async with JOBS_LOCK:
            if job.current_step:
                _set_step_error(job, job.current_step, str(exc))
            job.status = "failed"
            job.error = str(exc)
            job.message = "Generation failed"
            _update_timestamp(job)


@app.post("/api/jobs")
async def create_job(payload: CreateJobRequest) -> dict[str, Any]:
    job = JobState(job_id=str(uuid.uuid4()), prompt=payload.prompt, queue=_new_queue())
    async with JOBS_LOCK:
        JOBS[job.job_id] = job
    asyncio.create_task(_run_job(job.job_id, payload))
    return _job_payload(job)


@app.get("/api/jobs")
async def list_jobs() -> dict[str, Any]:
    async with JOBS_LOCK:
        ordered = sorted(JOBS.values(), key=lambda item: item.created_at, reverse=True)
        payload = [_job_payload(job) for job in ordered[:20]]
    return {"jobs": payload}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_payload(job)


@app.get("/api/jobs/{job_id}/video")
async def get_job_video(job_id: str):
    async with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not job.result or not job.result.get("final_video_path"):
        raise HTTPException(status_code=404, detail="Video not available")
    path = Path(job.result["final_video_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_html_template())


def _html_template() -> str:
    return """<!doctype html>
<html lang=\"en\"> 
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Instruction Video Generator Chat</title>
  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\"> 
  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
  <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap\" rel=\"stylesheet\">
  <style>
    :root {
      --bg: #f6f7f2;
      --card: #ffffff;
      --ink: #122025;
      --muted: #5d6f78;
      --brand: #0d9f8f;
      --brand-2: #ff7f3f;
      --ok: #1e8f4a;
      --warn: #db5d3b;
      --line: #d7dedf;
      --shadow: 0 14px 34px rgba(18, 32, 37, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'Space Grotesk', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 20%, rgba(13, 159, 143, 0.12), transparent 36%),
        radial-gradient(circle at 90% 5%, rgba(255, 127, 63, 0.10), transparent 30%),
        var(--bg);
    }

    .shell {
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 20px;
    }

    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: linear-gradient(90deg, rgba(13,159,143,0.08), rgba(255,127,63,0.06));
    }

    h1 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.02em;
    }

    .meta {
      color: var(--muted);
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
    }

    .chat-log {
      height: 58vh;
      overflow: auto;
      padding: 22px 20px;
      display: grid;
      align-content: start;
      gap: 14px;
      background: linear-gradient(180deg, rgba(13,159,143,0.02), transparent 25%);
    }

    .bubble {
      width: fit-content;
      max-width: min(72%, 540px);
      padding: 13px 15px;
      border-radius: 16px;
      font-size: 0.95rem;
      line-height: 1.45;
      box-shadow: 0 8px 20px rgba(18, 32, 37, 0.08);
      animation: bubbleIn 0.24s ease-out;
    }

    .bubble.user {
      justify-self: end;
      max-width: min(66%, 500px);
      background: #0f2f3a;
      color: #eff7fa;
      border-bottom-right-radius: 6px;
    }

    .bubble.assistant {
      justify-self: start;
      max-width: min(70%, 560px);
      background: #eef4f4;
      border: 1px solid var(--line);
      border-bottom-left-radius: 6px;
    }

    .controls {
      border-top: 1px solid var(--line);
      padding: 14px;
      display: grid;
      gap: 10px;
      background: #fbfcfa;
    }

    .input-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }

    input[type=\"text\"], select {
      width: 100%;
      border: 1px solid #bfd0d3;
      border-radius: 10px;
      font: inherit;
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
    }

    .btn {
      border: 0;
      border-radius: 10px;
      padding: 0 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: linear-gradient(120deg, var(--brand), #0a7e80);
      transition: transform 0.18s ease, filter 0.18s ease;
    }

    .btn:hover { transform: translateY(-1px); filter: brightness(1.04); }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }

    .settings-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .settings-grid label {
      display: grid;
      gap: 6px;
      font-size: 0.8rem;
      color: var(--muted);
    }

    .queue {
      padding: 14px;
      display: grid;
      gap: 10px;
    }

    .queue-item {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fff;
      position: relative;
      overflow: hidden;
      transition: border-color 0.2s ease, background 0.2s ease;
    }

    .queue-item .label {
      font-weight: 700;
      font-size: 0.88rem;
    }

    .queue-item .desc,
    .queue-item .detail {
      color: var(--muted);
      font-size: 0.78rem;
      margin-top: 4px;
      line-height: 1.35;
    }

    .queue-item.pending { opacity: 0.66; }

    .queue-item.active {
      border-color: var(--brand);
      background: rgba(13, 159, 143, 0.06);
      animation: pulse 1.25s ease-in-out infinite;
    }

    .queue-item.active::after {
      content: '';
      position: absolute;
      left: -32%;
      top: 0;
      width: 28%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(13, 159, 143, 0.22), transparent);
      animation: shimmer 1.5s linear infinite;
    }

    .queue-item.done {
      border-color: rgba(30, 143, 74, 0.5);
      background: rgba(30, 143, 74, 0.08);
    }

    .queue-item.error {
      border-color: rgba(219, 93, 59, 0.5);
      background: rgba(219, 93, 59, 0.10);
    }

    .video-wrap {
      border-top: 1px solid var(--line);
      padding: 14px;
      display: grid;
      gap: 8px;
      background: #f8faf9;
    }

    video {
      width: 100%;
      border-radius: 12px;
      border: 1px solid #c9d6d8;
      background: #0b0f12;
      max-height: 320px;
    }

    .path {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      color: var(--muted);
      overflow-wrap: anywhere;
    }

    @keyframes pulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.004); }
    }

    @keyframes shimmer {
      to { transform: translateX(460%); }
    }

    @keyframes bubbleIn {
      from { transform: translateY(4px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
    }

    @media (max-width: 980px) {
      .shell {
        grid-template-columns: 1fr;
        padding: 14px;
      }
      .chat-log {
        height: 44vh;
        padding: 14px;
      }
      .bubble,
      .bubble.user,
      .bubble.assistant {
        max-width: 88%;
      }
      .settings-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <section class=\"panel\">
      <div class=\"panel-header\">
        <h1>Prompt Chat</h1>
        <span class=\"meta\" id=\"sessionMeta\">idle</span>
      </div>
      <div class=\"chat-log\" id=\"chatLog\"></div>
      <form class=\"controls\" id=\"promptForm\">
        <div class=\"settings-grid\">
          <label>Provider
            <select id=\"provider\">
              <option value=\"anthropic\" selected>anthropic</option>
              <option value=\"openai\">openai</option>
              <option value=\"gemini\">gemini</option>
            </select>
          </label>
          <label>Model
            <input id=\"model\" type=\"text\" value=\"claude-sonnet-4-6\" />
          </label>
          <label>Mode
            <select id=\"cloudMode\">
              <option value=\"cloud\" selected>cloud</option>
              <option value=\"local\">local</option>
              <option value=\"auto\">auto</option>
            </select>
          </label>
          <label>Cloud Profile ID
            <input id=\"cloudProfile\" type=\"text\" value=\"__DEFAULT_PROFILE_ID__\" />
          </label>
        </div>
        <div class=\"input-row\">
          <input id=\"promptInput\" type=\"text\" placeholder=\"Ask for the next instructional video...\" required />
          <button class=\"btn\" id=\"sendBtn\" type=\"submit\">Generate</button>
        </div>
      </form>
    </section>

    <section class=\"panel\">
      <div class=\"panel-header\">
        <h1>Generation Queue</h1>
        <span class=\"meta\" id=\"jobStatus\">waiting</span>
      </div>
      <div class=\"queue\" id=\"queueList\"></div>
      <div class=\"video-wrap\">
        <video id=\"resultVideo\" controls preload=\"metadata\"></video>
        <div class=\"path\" id=\"resultPath\">No video generated yet.</div>
      </div>
    </section>
  </div>

  <script>
    const chatLog = document.getElementById('chatLog');
    const queueList = document.getElementById('queueList');
    const promptForm = document.getElementById('promptForm');
    const promptInput = document.getElementById('promptInput');
    const sendBtn = document.getElementById('sendBtn');
    const sessionMeta = document.getElementById('sessionMeta');
    const jobStatus = document.getElementById('jobStatus');
    const resultVideo = document.getElementById('resultVideo');
    const resultPath = document.getElementById('resultPath');

    let pollTimer = null;
    let activeJobId = null;

    function addMessage(role, text) {
      const bubble = document.createElement('div');
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;
      chatLog.appendChild(bubble);
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    function renderQueue(queue) {
      queueList.innerHTML = '';
      for (const step of queue) {
        const item = document.createElement('div');
        item.className = `queue-item ${step.status}`;
        item.innerHTML = `
          <div class=\"label\">${step.label}</div>
          <div class=\"desc\">${step.description}</div>
          <div class=\"detail\">${step.detail || ''}</div>
        `;
        queueList.appendChild(item);
      }
    }

    async function pollJob(jobId) {
      if (pollTimer) {
        clearInterval(pollTimer);
      }
      activeJobId = jobId;
      pollTimer = setInterval(async () => {
        try {
          const resp = await fetch(`/api/jobs/${jobId}`);
          if (!resp.ok) return;
          const data = await resp.json();
          sessionMeta.textContent = `job ${data.job_id.slice(0, 8)}`;
          jobStatus.textContent = `${data.status}${data.current_step ? ` · ${data.current_step}` : ''}`;
          renderQueue(data.queue || []);

          if (data.status === 'completed') {
            clearInterval(pollTimer);
            pollTimer = null;
            addMessage('assistant', 'Video is ready. You can preview it on the right panel.');
            resultVideo.src = `/api/jobs/${jobId}/video?ts=${Date.now()}`;
            resultPath.textContent = data?.result?.final_video_path || 'Video path unavailable';
          }

          if (data.status === 'failed') {
            clearInterval(pollTimer);
            pollTimer = null;
            addMessage('assistant', `Generation failed: ${data.error || 'Unknown error'}`);
          }
        } catch (err) {
          console.error(err);
        }
      }, 1200);
    }

    promptForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const prompt = promptInput.value.trim();
      if (!prompt) return;

      const payload = {
        prompt,
        provider: document.getElementById('provider').value,
        model: document.getElementById('model').value.trim() || 'claude-sonnet-4-6',
        cloud_mode: document.getElementById('cloudMode').value,
        cloud_profile_id: document.getElementById('cloudProfile').value.trim() || null,
      };

      addMessage('user', prompt);
      addMessage('assistant', 'Working on it. I started a new generation run and queued each pipeline stage below.');
      promptInput.value = '';
      sendBtn.disabled = true;
      jobStatus.textContent = 'starting';

      try {
        const resp = await fetch('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data?.detail || 'Unable to start generation');
        }
        renderQueue(data.queue || []);
        await pollJob(data.job_id);
      } catch (err) {
        addMessage('assistant', `Failed to start generation: ${err.message}`);
      } finally {
        sendBtn.disabled = false;
      }
    });

    addMessage('assistant', 'Send a prompt describing the instructional video you want to generate next.');
  </script>
</body>
</html>""".replace("__DEFAULT_PROFILE_ID__", DEFAULT_CLOUD_PROFILE_ID)
