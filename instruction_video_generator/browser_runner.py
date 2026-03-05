from __future__ import annotations

import base64
import json
import os
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser import BrowserProfile, BrowserSession

from .llm_factory import LLMFactory
from .models import ActionEvent, BrowserCloudMode, BrowserRunArtifacts, GenerationRequest


CHALLENGE_KEYWORDS = (
    "cloudflare",
    "verify you are human",
    "checking your browser",
    "captcha",
    "challenge",
    "attention required",
)

CHALLENGE_PRONE_DOMAINS = (
    "canva.com",
    "figma.com",
    "linkedin.com",
    "x.com",
    "twitter.com",
)

HOW_TO_HINTS = (
    "how do i",
    "how to",
    "show me how",
    "walk me through",
    "tutorial",
)

DOCS_PAGE_HINTS = (
    "help page",
    "support page",
    "support article",
    "documentation",
    "docs page",
    "knowledge base",
)


def extract_action_events(
    history: AgentHistoryList,
    viewport_width: int,
    viewport_height: int,
) -> list[ActionEvent]:
    events: list[ActionEvent] = []
    for index, item in enumerate(history.history, start=1):
        metadata = item.metadata
        state = item.state
        extracted_content: list[str] = []
        errors: list[str] = []
        success_values: list[bool] = []
        for result in item.result or []:
            if result.extracted_content:
                extracted_content.append(result.extracted_content)
            if result.error:
                errors.append(result.error)
            if result.success is not None:
                success_values.append(result.success)
        actions: list[dict[str, Any]] = []
        action_names: list[str] = []
        center_x: float | None = None
        center_y: float | None = None
        interacted = getattr(state, "interacted_element", None) or []
        for element in interacted:
            bounds = getattr(element, "bounds", None)
            if not bounds:
                continue
            center_x = max(0.0, min(1.0, (bounds.x + bounds.width / 2) / float(viewport_width)))
            center_y = max(0.0, min(1.0, (bounds.y + bounds.height / 2) / float(viewport_height)))
            break
        if item.model_output and item.model_output.action:
            for action in item.model_output.action:
                payload = action.model_dump(exclude_none=True)
                if not payload:
                    continue
                name = next(iter(payload.keys()))
                action_names.append(name)
                raw = payload.get(name)
                actions.append({name: raw if isinstance(raw, dict) else {"value": raw}})
        event = ActionEvent(
            step_number=metadata.step_number if metadata else index,
            started_at=metadata.step_start_time if metadata else None,
            ended_at=metadata.step_end_time if metadata else None,
            url=getattr(state, "url", None),
            title=getattr(state, "title", None),
            action_names=action_names,
            actions=actions,
            success=all(success_values) if success_values else None,
            extracted_content=extracted_content,
            errors=errors,
            center_x=center_x,
            center_y=center_y,
        )
        events.append(event)
    return events


class BrowserAutomationRunner:
    def __init__(
        self,
        llm_factory: LLMFactory,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        browser_use_api_key: str | None = None,
    ):
        self.llm_factory = llm_factory
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.browser_use_api_key = browser_use_api_key
        if browser_use_api_key:
            os.environ["BROWSER_USE_API_KEY"] = browser_use_api_key

    async def run(self, request: GenerationRequest, job_dir: Path) -> BrowserRunArtifacts:
        raw_video_dir = job_dir / "raw_browser_video"
        artifacts_dir = job_dir / "artifacts"
        raw_video_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        run_mode = self._resolve_initial_mode(request)
        history, events, video_path = await self._run_once(request, raw_video_dir, run_mode)
        if (
            run_mode == "local"
            and request.cloud_fallback_on_challenge
            and self.browser_use_api_key
            and self._has_challenge_block(events, request.challenge_stuck_threshold)
        ):
            history, events, video_path = await self._run_once(request, raw_video_dir, "cloud")
            run_mode = "cloud"

        history_path = artifacts_dir / "agent_history.json"
        history.save_to_file(history_path)
        actions_path = artifacts_dir / "action_events.json"
        actions_path.write_text(
            json.dumps([event.to_dict() for event in events], indent=2),
            encoding="utf-8",
        )
        video_path = self._ensure_min_duration(video_path, request.min_raw_video_seconds)
        return BrowserRunArtifacts(
            video_path=video_path,
            history_path=history_path,
            actions_path=actions_path,
            events=events,
            final_result=history.final_result(),
            run_mode=run_mode,
        )

    async def _run_once(
        self,
        request: GenerationRequest,
        raw_video_dir: Path,
        run_mode: str,
    ) -> tuple[AgentHistoryList, list[ActionEvent], Path]:
        llm = self.llm_factory.create(request.llm_provider, request.llm_model)
        is_challenge_prone = self._targets_challenge_prone_site(request)
        profile_kwargs: dict[str, Any] = {
            "headless": request.headless if run_mode == "local" else True,
            "window_size": {"width": request.viewport_width, "height": request.viewport_height},
            "viewport": {"width": request.viewport_width, "height": request.viewport_height},
            "record_video_dir": raw_video_dir,
            "record_video_size": {"width": request.viewport_width, "height": request.viewport_height},
            "highlight_elements": True,
            "enable_default_extensions": not is_challenge_prone,
            "wait_between_actions": 0.6,
            "wait_for_network_idle_page_load_time": 1.2,
        }
        if run_mode == "local":
            if request.local_user_data_dir:
                profile_kwargs["user_data_dir"] = request.local_user_data_dir
            if request.local_profile_directory:
                profile_kwargs["profile_directory"] = request.local_profile_directory
            if request.use_system_chrome or request.local_user_data_dir or request.local_profile_directory:
                profile_kwargs["enable_default_extensions"] = False
        browser_profile = BrowserProfile(**profile_kwargs)
        session_kwargs: dict[str, Any] = {"browser_profile": browser_profile}
        cloud_session_id: str | None = None
        if run_mode == "cloud":
            cloud_session = await self._create_cloud_session(request)
            cloud_session_id = str(cloud_session.id)
            session_kwargs["cdp_url"] = self._extract_cdp_url_from_live_url(cloud_session.live_url)
            browser_session = BrowserSession(**session_kwargs)
        elif request.use_system_chrome and not request.local_user_data_dir:
            system_profile_kwargs: dict[str, Any] = {
                "headless": request.headless,
                "window_size": {"width": request.viewport_width, "height": request.viewport_height},
                "viewport": {"width": request.viewport_width, "height": request.viewport_height},
                "record_video_dir": raw_video_dir,
                "record_video_size": {"width": request.viewport_width, "height": request.viewport_height},
                "highlight_elements": True,
                "enable_default_extensions": False,
                "wait_between_actions": 0.6,
                "wait_for_network_idle_page_load_time": 1.2,
            }
            browser_session = BrowserSession.from_system_chrome(
                profile_directory=request.local_profile_directory,
                **system_profile_kwargs,
            )
        else:
            browser_session = BrowserSession(**session_kwargs)
        task = self._build_task(request)
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            use_vision=True,
            include_recent_events=True,
            max_actions_per_step=request.max_actions_per_step,
            step_timeout=request.step_timeout,
            enable_planning=True,
        )
        existing_files = {
            path
            for path in raw_video_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".webm", ".mp4", ".mkv"}
        }
        history: AgentHistoryList | None = None
        try:
            history = await agent.run(max_steps=request.max_steps)
        finally:
            with suppress(Exception):
                await browser_session.stop()
            if cloud_session_id:
                with suppress(Exception):
                    await self._stop_cloud_session(cloud_session_id)
        if history is None:
            raise RuntimeError("browser-use run did not produce history")
        events = extract_action_events(history, request.viewport_width, request.viewport_height)
        try:
            video_path = self._resolve_recorded_video(raw_video_dir, existing_files=existing_files)
        except RuntimeError:
            video_path = self._build_video_from_history_screenshots(history, raw_video_dir, request)
        return history, events, video_path

    def _build_task(self, request: GenerationRequest) -> str:
        lines = [
            "You are creating a browser walkthrough that will become an instructional video.",
            "Execute the user instructions exactly in order.",
            "Use visible UI interactions whenever possible.",
            "After each major action, briefly pause so viewers can see the result.",
            "Do not explore unrelated pages or flows.",
            "When all instructions are complete, finish the task with a concise completion summary.",
        ]
        if self._is_how_to_instruction(request.instructions):
            lines.append("This is a how-to request: demonstrate by doing the steps in the real product UI.")
            if not self._explicitly_requests_docs_page(request.instructions):
                lines.append(
                    "Do not fulfill the request by only visiting help/support/documentation pages."
                )
            lines.append("Show exact clicks/menus/settings directly in the product whenever possible.")
        if self._is_google_docs_layout_request(request.instructions):
            lines.extend(
                [
                    "For Google Docs layout requests, demonstrate this directly in a document editor, not only in an article:",
                    "Open a document, then use File > Page setup.",
                    "Show Orientation and Margins settings with clear on-screen interactions.",
                ]
            )
        if self._is_google_docs_table_request(request.instructions):
            lines.extend(
                [
                    "For Google Docs table requests, demonstrate directly in the editor:",
                    "Open a document and use Insert > Table.",
                    "Select a table size in the grid and click into a table cell.",
                    "Do not redirect to support/help pages for this request.",
                ]
            )
        lines.append(f"User instructions:\n{request.instructions.strip()}")
        if request.start_url:
            lines.append(f"Start URL: {request.start_url.strip()}")
        return "\n\n".join(lines)

    def _is_how_to_instruction(self, instructions: str) -> bool:
        text = instructions.lower()
        if any(hint in text for hint in HOW_TO_HINTS):
            return True
        if "?" in instructions and ("how" in text):
            return True
        return False

    def _explicitly_requests_docs_page(self, instructions: str) -> bool:
        text = instructions.lower()
        return any(hint in text for hint in DOCS_PAGE_HINTS)

    def _is_google_docs_layout_request(self, instructions: str) -> bool:
        text = instructions.lower()
        has_docs = "google doc" in text or "docs.google.com" in text
        has_layout = "layout" in text or "page setup" in text or "margins" in text
        return has_docs and has_layout

    def _is_google_docs_table_request(self, instructions: str) -> bool:
        text = instructions.lower()
        has_docs = "google doc" in text or "docs.google.com" in text
        has_table = "table" in text and ("create" in text or "insert" in text or "add" in text)
        return has_docs and has_table

    def _resolve_initial_mode(self, request: GenerationRequest) -> str:
        if request.cloud_mode == BrowserCloudMode.CLOUD:
            if not self.browser_use_api_key:
                raise ValueError("BROWSER_USE_API_KEY is required when --cloud-mode cloud")
            return "cloud"
        if request.cloud_mode == BrowserCloudMode.LOCAL:
            return "local"
        if self.browser_use_api_key and self._targets_challenge_prone_site(request):
            return "cloud"
        return "local"

    def _targets_challenge_prone_site(self, request: GenerationRequest) -> bool:
        scope = f"{request.start_url or ''} {request.instructions}".lower()
        return any(domain in scope for domain in CHALLENGE_PRONE_DOMAINS)

    def _has_challenge_block(self, events: list[ActionEvent], threshold: int) -> bool:
        streak = 0
        for event in events:
            haystack = " ".join(
                [
                    event.title or "",
                    event.url or "",
                    " ".join(event.errors),
                    " ".join(event.extracted_content),
                ]
            ).lower()
            if any(keyword in haystack for keyword in CHALLENGE_KEYWORDS):
                streak += 1
                if streak >= threshold:
                    return True
            else:
                streak = 0
        return False

    async def _create_cloud_session(self, request: GenerationRequest):
        if not self.browser_use_api_key:
            raise ValueError("BROWSER_USE_API_KEY is required for cloud sessions")
        try:
            from browser_use_sdk import AsyncBrowserUse
        except Exception as exc:
            raise RuntimeError(
                "browser-use-sdk is required for cloud remote sessions. Install with: pip install browser-use-sdk"
            ) from exc
        client = AsyncBrowserUse(api_key=self.browser_use_api_key)
        sessions = client.sessions
        create_fn = getattr(sessions, "create", None) or getattr(sessions, "create_session", None)
        if create_fn is None:
            raise RuntimeError("Unsupported browser-use-sdk: no sessions.create/create_session method")
        return await create_fn(
            profile_id=request.cloud_profile_id,
            proxy_country_code=request.cloud_proxy_country_code,
        )

    async def _stop_cloud_session(self, cloud_session_id: str) -> None:
        if not self.browser_use_api_key:
            return
        try:
            from browser_use_sdk import AsyncBrowserUse
        except Exception:
            return
        client = AsyncBrowserUse(api_key=self.browser_use_api_key)
        sessions = client.sessions
        stop_fn = getattr(sessions, "stop", None)
        if stop_fn is not None:
            await stop_fn(cloud_session_id)
            return
        # Newer SDKs use update_session(action='stop')
        update_fn = getattr(sessions, "update", None) or getattr(sessions, "update_session", None)
        if update_fn is not None:
            await update_fn(cloud_session_id, action="stop")

    def _extract_cdp_url_from_live_url(self, live_url: str | None) -> str:
        if not live_url:
            raise RuntimeError("Browser Use Cloud session did not provide a live_url")
        parsed = urlparse(live_url)
        wss_values = parse_qs(parsed.query).get("wss")
        if not wss_values:
            raise RuntimeError("Browser Use Cloud live_url did not include a CDP endpoint")
        return unquote(wss_values[0])

    def _resolve_recorded_video(self, raw_video_dir: Path, existing_files: set[Path] | None = None) -> Path:
        candidates = [
            path
            for path in raw_video_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".webm", ".mp4", ".mkv"}
        ]
        if existing_files:
            new_candidates = [path for path in candidates if path not in existing_files]
            if new_candidates:
                candidates = new_candidates
        if not candidates:
            raise RuntimeError(
                "No browser recording found. Ensure Playwright video recording is enabled."
            )
        candidates.sort(key=lambda path: (path.stat().st_size, path.stat().st_mtime), reverse=True)
        return candidates[0]

    def _build_video_from_history_screenshots(
        self,
        history: AgentHistoryList,
        raw_video_dir: Path,
        request: GenerationRequest,
    ) -> Path:
        frames_dir = raw_video_dir / "frames_fallback"
        frames_dir.mkdir(parents=True, exist_ok=True)
        concat_path = frames_dir / "frames.txt"
        frame_entries: list[tuple[Path, float]] = []

        for index, item in enumerate(history.history, start=1):
            state = getattr(item, "state", None)
            screenshot = getattr(state, "screenshot", None)
            if not screenshot:
                continue
            encoded = screenshot
            if encoded.startswith("data:image") and "," in encoded:
                encoded = encoded.split(",", 1)[1]
            frame_path = frames_dir / f"frame-{index:04d}.png"
            try:
                frame_path.write_bytes(base64.b64decode(encoded))
            except Exception:
                continue
            metadata = getattr(item, "metadata", None)
            start = getattr(metadata, "step_start_time", None)
            end = getattr(metadata, "step_end_time", None)
            duration = 1.2
            if isinstance(start, (float, int)) and isinstance(end, (float, int)) and end > start:
                duration = float(end - start)
            duration = max(0.45, min(4.0, duration))
            frame_entries.append((frame_path, duration))

        if not frame_entries:
            raise RuntimeError("No browser recording or screenshot frames were available for this run")

        lines: list[str] = []
        for frame_path, duration in frame_entries:
            lines.append(f"file '{frame_path.as_posix()}'")
            lines.append(f"duration {duration:.3f}")
        # concat demuxer needs the final file repeated for the last duration to take effect.
        lines.append(f"file '{frame_entries[-1][0].as_posix()}'")
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        output_path = raw_video_dir / "recording-from-steps.mp4"
        vf = (
            f"scale={request.viewport_width}:{request.viewport_height}:force_original_aspect_ratio=decrease,"
            f"pad={request.viewport_width}:{request.viewport_height}:(ow-iw)/2:(oh-ih)/2,"
            "fps=30,format=yuv420p"
        )
        command = [
            self.ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        return output_path

    def _probe_duration(self, media_path: Path) -> float:
        command = [
            self.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())

    def _ensure_min_duration(self, video_path: Path, minimum_seconds: float) -> Path:
        if minimum_seconds <= 0:
            return video_path
        duration = self._probe_duration(video_path)
        if duration >= minimum_seconds:
            return video_path
        extension_seconds = max(0.01, minimum_seconds - duration)
        output_path = video_path.with_name(f"{video_path.stem}-extended.mp4")
        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={extension_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        return output_path
