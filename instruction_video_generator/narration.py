from __future__ import annotations

import re
import subprocess
from pathlib import Path

from openai import OpenAI

from .models import ActionEvent, GenerationRequest, NarrationArtifacts
from .settings import Settings


ACTION_PHRASES = {
    "navigate": "open the requested page",
    "click": "click the target control",
    "input": "enter text in the active field",
    "scroll": "scroll the page",
    "switch": "switch to the active browser tab",
    "go_to_url": "open the target page",
    "search_google": "search for the requested page",
    "open_tab": "open a new browser tab",
    "switch_tab": "switch to another browser tab",
    "click_element_by_index": "click the highlighted element",
    "click_element": "click the highlighted element",
    "input_text": "type the requested text",
    "scroll_down": "scroll down",
    "scroll_up": "scroll up",
    "send_keys": "use keyboard input",
    "done": "finish the workflow",
}


class NarrationService:
    def __init__(self, settings: Settings):
        self.settings = settings

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
        script_text = self.build_script(
            request.instructions,
            events,
            final_result,
            max_chars=request.narration_max_chars,
            target_duration_seconds=target_duration_seconds,
            speaking_speed=request.narration_speed,
        )
        audio_path = narration_dir / "narration.wav"
        script_text, speed_used, duration_seconds = self.generate_audio_with_dynamic_speed(
            script_text=script_text,
            output_path=audio_path,
            voice=request.narration_voice,
            model=request.narration_model,
            requested_speed=request.narration_speed,
            min_speed=request.narration_min_speed,
            max_speed=request.narration_max_speed,
            target_duration_seconds=target_duration_seconds,
        )
        script_path = narration_dir / "narration_script.txt"
        script_path.write_text(script_text, encoding="utf-8")
        return NarrationArtifacts(
            script_path=script_path,
            audio_path=audio_path,
            script_text=script_text,
            speed_used=speed_used,
            duration_seconds=duration_seconds,
        )

    def generate_audio_with_dynamic_speed(
        self,
        script_text: str,
        output_path: Path,
        voice: str,
        model: str,
        requested_speed: float,
        min_speed: float,
        max_speed: float,
        target_duration_seconds: float | None,
    ) -> tuple[str, float, float]:
        working_script = script_text
        speed_used = max(min_speed, min(max_speed, requested_speed))
        self.generate_audio(
            script_text=working_script,
            output_path=output_path,
            voice=voice,
            model=model,
            speed=speed_used,
        )
        duration_seconds = self._probe_duration(output_path)
        if not target_duration_seconds or target_duration_seconds <= 0:
            return working_script, speed_used, duration_seconds

        adjusted_speed = self._compute_dynamic_speed(
            current_speed=speed_used,
            current_audio_duration=duration_seconds,
            target_duration=target_duration_seconds,
            min_speed=min_speed,
            max_speed=max_speed,
        )
        if abs(adjusted_speed - speed_used) >= 0.03:
            speed_used = adjusted_speed
            self.generate_audio(
                script_text=working_script,
                output_path=output_path,
                voice=voice,
                model=model,
                speed=speed_used,
            )
            duration_seconds = self._probe_duration(output_path)

        # If narration is still too long at max speed, tighten script length to keep audio aligned.
        if duration_seconds > target_duration_seconds * 1.02 and speed_used >= (max_speed - 0.01):
            compressed = self._compress_script_for_target(
                working_script,
                current_duration=duration_seconds,
                target_duration=target_duration_seconds,
            )
            if compressed != working_script:
                working_script = compressed
                self.generate_audio(
                    script_text=working_script,
                    output_path=output_path,
                    voice=voice,
                    model=model,
                    speed=speed_used,
                )
                duration_seconds = self._probe_duration(output_path)

        if target_duration_seconds > 0:
            delta = abs(duration_seconds - target_duration_seconds)
            if delta > 0.03:
                tempo_factor = duration_seconds / target_duration_seconds
                self._retime_audio(output_path, tempo_factor)
                duration_seconds = self._probe_duration(output_path)

        return working_script, speed_used, duration_seconds

    def build_script(
        self,
        instructions: str,
        events: list[ActionEvent],
        final_result: str | None,
        max_chars: int = 420,
        target_duration_seconds: float | None = None,
        speaking_speed: float = 1.0,
    ) -> str:
        lines: list[str] = []
        guidance_limit = 4
        if target_duration_seconds and target_duration_seconds > 0:
            if target_duration_seconds < 14:
                guidance_limit = 3
            elif target_duration_seconds > 45:
                guidance_limit = 5
        guidance_points = self._to_guidance_points(events, max_points=guidance_limit)
        if guidance_points:
            lines.append("Let's walk through it together.")
            lines.extend(self._compose_guidance_sentences(guidance_points))
        else:
            lines.append(f"Let's walk through this flow: {self._clean_text(instructions.strip())}.")
        if final_result and final_result.strip():
            outcome = self._summarize_outcome(final_result)
            lines.append(f"At the end, you should see {outcome}.")
        else:
            lines.append("At the end, you should see the workflow completed successfully.")
        script = self._truncate_script(" ".join(lines), max_chars=max_chars)
        if target_duration_seconds and target_duration_seconds > 0:
            word_budget = self._estimate_word_budget(target_duration_seconds, speaking_speed)
            script = self._truncate_words(script, max_words=word_budget)
        return script

    def generate_audio(
        self,
        script_text: str,
        output_path: Path,
        voice: str,
        model: str,
        speed: float,
    ) -> None:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for narration generation")
        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=script_text,
            response_format="wav",
            speed=speed,
        )
        response.write_to_file(output_path)

    def _to_instructional_steps(self, events: list[ActionEvent], max_steps: int = 8) -> list[str]:
        if max_steps <= 0:
            return []
        descriptions: list[str] = []
        for event in events:
            description = self._describe_event(event)
            if not description:
                continue
            descriptions.append(description)
        if not descriptions:
            return []
        if len(descriptions) > max_steps:
            if max_steps == 1:
                descriptions = [descriptions[0]]
            else:
                count = len(descriptions)
                indices = {
                    round(i * (count - 1) / (max_steps - 1))
                    for i in range(max_steps)
                }
                descriptions = [descriptions[idx] for idx in sorted(indices)]
        steps: list[str] = []
        for description in descriptions:
            steps.append(f"Step {len(steps) + 1}: {description}.")
        return steps

    def _truncate_script(self, script: str, max_chars: int) -> str:
        trimmed = " ".join(script.split())
        if len(trimmed) <= max_chars:
            return trimmed
        clipped = trimmed[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
        if not clipped.endswith("."):
            clipped += "."
        return clipped

    def _clean_text(self, value: str) -> str:
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
        cleaned = cleaned.replace("\n", " ")
        return " ".join(cleaned.split())

    def _extract_quoted(self, value: str) -> str | None:
        match = re.search(r'"([^"]+)"', value)
        if not match:
            return None
        quoted = " ".join(match.group(1).split()).strip()
        return quoted or None

    def _describe_event(self, event: ActionEvent) -> str | None:
        for item in event.extracted_content:
            text = item.strip()
            if text.startswith("Navigated to "):
                url = text.replace("Navigated to ", "", 1).strip()
                return f"Open {url}"
            if text.startswith("Clicked "):
                quoted = self._extract_quoted(text)
                if quoted:
                    return f'Click "{quoted}"'
                return "Click the highlighted control"
            if text.startswith("Typed "):
                quoted = self._extract_quoted(text)
                if quoted:
                    return f'Type "{quoted}"'
                return "Type in the active field"
            if text.startswith("Switched to tab"):
                return "Switch to the active tab"
            if text.startswith("Scrolled down"):
                return "Scroll down"
            if text.startswith("Scrolled up"):
                return "Scroll up"
        if not event.action_names:
            return None
        phrases = [
            ACTION_PHRASES.get(name, name.replace("_", " "))
            for name in event.action_names
            if name not in {"wait", "done"}
        ]
        if not phrases:
            return None
        return ", then ".join(phrases)

    def _to_guidance_points(self, events: list[ActionEvent], max_points: int = 4) -> list[str]:
        if max_points <= 0:
            return []
        points: list[str] = []
        seen: set[str] = set()
        for event in events:
            phrase = self._guidance_from_event(event)
            if not phrase:
                continue
            key = " ".join(phrase.lower().split())
            if key in seen:
                continue
            seen.add(key)
            points.append(phrase)
        if not points:
            return []
        if len(points) > max_points:
            if max_points == 1:
                points = [points[0]]
            else:
                count = len(points)
                indices = {
                    round(i * (count - 1) / (max_points - 1))
                    for i in range(max_points)
                }
                points = [points[idx] for idx in sorted(indices)]
        return points

    def _guidance_from_event(self, event: ActionEvent) -> str | None:
        for item in event.extracted_content:
            text = item.strip()
            if text.startswith("Navigated to "):
                url = text.replace("Navigated to ", "", 1).strip()
                return self._navigation_phrase(url)
            if text.startswith("Clicked "):
                quoted = self._extract_quoted(text)
                return self._click_phrase(quoted)
            if text.startswith("Typed "):
                return "enter the required values in the active area"
            if text.startswith("Switched to tab"):
                return "switch to the tab that has the task"
            if text.startswith("Scrolled down") or text.startswith("Scrolled up"):
                return "scroll to the part of the page you need"

        action_set = {name for name in event.action_names if name not in {"wait", "done"}}
        if not action_set:
            return None
        if {"go_to_url", "navigate", "search_google"} & action_set:
            return "open the requested page"
        if {"open_tab", "switch_tab", "switch"} & action_set:
            return "move to the tab where you need to continue"
        if {"input_text", "input", "send_keys"} & action_set:
            return "enter the required information in the active area"
        if {"scroll_down", "scroll_up", "scroll"} & action_set:
            return "scroll to the section that contains the control"
        if {"click", "click_element", "click_element_by_index"} & action_set:
            return "select the next control in the interface"
        return None

    def _navigation_phrase(self, url: str) -> str:
        lowered = url.lower()
        if "docs.google.com" in lowered:
            return "open Google Docs in the editor view"
        if "canva.com" in lowered:
            return "open Canva in the design workspace"
        return "open the target page in the browser"

    def _click_phrase(self, label: str | None) -> str:
        if not label:
            return "select the next control in the interface"
        cleaned = " ".join(label.split()).strip()
        lowered = cleaned.lower()
        if any(token in lowered for token in ("insert", "file", "format", "table", "layout", "menu")):
            return f'open the "{cleaned}" menu'
        if "tab" in lowered:
            return f'select the "{cleaned}" tab'
        if "button" in lowered:
            return f'click the "{cleaned}" button'
        return f'select "{cleaned}"'

    def _compose_guidance_sentences(self, points: list[str]) -> list[str]:
        if not points:
            return []
        connectors = ["First", "Next", "Then", "After that"]
        sentences: list[str] = []
        for index, point in enumerate(points):
            prefix = "Finally" if index == len(points) - 1 and len(points) > 1 else connectors[min(index, len(connectors) - 1)]
            sentence = f"{prefix}, {point}."
            sentence = re.sub(r"\s+", " ", sentence).strip()
            sentences.append(sentence)
        return sentences

    def _summarize_outcome(self, final_result: str) -> str:
        cleaned = self._clean_text(final_result)
        cleaned = re.sub(r"#+\s*", "", cleaned).strip()
        cleaned = cleaned.replace("**", "")
        result_match = re.search(r"\bresult:\s*(.+)", cleaned, flags=re.IGNORECASE)
        candidate = result_match.group(1).strip() if result_match else cleaned
        candidate = re.split(r"(?<=[.!?])\s+", candidate)[0].strip()
        lowered = candidate.lower()
        if any(token in lowered for token in ("workflow complete", "task complete", "demonstration complete")):
            return "the requested workflow completed in the product UI"
        candidate = candidate.rstrip(".")
        if not candidate:
            return "the requested workflow completed in the product UI"
        return candidate

    def _compute_dynamic_speed(
        self,
        current_speed: float,
        current_audio_duration: float,
        target_duration: float,
        min_speed: float,
        max_speed: float,
    ) -> float:
        if current_audio_duration <= 0 or target_duration <= 0:
            return max(min_speed, min(max_speed, current_speed))
        ratio = current_audio_duration / target_duration
        proposed = current_speed * ratio
        return max(min_speed, min(max_speed, proposed))

    def _probe_duration(self, media_path: Path) -> float:
        command = [
            self.settings.ffprobe_bin,
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

    def _compress_script_for_target(
        self,
        script: str,
        current_duration: float,
        target_duration: float,
    ) -> str:
        if current_duration <= 0 or target_duration <= 0:
            return script
        ratio = target_duration / current_duration
        target_chars = max(120, int(len(script) * ratio * 0.98))
        if target_chars >= len(script):
            return script
        return self._truncate_script(script, max_chars=target_chars)

    def _truncate_words(self, script: str, max_words: int) -> str:
        words = script.split()
        if len(words) <= max_words:
            return script
        clipped = " ".join(words[:max_words]).rstrip(" ,;:")
        if not clipped.endswith("."):
            clipped += "."
        return clipped

    def _estimate_word_budget(self, target_duration_seconds: float, speaking_speed: float) -> int:
        # Keep speech understandable while targeting timeline fit.
        bounded_speed = max(1.0, min(1.4, speaking_speed))
        words_per_second = 2.2 * bounded_speed
        return max(8, int(target_duration_seconds * words_per_second))

    def _retime_audio(self, audio_path: Path, tempo_factor: float) -> None:
        if tempo_factor <= 0:
            return
        filters = self._build_atempo_filters(tempo_factor)
        temp_path = audio_path.with_name(f"{audio_path.stem}-retimed{audio_path.suffix}")
        encode_args = self._audio_encode_args(audio_path)
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(audio_path),
            "-filter:a",
            ",".join(filters),
            "-vn",
            *encode_args,
            str(temp_path),
        ]
        subprocess.run(command, check=True)
        temp_path.replace(audio_path)

    def _build_atempo_filters(self, tempo_factor: float) -> list[str]:
        remaining = tempo_factor
        filters: list[str] = []
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.6f}")
        return filters

    def _audio_encode_args(self, audio_path: Path) -> list[str]:
        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            return ["-c:a", "pcm_s16le"]
        return ["-c:a", "mp3", "-b:a", "128k"]
