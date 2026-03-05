from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .models import BrowserCloudMode, GenerationRequest, LLMProvider
from .pipeline import InstructionVideoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate instructional browser videos from prompts.")
    parser.add_argument("--instructions", type=str, help="Prompt with browser instructions.")
    parser.add_argument(
        "--instructions-file",
        type=Path,
        help="Path to a text file containing instructions.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--provider",
        type=str,
        choices=[p.value for p in LLMProvider],
        default=LLMProvider.ANTHROPIC.value,
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-6")
    parser.add_argument(
        "--cloud-mode",
        type=str,
        choices=[m.value for m in BrowserCloudMode],
        default=BrowserCloudMode.CLOUD.value,
    )
    parser.add_argument("--cloud-proxy-country-code", type=str, default="us")
    parser.add_argument("--cloud-profile-id", type=str, default=None)
    parser.add_argument("--no-cloud-fallback", action="store_true")
    parser.add_argument("--use-system-chrome", action="store_true")
    parser.add_argument("--local-user-data-dir", type=Path, default=None)
    parser.add_argument("--local-profile-directory", type=str, default=None)
    parser.add_argument("--start-url", type=str, default=None)
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--max-actions-per-step", type=int, default=5)
    parser.add_argument("--step-timeout", type=int, default=180)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--viewport-width", type=int, default=1920)
    parser.add_argument("--viewport-height", type=int, default=1080)
    parser.add_argument("--browser-video-speed", type=float, default=2.0)
    parser.add_argument("--voice", type=str, default="alloy")
    parser.add_argument("--narration-model", type=str, default="gpt-4o-mini-tts")
    parser.add_argument("--narration-speed", type=float, default=1.0)
    parser.add_argument("--narration-min-speed", type=float, default=1.0)
    parser.add_argument("--narration-max-speed", type=float, default=2.5)
    parser.add_argument("--narration-max-chars", type=int, default=420)
    parser.add_argument("--min-raw-video-seconds", type=float, default=10.0)
    parser.add_argument("--auto-zoom-level", type=float, default=1.8)
    parser.add_argument("--auto-zoom-dwell-threshold", type=float, default=0.5)
    parser.add_argument("--auto-zoom-transition-duration", type=float, default=0.4)
    return parser


def load_instructions(inline: str | None, file_path: Path | None) -> str:
    if inline and inline.strip():
        return inline.strip()
    if file_path:
        return file_path.read_text(encoding="utf-8").strip()
    raise ValueError("Provide --instructions or --instructions-file")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        instructions = load_instructions(args.instructions, args.instructions_file)
        request = GenerationRequest(
            instructions=instructions,
            output_dir=args.output_dir,
            llm_provider=LLMProvider(args.provider),
            llm_model=args.model,
            cloud_mode=BrowserCloudMode(args.cloud_mode),
            cloud_proxy_country_code=args.cloud_proxy_country_code,
            cloud_profile_id=args.cloud_profile_id,
            cloud_fallback_on_challenge=not args.no_cloud_fallback,
            use_system_chrome=args.use_system_chrome,
            local_user_data_dir=args.local_user_data_dir,
            local_profile_directory=args.local_profile_directory,
            start_url=args.start_url,
            max_steps=args.max_steps,
            max_actions_per_step=args.max_actions_per_step,
            step_timeout=args.step_timeout,
            headless=args.headless,
            viewport_width=args.viewport_width,
            viewport_height=args.viewport_height,
            browser_video_speed=args.browser_video_speed,
            narration_voice=args.voice,
            narration_model=args.narration_model,
            narration_speed=args.narration_speed,
            narration_min_speed=args.narration_min_speed,
            narration_max_speed=args.narration_max_speed,
            narration_max_chars=args.narration_max_chars,
            min_raw_video_seconds=args.min_raw_video_seconds,
            auto_zoom_level=args.auto_zoom_level,
            auto_zoom_dwell_threshold=args.auto_zoom_dwell_threshold,
            auto_zoom_transition_duration=args.auto_zoom_transition_duration,
            job_name=args.job_name,
        )
        pipeline = InstructionVideoPipeline()
        result = pipeline.generate_sync(request)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = {
        "job_dir": str(result.job_dir),
        "final_video_path": str(result.video_artifacts.final_video_path),
        "manifest_path": str(result.manifest_path),
        "narration_script_path": str(result.narration_artifacts.script_path),
        "narration_audio_path": str(result.narration_artifacts.audio_path),
        "browser_video_path": str(result.browser_artifacts.video_path),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
