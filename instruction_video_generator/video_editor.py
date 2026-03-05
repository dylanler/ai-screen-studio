from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .models import (
    BrowserRunArtifacts,
    GenerationRequest,
    NarrationArtifacts,
    VideoEditArtifacts,
    VideoSegment,
)
from .settings import Settings


@dataclass(slots=True)
class AutoZoomRegion:
    start_time: float
    end_time: float
    zoom_in_time: float
    hold_end_time: float
    zoom_out_time: float
    center_x: float
    center_y: float
    zoom_level: float


class VideoEditor:
    def __init__(self, settings: Settings, zoom_factor: float = 1.14):
        self.settings = settings
        self.zoom_factor = zoom_factor

    def render(
        self,
        request: GenerationRequest,
        browser_artifacts: BrowserRunArtifacts,
        narration_artifacts: NarrationArtifacts,
        job_dir: Path,
    ) -> VideoEditArtifacts:
        final_dir = job_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        output_path = final_dir / "instructional_video.mp4"
        end_padding_seconds = 0.35
        audio_duration = self.probe_duration(narration_artifacts.audio_path)
        normalized_video_path = self.normalize_video_geometry(
            browser_artifacts.video_path,
            request.viewport_width,
            request.viewport_height,
            final_dir / "normalized-video.mp4",
        )
        prepared_video_path = self.extend_video_to_duration(
            normalized_video_path,
            audio_duration + end_padding_seconds,
            final_dir / "prepared-video.mp4",
        )
        duration = self.probe_duration(prepared_video_path)
        target_duration = max(duration, audio_duration + end_padding_seconds)
        width, height = self.probe_geometry(prepared_video_path)
        action_points = self._relative_action_points(
            browser_artifacts.events,
            time_scale=request.browser_video_speed,
        )
        segments = self.build_segments(
            duration=duration,
            action_points=action_points,
            zoom_level=request.auto_zoom_level,
            dwell_threshold=request.auto_zoom_dwell_threshold,
            transition_duration=request.auto_zoom_transition_duration,
        )
        filter_complex, video_label = self.build_video_filter(
            segments,
            width,
            height,
        )
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(prepared_video_path),
            "-i",
            str(narration_artifacts.audio_path),
            "-filter_complex",
            filter_complex,
            "-map",
            video_label,
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-af",
            f"apad=pad_dur={target_duration:.3f},loudnorm=I=-16:LRA=11:TP=-1.5",
            "-t",
            f"{target_duration:.3f}",
            str(output_path),
        ]
        self.run_command(command)
        return VideoEditArtifacts(
            final_video_path=output_path,
            segments=segments,
            duration_seconds=duration,
            ffmpeg_command=command,
        )

    def speed_adjust_video(
        self,
        video_path: Path,
        speed_factor: float,
        output_path: Path,
    ) -> Path:
        if speed_factor <= 0:
            raise ValueError("speed_factor must be > 0")
        if abs(speed_factor - 1.0) < 1e-6:
            return video_path
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"setpts=PTS/{speed_factor:.6f}",
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
        self.run_command(command)
        return output_path

    def probe_duration(self, video_path: Path) -> float:
        command = [
            self.settings.ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())

    def extend_video_to_duration(
        self,
        video_path: Path,
        target_duration: float,
        output_path: Path,
    ) -> Path:
        source_duration = self.probe_duration(video_path)
        if source_duration >= target_duration:
            return video_path
        extra = max(0.01, target_duration - source_duration)
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"tpad=stop_mode=clone:stop_duration={extra:.3f}",
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
        self.run_command(command)
        return output_path

    def probe_geometry(self, video_path: Path) -> tuple[int, int]:
        command = [
            self.settings.ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        width_text, height_text = result.stdout.strip().split("x")
        return int(width_text), int(height_text)

    def normalize_video_geometry(
        self,
        video_path: Path,
        target_width: int,
        target_height: int,
        output_path: Path,
    ) -> Path:
        source_width, source_height = self.probe_geometry(video_path)
        if source_width == target_width and source_height == target_height:
            return video_path

        source_ratio = source_width / source_height
        target_ratio = target_width / target_height
        if source_ratio >= target_ratio:
            scale_height = target_height
            scale_width = max(
                target_width,
                self._even_int(source_width * target_height / source_height),
            )
        else:
            scale_width = target_width
            scale_height = max(
                target_height,
                self._even_int(source_height * target_width / source_width),
            )

        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            (
                f"scale={scale_width}:{scale_height},"
                f"crop={target_width}:{target_height},"
                "setsar=1"
            ),
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
        self.run_command(command)
        return output_path

    def _even_int(self, value: float) -> int:
        rounded = int(round(value))
        if rounded % 2:
            rounded += 1
        return max(2, rounded)

    def build_segments(
        self,
        duration: float,
        action_times: list[float] | None = None,
        action_points: list[tuple[float, float, float]] | None = None,
        zoom_level: float = 1.8,
        dwell_threshold: float = 0.5,
        transition_duration: float = 0.4,
    ) -> list[VideoSegment]:
        if duration <= 0:
            raise ValueError("duration must be > 0")
        if action_points is None:
            action_points = []
            for t in action_times or []:
                action_points.append((t, 0.5, 0.5))

        auto_regions = self._build_auto_zoom_regions(
            duration=duration,
            action_points=action_points,
            zoom_level=zoom_level,
            dwell_threshold=dwell_threshold,
            transition_duration=transition_duration,
        )

        segments: list[VideoSegment] = []
        cursor = 0.0
        min_segment = 0.06
        for region in auto_regions:
            if region.zoom_in_time - cursor >= min_segment:
                segments.append(
                    VideoSegment(
                        start=cursor,
                        end=region.zoom_in_time,
                        is_zoomed=False,
                        phase="normal",
                        center_x=0.5,
                        center_y=0.5,
                        zoom_level=1.0,
                    )
                )
            if region.start_time - region.zoom_in_time >= min_segment:
                segments.append(
                    VideoSegment(
                        start=region.zoom_in_time,
                        end=region.start_time,
                        is_zoomed=True,
                        phase="zoom_in",
                        center_x=region.center_x,
                        center_y=region.center_y,
                        zoom_level=region.zoom_level,
                    )
                )
            if region.hold_end_time - region.start_time >= min_segment:
                segments.append(
                    VideoSegment(
                        start=region.start_time,
                        end=region.hold_end_time,
                        is_zoomed=True,
                        phase="zoom_hold",
                        center_x=region.center_x,
                        center_y=region.center_y,
                        zoom_level=region.zoom_level,
                    )
                )
            if region.zoom_out_time - region.hold_end_time >= min_segment:
                segments.append(
                    VideoSegment(
                        start=region.hold_end_time,
                        end=region.zoom_out_time,
                        is_zoomed=True,
                        phase="zoom_out",
                        center_x=region.center_x,
                        center_y=region.center_y,
                        zoom_level=region.zoom_level,
                    )
                )
            cursor = max(cursor, region.zoom_out_time)
        if duration - cursor >= 0.08:
            segments.append(
                VideoSegment(
                    start=cursor,
                    end=duration,
                    is_zoomed=False,
                    phase="normal",
                    center_x=0.5,
                    center_y=0.5,
                    zoom_level=1.0,
                )
            )
        if not segments:
            segments.append(
                VideoSegment(
                    start=0.0,
                    end=duration,
                    is_zoomed=False,
                    phase="normal",
                    center_x=0.5,
                    center_y=0.5,
                    zoom_level=1.0,
                )
            )
        return segments

    def build_video_filter(
        self,
        segments: list[VideoSegment],
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> tuple[str, str]:
        chains: list[str] = []
        labels: list[str] = []
        zoom = f"{self.zoom_factor:.4f}"
        for index, segment in enumerate(segments):
            label = f"v{index}"
            labels.append(f"[{label}]")
            chain = (
                f"[0:v]trim=start={segment.start:.3f}:end={segment.end:.3f},"
                "setpts=PTS-STARTPTS"
            )
            if segment.is_zoomed and source_width and source_height:
                if segment.phase == "zoom_hold":
                    chain += self._zoom_filter_static(
                        source_width=source_width,
                        source_height=source_height,
                        center_x=segment.center_x,
                        center_y=segment.center_y,
                        zoom_level=segment.zoom_level if segment.zoom_level > 1 else float(zoom),
                    )
                elif segment.phase == "zoom_in":
                    chain += self._zoom_filter_animated(
                        source_width=source_width,
                        source_height=source_height,
                        center_x=segment.center_x,
                        center_y=segment.center_y,
                        zoom_level=segment.zoom_level if segment.zoom_level > 1 else float(zoom),
                        duration=max(0.001, segment.end - segment.start),
                        zoom_in=True,
                    )
                elif segment.phase == "zoom_out":
                    chain += self._zoom_filter_animated(
                        source_width=source_width,
                        source_height=source_height,
                        center_x=segment.center_x,
                        center_y=segment.center_y,
                        zoom_level=segment.zoom_level if segment.zoom_level > 1 else float(zoom),
                        duration=max(0.001, segment.end - segment.start),
                        zoom_in=False,
                    )
                else:
                    chain += self._zoom_filter_static(
                        source_width=source_width,
                        source_height=source_height,
                        center_x=segment.center_x,
                        center_y=segment.center_y,
                        zoom_level=segment.zoom_level if segment.zoom_level > 1 else float(zoom),
                    )
            chain += f"[{label}]"
            chains.append(chain)
        if len(labels) == 1:
            chains.append(f"{labels[0]}null[vbase]")
        else:
            chains.append("".join(labels) + f"concat=n={len(labels)}:v=1:a=0[vbase]")
        chains.append("[vbase]null[vout]")
        return ";".join(chains), "[vout]"

    def _relative_action_points(self, events, time_scale: float = 1.0) -> list[tuple[float, float, float]]:
        timed_events = [event for event in events if event.started_at is not None and event.action_names]
        if not timed_events:
            return []
        baseline = min(event.started_at for event in timed_events if event.started_at is not None)
        if time_scale <= 0:
            time_scale = 1.0
        output: list[tuple[float, float, float]] = []
        for event in timed_events:
            if event.started_at is None:
                continue
            rel = (event.started_at - baseline) / time_scale
            if rel >= 0:
                output.append((rel, event.center_x if event.center_x is not None else 0.5, event.center_y if event.center_y is not None else 0.5))
        return sorted(output, key=lambda value: value[0])

    def _build_auto_zoom_regions(
        self,
        duration: float,
        action_points: list[tuple[float, float, float]],
        zoom_level: float,
        dwell_threshold: float,
        transition_duration: float,
    ) -> list[AutoZoomRegion]:
        filtered = [(t, cx, cy) for t, cx, cy in action_points if 0 <= t <= duration]
        if not filtered:
            return []
        hold_duration = max(dwell_threshold, 0.5)

        regions: list[dict[str, float]] = []
        for t, cx, cy in sorted(filtered, key=lambda value: value[0]):
            if regions:
                last = regions[-1]
                if t - last["end_time"] < dwell_threshold:
                    count = last["count"] + 1
                    last["center_x"] = (last["center_x"] * last["count"] + cx) / count
                    last["center_y"] = (last["center_y"] * last["count"] + cy) / count
                    last["end_time"] = t
                    last["count"] = count
                    continue
            regions.append(
                {
                    "start_time": t,
                    "end_time": t,
                    "center_x": cx,
                    "center_y": cy,
                    "count": 1,
                }
            )

        merged: list[dict[str, float]] = []
        for region in regions:
            if merged:
                last = merged[-1]
                last_hold_end = max(last["end_time"], last["start_time"] + hold_duration)
                last_zoom_out = min(duration, last_hold_end + transition_duration)
                this_zoom_in = max(0.0, region["start_time"] - transition_duration)
                if last_zoom_out > this_zoom_in:
                    total = last["count"] + region["count"]
                    last["center_x"] = (last["center_x"] * last["count"] + region["center_x"] * region["count"]) / total
                    last["center_y"] = (last["center_y"] * last["count"] + region["center_y"] * region["count"]) / total
                    last["end_time"] = region["end_time"]
                    last["count"] = total
                    continue
            merged.append(dict(region))

        output: list[AutoZoomRegion] = []
        for region in merged:
            hold_end = max(region["end_time"], region["start_time"] + hold_duration)
            zoom_in_time = max(0.0, region["start_time"] - transition_duration)
            zoom_out_time = min(duration, hold_end + transition_duration)
            output.append(
                AutoZoomRegion(
                    start_time=region["start_time"],
                    end_time=region["end_time"],
                    zoom_in_time=zoom_in_time,
                    hold_end_time=hold_end,
                    zoom_out_time=zoom_out_time,
                    center_x=region["center_x"],
                    center_y=region["center_y"],
                    zoom_level=zoom_level,
                )
            )
        return output

    def _zoom_filter_static(
        self,
        source_width: int,
        source_height: int,
        center_x: float,
        center_y: float,
        zoom_level: float,
    ) -> str:
        return (
            f",scale=iw*{zoom_level:.5f}:ih*{zoom_level:.5f},"
            f"crop={source_width}:{source_height}:"
            f"x='min(max(0,{center_x:.6f}*(iw-{source_width})),iw-{source_width})':"
            f"y='min(max(0,{center_y:.6f}*(ih-{source_height})),ih-{source_height})'"
        )

    def _zoom_filter_animated(
        self,
        source_width: int,
        source_height: int,
        center_x: float,
        center_y: float,
        zoom_level: float,
        duration: float,
        zoom_in: bool,
    ) -> str:
        progress = f"min(max(t/{duration:.6f},0),1)"
        smooth = f"({progress}*{progress}*(3-2*{progress}))"
        if zoom_in:
            zoom_expr = f"1+({zoom_level:.6f}-1)*{smooth}"
        else:
            zoom_expr = f"{zoom_level:.6f}-({zoom_level:.6f}-1)*{smooth}"
        return (
            f",scale='iw*({zoom_expr})':'ih*({zoom_expr})':eval=frame,"
            f"crop={source_width}:{source_height}:"
            f"x='min(max(0,{center_x:.6f}*(iw-{source_width})),iw-{source_width})':"
            f"y='min(max(0,{center_y:.6f}*(ih-{source_height})),ih-{source_height})'"
        )

    def run_command(self, command: list[str]) -> None:
        subprocess.run(command, check=True)

    def segments_to_dicts(self, segments: list[VideoSegment]) -> list[dict]:
        return [asdict(segment) for segment in segments]
