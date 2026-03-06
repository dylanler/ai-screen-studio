from instruction_video_generator.settings import Settings
from instruction_video_generator.video_editor import VideoEditor


def test_build_segments_generates_zoom_and_normal_blocks():
    editor = VideoEditor(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    segments = editor.build_segments(duration=12.0, action_times=[1.0, 1.3, 6.5])
    assert segments[0].is_zoomed is False
    assert any(segment.is_zoomed for segment in segments)
    assert any(segment.phase == "zoom_in" for segment in segments)
    assert any(segment.phase == "zoom_hold" for segment in segments)
    assert any(segment.phase == "zoom_out" for segment in segments)
    assert abs(segments[0].start - 0.0) < 1e-6
    assert abs(segments[-1].end - 12.0) < 1e-6


def test_build_video_filter_single_segment():
    editor = VideoEditor(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    segments = editor.build_segments(duration=5.0, action_times=[])
    filter_complex, label = editor.build_video_filter(segments)
    assert "[0:v]trim" in filter_complex
    assert "null[vout]" in filter_complex
    assert label == "[vout]"


def test_build_target_duration_extends_for_longer_narration():
    editor = VideoEditor(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    target = editor.build_target_duration(video_duration=8.0, audio_duration=12.0)
    assert abs(target - 13.1) < 1e-6


def test_build_target_duration_preserves_longer_video():
    editor = VideoEditor(
        Settings(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    target = editor.build_target_duration(video_duration=20.0, audio_duration=12.0)
    assert abs(target - 20.0) < 1e-6
