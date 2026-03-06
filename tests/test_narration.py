from instruction_video_generator.models import ActionEvent
from instruction_video_generator.narration import NarrationService
from instruction_video_generator.settings import Settings


def test_build_script_with_actions_and_result():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
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
        ),
        ActionEvent(
            step_number=2,
            started_at=1.2,
            ended_at=2.0,
            url="https://example.com/docs",
            title="Docs",
            action_names=["input_text"],
            actions=[],
            success=True,
            extracted_content=[],
            errors=[],
        ),
    ]
    script = service.build_script("Open docs and search", events, "Done successfully")
    assert "Let's walk through it together." in script
    assert "First," in script
    assert "open the requested page" in script
    assert "At the end, you should see Done successfully." in script
    assert "Step 1:" not in script


def test_build_script_falls_back_to_instruction_summary():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    script = service.build_script("Click settings then save", [], None)
    assert "Let's walk through this flow: Click settings then save." in script
    assert "At the end, you should see the workflow completed successfully." in script


def test_build_script_is_truncated():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    script = service.build_script("x " * 600, [], None, max_chars=150)
    assert len(script) <= 150


def test_compute_dynamic_speed_speeds_up_when_audio_is_too_long():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    speed = service._compute_dynamic_speed(
        current_speed=1.0,
        current_audio_duration=20.0,
        target_duration=10.0,
        min_speed=0.8,
        max_speed=2.5,
    )
    assert speed == 2.0


def test_compute_dynamic_speed_clamps_to_bounds():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    speed_fast = service._compute_dynamic_speed(
        current_speed=1.0,
        current_audio_duration=40.0,
        target_duration=10.0,
        min_speed=1.0,
        max_speed=2.5,
    )
    speed_slow = service._compute_dynamic_speed(
        current_speed=1.0,
        current_audio_duration=2.0,
        target_duration=10.0,
        min_speed=1.0,
        max_speed=2.5,
    )
    assert speed_fast == 2.5
    assert speed_slow == 1.0


def test_compute_dynamic_speed_never_slows_down_requested_speed():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    speed = service._compute_dynamic_speed(
        current_speed=1.4,
        current_audio_duration=3.0,
        target_duration=10.0,
        min_speed=1.0,
        max_speed=2.5,
    )
    assert speed == 1.4


def test_generate_audio_with_dynamic_speed_does_not_slow_or_retime_short_audio(tmp_path, monkeypatch):
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    output_path = tmp_path / "narration.wav"
    generated_speeds: list[float] = []
    retime_calls: list[float] = []

    def fake_generate_audio(*, script_text, output_path, voice, model, speed):
        generated_speeds.append(speed)
        output_path.write_bytes(b"fake")

    def fake_probe_duration(_path):
        return 4.0

    def fake_retime_audio(_path, tempo_factor):
        retime_calls.append(tempo_factor)

    monkeypatch.setattr(service, "generate_audio", fake_generate_audio)
    monkeypatch.setattr(service, "_probe_duration", fake_probe_duration)
    monkeypatch.setattr(service, "_retime_audio", fake_retime_audio)

    script_text, speed_used, duration_seconds = service.generate_audio_with_dynamic_speed(
        script_text="Short narration",
        output_path=output_path,
        voice="alloy",
        model="gpt-4o-mini-tts",
        requested_speed=1.0,
        min_speed=1.0,
        max_speed=2.5,
        target_duration_seconds=10.0,
    )

    assert script_text == "Short narration"
    assert speed_used == 1.0
    assert duration_seconds == 4.0
    assert generated_speeds == [1.0]
    assert retime_calls == []


def test_build_atempo_filters_splits_large_tempo():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    filters = service._build_atempo_filters(4.0)
    assert filters == ["atempo=2.0", "atempo=2.000000"]


def test_to_instructional_steps_downsamples_in_order():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    events = []
    for idx in range(6):
        events.append(
            ActionEvent(
                step_number=idx + 1,
                started_at=float(idx),
                ended_at=float(idx) + 0.5,
                url=None,
                title=None,
                action_names=["click_element_by_index"],
                actions=[],
                success=True,
                extracted_content=[f'Clicked "Button {idx + 1}"'],
                errors=[],
            )
        )
    steps = service._to_instructional_steps(events, max_steps=3)
    assert len(steps) == 3
    assert 'Step 1: Click "Button 1".' in steps[0]
    assert 'Step 2: Click "Button 3".' in steps[1]
    assert 'Step 3: Click "Button 6".' in steps[2]


def test_estimate_word_budget_short_video_is_constrained():
    service = NarrationService(
        Settings(
            openai_api_key="test",
            anthropic_api_key=None,
            gemini_api_key=None,
            browser_use_api_key=None,
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
        )
    )
    budget = service._estimate_word_budget(target_duration_seconds=4.0, speaking_speed=1.0)
    assert budget == 8
