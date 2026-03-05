from pathlib import Path

import pytest

from instruction_video_generator.models import GenerationRequest


def test_generation_request_default_narration_min_speed():
    request = GenerationRequest(
        instructions="Open example.com and click sign in",
        output_dir=Path("."),
    )
    assert request.narration_min_speed == 1.0


def test_generation_request_rejects_narration_min_speed_below_one():
    request = GenerationRequest(
        instructions="Open example.com and click sign in",
        output_dir=Path("."),
        narration_min_speed=0.95,
    )
    with pytest.raises(ValueError, match="narration_min_speed must be >= 1.0"):
        request.validate()
