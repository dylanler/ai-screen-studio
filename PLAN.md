# AI Instruction Video Generator Plan

## Product Goal
Transform a natural-language instruction prompt into a polished instructional video:
1. Run browser actions from prompt text.
2. Capture automation as a browser recording.
3. Generate spoken narration from the executed flow.
4. Produce a final edited video suitable for end-user guidance.

## Architecture

### 1) Orchestration Layer
- Module: `instruction_video_generator/pipeline.py`
- Responsibility:
  - Create job workspace.
  - Sequence automation, narration, and editing stages.
  - Persist a run manifest with outputs and metadata.

### 2) Browser Automation Layer
- Module: `instruction_video_generator/browser_runner.py`
- Technology:
  - `browser-use` `Agent`
  - `BrowserSession` + `BrowserProfile`
- Responsibility:
  - Convert instruction prompt into an executable browser task.
  - Execute with selected LLM provider (`openai`, `anthropic`, `gemini`).
  - Record browser video (`record_video_dir`).
  - Save action history (`agent_history.json`) and normalized action events (`action_events.json`).

### 3) Narration Layer
- Module: `instruction_video_generator/narration.py`
- Technology:
  - OpenAI Speech API (text-to-speech)
- Responsibility:
  - Build deterministic narration text from executed action events.
  - Generate MP3 narration.
  - Persist narration script + audio artifacts.

### 4) Editing Layer
- Module: `instruction_video_generator/video_editor.py`
- Technology:
  - `ffmpeg` + `ffprobe`
- Responsibility:
  - Build zoom-pulse segments around interaction timestamps.
  - Stitch video segments and mux narration audio.
  - Produce final MP4 export.

### 5) CLI Layer
- Module: `instruction_video_generator/cli.py`
- Responsibility:
  - Accept prompt and runtime options.
  - Trigger the pipeline.
  - Emit final artifact locations as JSON.

## Data/Artifact Contract
Each run writes to `output_dir/<job_name_or_timestamp>/`:
- `raw_browser_video/` raw browser recording files
- `artifacts/agent_history.json` raw browser-use history
- `artifacts/action_events.json` normalized timeline events
- `narration/narration_script.txt`
- `narration/narration.mp3`
- `final/instructional_video.mp4`
- `manifest.json` canonical metadata and paths

## Testing Strategy

### Unit Tests
1. Narration script generation:
- Validates action-to-speech conversion and fallback behavior.
2. Video edit timeline segmentation:
- Validates merged zoom windows and contiguous segment generation.
3. LLM factory:
- Validates provider selection and missing-key failures.

### Pipeline Tests
1. Orchestration happy path (fully mocked boundaries):
- Asserts pipeline order, artifact creation, and manifest integrity.
2. Job naming behavior:
- Asserts deterministic job directory naming.

### Scope for Live E2E
- Kept optional because live browser automation + external model calls are costful and flaky for CI.
- Core E2E contract is covered through deterministic mocks + strong artifact assertions.

## Acceptance Criteria
1. Input is a natural-language browser instruction prompt.
2. Browser automation executes through `browser-use`.
3. Raw automation recording is persisted.
4. Narration script + MP3 are generated.
5. Final MP4 is produced by muxing edited video and narration.
6. Final run manifest contains all artifacts and runtime metadata.
7. Automated test suite passes locally (`pytest`).

## Delivery Constraints
- Preserve existing Swift/macOS app codepaths.
- Ship as an additive sidecar pipeline in this repository.
- Use `.env` provider keys without hardcoding secrets.
