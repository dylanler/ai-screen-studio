# AI Screen Studio

AI Screen Studio is an instructional video generator.

Input: a natural-language prompt (for example: "show me how to create a table in Google Docs").

Output: a rendered tutorial video with:
- browser automation capture (Browser Use local/cloud)
- conversational narration (OpenAI TTS)
- timing alignment between video and narration
- auto zoom/editing for watchable walkthroughs

## Core Flow

1. Browser agent executes the prompt in a real browser session.
2. Raw browser video is recorded.
3. Narration script is generated from workflow guidance (not click-by-click literal logs).
4. Narration audio is synthesized and dynamically time-aligned.
5. Final composition is rendered with zoom effects and mixed audio.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` on PATH
- API keys in `.env`

Required env vars:
- `OPENAI_API_KEY` (TTS narration)
- `ANTHROPIC_API_KEY` (default browser agent LLM)
- `BROWSER_USE_API_KEY` (required for cloud mode)

Optional:
- `GEMINI_API_KEY`
- `BROWSER_USE_DEFAULT_PROFILE_ID` (defaults to `536cd6ff-add0-4b96-a4e7-c8794254a4cc` if unset)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## CLI Usage

```bash
source .venv/bin/activate
instruction-video-generator \
  --instructions "Show me how to create a table in Google Docs" \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --cloud-mode cloud \
  --cloud-profile-id 536cd6ff-add0-4b96-a4e7-c8794254a4cc \
  --output-dir outputs \
  --job-name gdocs-table-demo
```

Important options:
- `--cloud-mode cloud|local|auto` (default: `cloud`)
- `--browser-video-speed` (default: `2.0`)
- `--narration-min-speed` (default: `1.0`)
- `--narration-max-speed` (default: `2.5`)

## Web Chat UI

Run:

```bash
source .venv/bin/activate
instruction-video-web --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010`.

UI features:
- prompt chat for generating the next instructional video
- animated queue states: `Queued`, `Browser Run`, `Narration`, `Render`, `Completed`
- in-app final video preview
- default cloud profile ID pre-filled as `536cd6ff-add0-4b96-a4e7-c8794254a4cc`

## Output Structure

Each run writes to `outputs/<job-name>/`:
- `raw_browser_video/`
- `artifacts/agent_history.json`
- `artifacts/action_events.json`
- `narration/narration_script.txt`
- `narration/narration.wav`
- `final/instructional_video.mp4`
- `manifest.json`

## Tests

```bash
source .venv/bin/activate
pytest
```

## License

MIT
