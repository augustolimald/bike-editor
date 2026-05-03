# AGENTS.md

This file gives maintainers and coding agents project-specific context. For user-facing documentation, update `README.md`.

## Project

Bike Editor is a Python CLI for turning long motorcycle ride recordings into a short YouTube-ready video. The main entry point is `moto_editor.py`.

The pipeline:

1. Discover camera clips in an input folder.
2. Remove fixed overlap from the beginning of every clip after the first one.
3. Concatenate clips into a clean source with FFmpeg.
4. Extract and clean narration audio.
5. Transcribe locally with `faster-whisper`.
6. Analyze visual changes with OpenCV.
7. Build candidate windows and score them with narration, movement, stopped/idling, and visual metrics.
8. Use OpenAI, when enabled, to choose the edit plan and generate YouTube metadata.
9. Render selected segments with FFmpeg filters.
10. Generate a thumbnail using a real frame plus an AI-created overlay.

## Configuration

Do not hard-code personal channel details, model choices, filter tuning, API keys, or camera parameters in Python code. Put configurable values in `.env` and document them in `.env.example`.

Current `.env`-driven settings include:

- `OPENAI_API_KEY`
- `CAMERA_OVERLAP_SECONDS`
- `AI_MODEL`
- `IMAGE_MODEL`
- `RENDER_PRESET`
- `RENDER_CRF`
- `AUDIO_BITRATE`
- `VIDEO_ENHANCE_FILTER`
- `AUDIO_ENHANCE_FILTER`
- `CREATOR_CONTEXT`
- `CREATOR_NAME`
- `CREATOR_AGE`
- `CREATOR_INTERESTS`
- `PREVIOUS_MOTORCYCLE`
- `PREVIOUS_MOTORCYCLE_CONTEXT`
- `CURRENT_MOTORCYCLE`
- `CURRENT_MOTORCYCLE_CONTEXT`
- `CONTENT_CONTEXT_GUIDANCE`

`.env` must stay ignored by Git.

## Generated Files

Keep generated videos, working folders, and outputs out of version control. The `outputs/` folder is ignored and is the expected location for rendered test runs.

Typical output files:

- `moto_editado_XXmin.mp4`
- `thumbnail_openai_XXmin.png`
- `youtube.md`
- `edit_plan_XXmin.json`
- `edit_plan_XXmin.csv`
- `transcript.json`
- `candidates.json`
- `_work/`

## Development

Use focused changes. This is currently a single-file CLI with one helper script for the process diagram:

- `moto_editor.py`: application logic.
- `tools/make_process_diagram.py`: regenerates `processo_video_youtube.jpg`.
- `processo_video_youtube.jpg`: public diagram linked by the README.

Before committing Python changes, run:

```bash
PYTHONPYCACHEPREFIX=.pycache .venv/bin/python -m py_compile moto_editor.py tools/make_process_diagram.py
```

For dependency checks:

```bash
.venv/bin/python -c "import cv2, numpy, PIL, openai, faster_whisper; print('python_deps_ok')"
```

## OpenAI Usage

The script uses the OpenAI API only when `OPENAI_API_KEY` is present and `--no-openai` is not used.

OpenAI calls are used for:

- Selecting an edit plan from local candidate segments.
- Reviewing or generating title, description, tags, and thumbnail text.
- Creating a transparent thumbnail overlay.

The script should continue to support a local fallback via heuristics and local thumbnail generation.

## Edit Selection

Candidate scoring should preserve this priority order:

1. Narrated moments with complete spoken ideas.
2. Non-narrated moments where the motorcycle is clearly moving.
3. Visual variety only after narration and movement are considered.

Avoid selecting repeated stopped/idling segments such as traffic lights unless the narration in that segment is important. OpenAI receives `speech_score`, `movement_score`, `stopped_score`, and `visual_score`; keep those fields useful and auditable in `edit_plan_XXmin.csv`.

After plan selection, preserve complete speech by expanding narrated cuts to include the full overlapping transcript segments, then merge adjacent/overlapping cuts. Bump the cached plan format version whenever changing this post-processing behavior.

## Thumbnail Behavior

The thumbnail should preserve the original video frame. The image model should create only an overlay layer with text and minimal contextually justified graphic elements. Avoid prompts that let the model recreate the motorcycle dashboard, rider, road, or scenery.

## README

`README.md` is public-facing. Keep it clear for a random GitHub visitor:

- What the tool does.
- Requirements.
- Setup.
- `.env` configuration.
- Basic usage.
- Output files.
- Cache and force flags.

Do not put personal API keys or private local paths in `README.md`.
