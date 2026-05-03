# Bike Editor

Bike Editor is a Python tool that turns long motorcycle camera recordings into a shorter YouTube-ready motovlog.

It was built for the common action-camera workflow where one ride becomes many video files and each file overlaps a little with the next one. The script joins the clips, removes the duplicated overlap, analyzes the ride, selects the best moments, renders a final edit, and prepares YouTube metadata.

![Bike Editor process diagram](processo_video_youtube.jpg)

## What It Does

- Finds all videos in an input folder.
- Joins them into one continuous source while removing the fixed camera overlap.
- Improves image and audio with FFmpeg filters.
- Transcribes narration locally with `faster-whisper`.
- Scores candidate segments using narration density and visual changes.
- Uses the OpenAI API, when configured, to choose the final edit plan.
- Renders a final MP4 with the selected segments.
- Generates a thumbnail from a real frame plus an AI-created overlay.
- Writes `youtube.md` with title, description, tags, and thumbnail text.

## Requirements

- Python 3.10+
- FFmpeg and FFprobe available in your terminal
- An OpenAI API key if you want AI-assisted edit decisions and thumbnails

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS, FFmpeg can be installed with:

```bash
brew install ffmpeg
```

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Then edit `.env` with your own values.

Important settings:

- `OPENAI_API_KEY`: your OpenAI API key.
- `CAMERA_OVERLAP_SECONDS`: how much duplicated video exists at the beginning of each clip after the first one.
- `AI_MODEL`: model used for edit decisions and YouTube metadata.
- `IMAGE_MODEL`: model used for thumbnail overlay generation.
- `CREATOR_*`, `PREVIOUS_MOTORCYCLE_*`, `CURRENT_MOTORCYCLE_*`: optional channel context used so the AI can avoid factual mistakes.
- `VIDEO_ENHANCE_FILTER` and `AUDIO_ENHANCE_FILTER`: FFmpeg filters for image and audio cleanup.

`.env` is ignored by Git, so your API key and personal context should not be committed.

## Usage

Basic run:

```bash
python moto_editor.py \
  --input "/path/to/videos" \
  --target-minutes 10 \
  --output-dir "./outputs/my-ride"
```

With an optional title and description:

```bash
python moto_editor.py \
  --input "/path/to/videos" \
  --target-minutes 10 \
  --title "Primeiro passeio com a moto nova" \
  --description "Um rolê curto para sentir a moto na cidade." \
  --output-dir "./outputs/my-ride"
```

If title or description are provided, the AI reviews and improves them while preserving the intent. If they are omitted, the AI generates them from the video context.

## Outputs

For a 10-minute target, the output folder will contain files like:

- `moto_editado_10min.mp4`: final edited video.
- `thumbnail_openai_10min.png`: generated thumbnail, when OpenAI image generation is enabled.
- `youtube.md`: title, description, tags, and thumbnail text.
- `edit_plan_10min.json`: structured edit plan.
- `edit_plan_10min.csv`: selected cuts in spreadsheet-friendly format.
- `transcript.json`: local transcription.
- `candidates.json`: scored candidate segments.

The `outputs/` folder is ignored by Git because rendered videos and work files can be very large.

## Caching

The script reuses work already present in the output folder:

- Existing transcription is reused.
- Visual analysis is reused.
- Candidate segment scoring is reused.
- A matching edit plan is reused for the same target duration.
- Existing rendered videos and thumbnails are reused unless forced.

Useful flags:

```bash
--force-analysis    # recompute transcript, visual samples, and candidates
--force-ai          # ask OpenAI for a fresh edit plan
--force-render      # render segments and final video again
--force-thumbnail   # generate a new thumbnail
--no-openai         # run without OpenAI
--local-thumbnail   # skip OpenAI thumbnail and use local text overlay
```

## Notes

ChatGPT Plus and OpenAI API billing are separate products. This project uses the OpenAI API, so you need an API key with available billing or credits.
