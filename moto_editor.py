#!/usr/bin/env python3
"""
Automatic motorcycle ride editor.

Pipeline:
1. Read all camera clips from an input folder.
2. Remove the fixed overlap between clips.
3. Build a continuous clean source.
4. Analyze narration, audio energy, and visual changes locally.
5. Ask OpenAI to choose the strongest segments when an API key is present.
6. Render the selected cuts with video/audio enhancement filters.
7. Generate a YouTube thumbnail, using OpenAI image generation when available.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mts", ".m2ts", ".avi"}

DEFAULT_CAMERA_OVERLAP_SECONDS = 2.0
DEFAULT_AI_MODEL = "gpt-5.5"
DEFAULT_IMAGE_MODEL = "gpt-image-1.5"

DEFAULT_VIDEO_ENHANCE_FILTER = (
    "hqdn3d=1.0:1.0:3:3,"
    "eq=contrast=1.06:brightness=0.01:saturation=1.12:gamma=0.99,"
    "unsharp=5:5:0.45:3:3:0.20"
)

DEFAULT_AUDIO_ENHANCE_FILTER = (
    "highpass=f=100,"
    "lowpass=f=8500,"
    "afftdn=nf=-22,"
    "acompressor=threshold=-18dB:ratio=2.6:attack=12:release=180,"
    "volume=1.6,"
    "alimiter=limit=0.95"
)

DEFAULT_RENDER_PRESET = "fast"
DEFAULT_RENDER_CRF = 22
DEFAULT_AUDIO_BITRATE = "128k"
OUTPUT_VERSION = 2


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class CandidateSegment:
    id: int
    start: float
    end: float
    score: float
    visual_score: float
    speech_score: float
    audio_score: float
    reason: str
    transcript: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class EditPlan:
    title: str
    description: str
    thumbnail_text: str
    tags: list[str]
    segments: list[CandidateSegment]
    notes: str
    requested_title: str = ""
    requested_description: str = ""


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    printable = " ".join(quote_arg(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def capture_json(cmd: list[str]) -> dict[str, Any]:
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def quote_arg(value: str) -> str:
    if re.search(r"[^A-Za-z0-9_./:=+-]", value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required binary: {name}. Install ffmpeg/ffprobe first.")


def target_slug(target_minutes: float) -> str:
    rounded = round(target_minutes, 2)
    if rounded.is_integer():
        return f"{int(rounded)}min"
    return f"{str(rounded).replace('.', 'p')}min"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using {default}.")
        return default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"Invalid {name}={raw!r}; using {default}.")
        return default


def camera_overlap_seconds() -> float:
    return env_float("CAMERA_OVERLAP_SECONDS", DEFAULT_CAMERA_OVERLAP_SECONDS)


def video_enhance_filter() -> str:
    return env_str("VIDEO_ENHANCE_FILTER", DEFAULT_VIDEO_ENHANCE_FILTER)


def audio_enhance_filter() -> str:
    return env_str("AUDIO_ENHANCE_FILTER", DEFAULT_AUDIO_ENHANCE_FILTER)


def creator_context() -> str:
    explicit = os.getenv("CREATOR_CONTEXT", "").strip()
    if explicit:
        return explicit

    fields = [
        ("Creator name", os.getenv("CREATOR_NAME", "").strip()),
        ("Creator age", os.getenv("CREATOR_AGE", "").strip()),
        ("Creator interests", os.getenv("CREATOR_INTERESTS", "").strip()),
        ("Previous motorcycle", os.getenv("PREVIOUS_MOTORCYCLE", "").strip()),
        ("Previous motorcycle context", os.getenv("PREVIOUS_MOTORCYCLE_CONTEXT", "").strip()),
        ("Current motorcycle", os.getenv("CURRENT_MOTORCYCLE", "").strip()),
        ("Current motorcycle context", os.getenv("CURRENT_MOTORCYCLE_CONTEXT", "").strip()),
        ("Content guidance", os.getenv("CONTENT_CONTEXT_GUIDANCE", "").strip()),
    ]
    lines = [f"- {label}: {value}" for label, value in fields if value]
    if not lines:
        return "No creator-specific context was provided."
    lines.append("- Use this context only when relevant. Do not force it into every title, description, tag, or thumbnail hook.")
    return "\n".join(lines)


def discover_videos(input_dir: Path) -> list[Path]:
    videos = [
        path
        for path in sorted(input_dir.iterdir(), key=lambda p: (p.stat().st_mtime, p.name))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise SystemExit(f"No video files found in {input_dir}")
    return videos


def ffprobe_duration(path: Path) -> float:
    data = capture_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    return float(data["format"]["duration"])


def build_clean_source(videos: list[Path], work_dir: Path, dry_run: bool = False) -> Path:
    trimmed_dir = work_dir / "trimmed"
    trimmed_dir.mkdir(parents=True, exist_ok=True)
    overlap = camera_overlap_seconds()

    trimmed_files: list[Path] = []
    for index, video in enumerate(videos):
        out = trimmed_dir / f"{index:04d}_{video.stem}.mp4"
        trimmed_files.append(out)
        if out.exists() and out.stat().st_size > 0:
            continue

        cmd = ["ffmpeg", "-y"]
        if index > 0:
            cmd.extend(["-ss", str(overlap)])
        cmd.extend(["-i", str(video), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", str(out)])
        run(cmd, dry_run=dry_run)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for path in trimmed_files),
        encoding="utf-8",
    )

    clean_source = work_dir / "source_clean.mp4"
    if clean_source.exists() and clean_source.stat().st_size > 0:
        return clean_source

    try:
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(clean_source),
            ],
            dry_run=dry_run,
        )
    except subprocess.CalledProcessError:
        print("Concat without re-encode failed; retrying with normalization.")
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(clean_source),
            ],
            dry_run=dry_run,
        )

    return clean_source


def extract_audio(source: Path, work_dir: Path, dry_run: bool = False) -> Path:
    audio_path = work_dir / "speech_denoised.wav"
    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            audio_enhance_filter(),
            str(audio_path),
        ],
        dry_run=dry_run,
    )
    return audio_path


def transcribe_audio(audio_path: Path, model_size: str, language: str) -> list[TranscriptSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("faster-whisper is not installed; continuing without local transcription.")
        return []

    print(f"Transcribing locally with faster-whisper ({model_size})...")
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 600},
    )
    return [
        TranscriptSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
        for seg in segments
        if seg.text and seg.text.strip()
    ]


def analyze_visuals(source: Path, sample_every: float) -> list[dict[str, float]]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("opencv-python/numpy not installed; continuing with transcript/audio-only analysis.")
        return []

    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if frame_count else ffprobe_duration(source)

    samples: list[dict[str, float]] = []
    previous_hist = None
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break

        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)
        brightness = float(np.mean(hsv[:, :, 2]) / 255.0)

        change = 0.0
        if previous_hist is not None:
            change = 1.0 - float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))
        previous_hist = hist

        samples.append(
            {
                "time": t,
                "change": max(0.0, min(change, 2.0)),
                "sharpness": sharpness,
                "saturation": saturation,
                "brightness": brightness,
            }
        )
        t += sample_every

    cap.release()
    return samples


def transcript_in_window(transcript: list[TranscriptSegment], start: float, end: float) -> str:
    parts = [
        seg.text
        for seg in transcript
        if seg.end >= start and seg.start <= end
    ]
    return " ".join(parts).strip()


def speech_seconds_in_window(transcript: list[TranscriptSegment], start: float, end: float) -> float:
    total = 0.0
    for seg in transcript:
        overlap = max(0.0, min(end, seg.end) - max(start, seg.start))
        total += overlap
    return total


def window_visual_score(samples: list[dict[str, float]], start: float, end: float) -> float:
    values = [s for s in samples if start <= s["time"] < end]
    if not values:
        return 0.0

    change = sum(s["change"] for s in values) / len(values)
    saturation = sum(s["saturation"] for s in values) / len(values)
    brightness = sum(s["brightness"] for s in values) / len(values)
    sharpness = sum(min(s["sharpness"] / 500.0, 1.0) for s in values) / len(values)

    exposure_bonus = 1.0 - min(abs(brightness - 0.52) * 1.6, 0.8)
    return max(0.0, min(1.0, 0.48 * min(change * 4.0, 1.0) + 0.22 * saturation + 0.18 * sharpness + 0.12 * exposure_bonus))


def build_candidates(
    duration: float,
    transcript: list[TranscriptSegment],
    visual_samples: list[dict[str, float]],
    window_seconds: float,
    stride_seconds: float,
    limit: int,
) -> list[CandidateSegment]:
    candidates: list[CandidateSegment] = []
    segment_id = 1
    start = 0.0
    while start + 8.0 < duration:
        end = min(duration, start + window_seconds)
        text = transcript_in_window(transcript, start, end)
        speech_seconds = speech_seconds_in_window(transcript, start, end)
        speech_density = min(1.0, speech_seconds / max(1.0, end - start))
        word_count = len(text.split())
        speech_score = min(1.0, 0.55 * speech_density + 0.45 * min(word_count / 55.0, 1.0))
        visual_score = window_visual_score(visual_samples, start, end)
        audio_score = speech_score
        score = 0.42 * visual_score + 0.46 * speech_score + 0.12 * audio_score

        if score > 0.08:
            candidates.append(
                CandidateSegment(
                    id=segment_id,
                    start=round(start, 2),
                    end=round(end, 2),
                    score=round(score, 4),
                    visual_score=round(visual_score, 4),
                    speech_score=round(speech_score, 4),
                    audio_score=round(audio_score, 4),
                    reason="heuristic: visual change + narration density",
                    transcript=text[:900],
                )
            )
            segment_id += 1
        start += stride_seconds

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:limit]


def heuristic_plan(candidates: list[CandidateSegment], target_seconds: float) -> EditPlan:
    selected: list[CandidateSegment] = []
    used_until = -1.0
    total = 0.0

    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        if total >= target_seconds:
            break
        too_close = any(abs(candidate.start - existing.start) < 25 for existing in selected)
        if too_close:
            continue
        selected.append(candidate)
        total += candidate.duration
        used_until = max(used_until, candidate.end)

    selected.sort(key=lambda c: c.start)
    return EditPlan(
        title="Passeio de moto",
        description="Passeio de moto editado automaticamente com os melhores momentos da gravação.",
        thumbnail_text="PASSEIO DE MOTO",
        tags=["passeio de moto", "motovlog", "moto", "primeiras impressões", "rolê de moto"],
        segments=selected,
        notes="Heuristic plan generated without OpenAI.",
        requested_title="",
        requested_description="",
    )


def plan_with_openai(
    candidates: list[CandidateSegment],
    transcript: list[TranscriptSegment],
    target_seconds: float,
    model: str,
    title_hint: str,
    requested_title: str | None,
    requested_description: str | None,
) -> EditPlan | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("openai package is not installed; using heuristic edit plan.")
        return None

    client = OpenAI(api_key=api_key)
    full_transcript_context = transcript_context_for_ai(transcript)
    payload = [
        {
            "id": c.id,
            "start": c.start,
            "end": c.end,
            "duration": round(c.duration, 2),
            "score": c.score,
            "visual_score": c.visual_score,
            "speech_score": c.speech_score,
            "transcript": c.transcript,
        }
        for c in sorted(candidates, key=lambda c: c.start)
    ]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "thumbnail_text": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "notes": {"type": "string"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "integer"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "start", "end", "reason"],
                },
            },
        },
        "required": ["title", "description", "thumbnail_text", "tags", "notes", "segments"],
    }

    title_instruction = (
        f'Review and improve this user-provided title while preserving its intent: "{requested_title}"'
        if requested_title
        else "Generate a strong YouTube title from the content."
    )
    description_instruction = (
        f'Review and improve this user-provided description while preserving factual intent: "{requested_description}"'
        if requested_description
        else "Generate a concise YouTube description with useful context, natural keywords, and no invented facts."
    )

    prompt = f"""
You are an expert YouTube editor for Portuguese motorcycle ride videos.

Creator context:
{creator_context()}

Goal:
- Build an engaging {round(target_seconds / 60, 1)} minute video.
- Prefer moments with interesting narration, reactions, road/scenery changes, and story continuity.
- Avoid repetitive riding shots unless the scenery clearly changes.
- Keep segments mostly chronological.
- Use 12 to 45 second segments.
- Total duration should be close to {round(target_seconds)} seconds.
- Return only segments from the candidate list.
- Title and thumbnail text must be in Brazilian Portuguese and click-worthy without being misleading.
- Description must be in Brazilian Portuguese, ready to paste into YouTube, 1 to 3 short paragraphs.
- Generate 10 to 15 relevant YouTube tags in Brazilian Portuguese. Include broad search terms and specific content hooks, but no invented facts.
- Do not invent facts, locations, bike models, problems, or outcomes that are not supported by transcript/candidates/user input.
- Use the full transcript context to validate factual claims, especially bike specs, model details, locations, problems, and comparisons.
- If a spec is mentioned only as a comparison with a previous motorcycle, do not use it as the current motorcycle's spec in title, description, thumbnail text, or tags.
- Use the creator context to disambiguate BMW G310 GS versus Yamaha FZ25 facts, but do not add creator bio details unless they are relevant to the video.

Title task:
{title_instruction}

Description task:
{description_instruction}

Video title hint from user/path: {title_hint}

Full transcript context, for fact checking:
{full_transcript_context}

Candidate segments:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    print(f"Asking OpenAI ({model}) to choose the edit plan...")
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={"effort": "medium"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "motorcycle_edit_plan",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        raw = getattr(response, "output_text", "") or ""
    except Exception as exc:
        print(f"OpenAI structured request failed: {exc}")
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("OpenAI did not return valid JSON; using heuristic edit plan.")
        return None

    by_id = {c.id: c for c in candidates}
    selected: list[CandidateSegment] = []
    for item in data.get("segments", []):
        original = by_id.get(int(item["id"]))
        if not original:
            continue
        start = max(original.start, float(item["start"]))
        end = min(original.end, float(item["end"]))
        if end - start < 5:
            continue
        selected.append(
            CandidateSegment(
                id=original.id,
                start=round(start, 2),
                end=round(end, 2),
                score=original.score,
                visual_score=original.visual_score,
                speech_score=original.speech_score,
                audio_score=original.audio_score,
                reason=str(item.get("reason", "OpenAI selection")),
                transcript=original.transcript,
            )
        )

    if not selected:
        return None

    selected.sort(key=lambda c: c.start)
    return EditPlan(
        title=str(data.get("title") or "Passeio de moto"),
        description=str(data.get("description") or ""),
        thumbnail_text=str(data.get("thumbnail_text") or "PASSEIO DE MOTO"),
        tags=normalize_tags(data.get("tags")),
        segments=selected,
        notes=str(data.get("notes") or "OpenAI edit plan."),
        requested_title=requested_title or "",
        requested_description=requested_description or "",
    )


def normalize_tags(raw_tags: Any) -> list[str]:
    if not isinstance(raw_tags, list):
        raw_tags = []

    tags: list[str] = []
    seen: set[str] = set()
    for item in raw_tags:
        tag = re.sub(r"\s+", " ", str(item).strip().lstrip("#")).strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag[:60])
        if len(tags) >= 15:
            break

    defaults = [
        "motovlog",
        "passeio de moto",
        "rolê de moto",
        "moto",
        "primeiras impressões",
        "moto no Brasil",
        "viagem de moto",
        "pilotando moto",
        "review de moto",
        "YouTube moto",
    ]
    for tag in defaults:
        if len(tags) >= 10:
            break
        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            tags.append(tag)
    return tags


def selected_transcript_context(plan: EditPlan, limit: int = 1800) -> str:
    parts: list[str] = []
    for segment in plan.segments:
        text = re.sub(r"\s+", " ", segment.transcript).strip()
        if text:
            parts.append(f"{segment.start:.0f}s-{segment.end:.0f}s: {text}")
    context = "\n".join(parts)
    return context[:limit]


def transcript_context_for_ai(transcript: list[TranscriptSegment], limit: int = 18000) -> str:
    if not transcript:
        return "(sem transcrição disponível)"

    lines: list[str] = []
    for segment in transcript:
        text = re.sub(r"\s+", " ", segment.text).strip()
        if text:
            lines.append(f"{segment.start:.0f}s-{segment.end:.0f}s: {text}")

    context = "\n".join(lines)
    if len(context) <= limit:
        return context

    head = context[: int(limit * 0.55)].rsplit("\n", 1)[0]
    tail = context[-int(limit * 0.35):].split("\n", 1)[-1]
    return f"{head}\n...\n[transcrição abreviada para caber no prompt]\n...\n{tail}"


def write_youtube_metadata(plan: EditPlan, output_dir: Path) -> None:
    tags = normalize_tags(plan.tags)
    tags_line = ", ".join(tags)
    hashtags = " ".join(f"#{re.sub(r'[^0-9A-Za-zÀ-ÿ]+', '', tag.title())}" for tag in tags[:5])
    description = plan.description.strip()
    if hashtags and hashtags not in description:
        description = f"{description}\n\n{hashtags}".strip()

    content = f"""# YouTube

## Título
{plan.title.strip()}

## Descrição
{description}

## Tags
{tags_line}

## Texto sugerido para thumbnail
{plan.thumbnail_text.strip()}
"""
    (output_dir / "youtube.md").write_text(content, encoding="utf-8")


def write_plan_outputs(plan: EditPlan, output_dir: Path, plan_json_path: Path | None = None, plan_csv_path: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_json_path = plan_json_path or (output_dir / "edit_plan.json")
    plan_csv_path = plan_csv_path or (output_dir / "edit_plan.csv")
    plan.tags = normalize_tags(plan.tags)

    plan_json_path.write_text(
        json.dumps(plan_to_json(plan), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if plan_json_path.name != "edit_plan.json":
        (output_dir / "edit_plan.json").write_text(plan_json_path.read_text(encoding="utf-8"), encoding="utf-8")

    with plan_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "start", "end", "duration", "score", "reason", "transcript"])
        for segment in plan.segments:
            writer.writerow(
                [
                    segment.id,
                    segment.start,
                    segment.end,
                    round(segment.duration, 2),
                    segment.score,
                    segment.reason,
                    segment.transcript,
                ]
            )
    if plan_csv_path.name != "edit_plan.csv":
        shutil.copy2(plan_csv_path, output_dir / "edit_plan.csv")
    write_youtube_metadata(plan, output_dir)


def render_final(
    source: Path,
    plan: EditPlan,
    output_video: Path,
    work_dir: Path,
    render_preset: str,
    render_crf: int,
    audio_bitrate: str,
    force_render: bool,
    dry_run: bool = False,
) -> None:
    if output_video.exists() and output_video.stat().st_size > 0 and not force_render:
        print(f"Final video already exists; keeping it: {output_video}")
        return

    segments_dir = work_dir / "selected_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    for index, segment in enumerate(plan.segments):
        out = segments_dir / f"segment_{index:03d}.mp4"
        rendered.append(out)
        if out.exists() and out.stat().st_size > 0 and not force_render:
            continue
        duration = max(0.1, segment.duration)
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{segment.start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(source),
                "-vf",
                video_enhance_filter(),
                "-af",
                audio_enhance_filter(),
                "-c:v",
                "libx264",
                "-preset",
                render_preset,
                "-crf",
                str(render_crf),
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-movflags",
                "+faststart",
                str(out),
            ],
            dry_run=dry_run,
        )

    concat_file = work_dir / "selected_concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for path in rendered),
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_video),
        ],
        dry_run=dry_run,
    )


def find_existing_thumbnail_asset(input_dir: Path) -> Path | None:
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    if not images:
        return None
    preferred = [p for p in images if re.search(r"thumb|thumbnail|capa", p.name, re.I)]
    return sorted(preferred or images, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def extract_thumbnail_frame(source: Path, plan: EditPlan, output_dir: Path, dry_run: bool = False) -> Path:
    if plan.segments:
        best = max(plan.segments, key=lambda c: c.visual_score + c.score)
        timestamp = best.start + min(4.0, best.duration / 2)
    else:
        timestamp = 10.0
    frame = output_dir / "thumbnail_frame.jpg"
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-update",
            "1",
            str(frame),
        ],
        dry_run=dry_run,
    )
    return frame


def prepare_thumbnail_background(base_image: Path) -> Any:
    from PIL import Image, ImageEnhance

    img = Image.open(base_image).convert("RGB")
    target_w, target_h = 1280, 720
    scale = max(target_w / img.width, target_h / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)))
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    img = resized.crop((left, top, left + target_w, top + target_h))
    img = ImageEnhance.Contrast(img).enhance(1.18)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img


def draw_thumbnail_text(img: Any, text: str) -> Any:
    from PIL import Image, ImageDraw

    target_w, target_h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, int(target_h * 0.58), target_w, target_h), fill=(0, 0, 0, 145))

    clean_text = re.sub(r"\s+", " ", text.upper()).strip()[:42]
    font = load_font(target_w)
    box = draw.textbbox((0, 0), clean_text, font=font, stroke_width=3)
    text_w = box[2] - box[0]
    x = max(36, (target_w - text_w) // 2)
    y = int(target_h * 0.66)
    draw.text((x, y), clean_text, font=font, fill=(255, 244, 64), stroke_width=5, stroke_fill=(0, 0, 0))
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def local_thumbnail(base_image: Path, text: str, out: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; thumbnail frame will be used as-is.")
        shutil.copy2(base_image, out)
        return

    final = draw_thumbnail_text(prepare_thumbnail_background(base_image), text).convert("RGB")
    final.save(out, quality=94)


def load_font(target_w: int) -> Any:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=max(56, target_w // 14))
    return ImageFont.load_default()


def generate_openai_thumbnail(base_image: Path, plan: EditPlan, out: Path, image_model: str) -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    try:
        from openai import OpenAI
    except ImportError:
        return False

    transcript_context = selected_transcript_context(plan)
    prompt = f"""
Create ONLY a transparent PNG overlay layer for a high-click-through Brazilian YouTube motorcycle thumbnail.

Important constraints:
- Transparent background. No road, no motorcycle dashboard, no rider, no scenery, no photo background.
- Do not modify or recreate the underlying video frame.
- Create a focused upper thumbnail design layer: short punchy Portuguese text plus only the minimum supporting shapes needed for readability and emphasis.
- Do not add symbols, icons, arrows, badges, emojis, warning signs, or speed lines unless they are clearly justified by the video context.
- Prefer clean text treatment, subtle contrast panels, outlines, shadow, and one contextual accent over decorative clutter.
- Avoid realistic objects that could be mistaken for events that did not happen.
- The overlay should attract viewers while staying truthful.
- The text does NOT need to match the video title. Prefer a short curiosity hook with 2 to 5 words.
- Text must be large, bold, readable at phone size, and in Brazilian Portuguese.
- Leave most of the overlay transparent so the real motorcycle frame remains visible.
- Energetic Brazilian motorcycle YouTube style, high contrast, clean edges, modern creator thumbnail style, restrained graphics.
- Treat the transcript context below as the factual source of truth.
- If there is a comparison between motorcycles, do not attribute specs from the previous motorcycle to the current motorcycle.
- Use the creator context to disambiguate current motorcycle versus previous motorcycle specs, but do not add biographical details unless they make the thumbnail clearer.

Creator context:
{creator_context()}

Video title context: "{plan.title}".
Video description context: "{plan.description[:600]}".
Suggested thumbnail hook, optional to improve: "{plan.thumbnail_text[:42]}".
Selected transcript context:
{transcript_context}
""".strip()

    client = OpenAI(api_key=api_key)
    print(f"Generating transparent thumbnail overlay with OpenAI image model ({image_model})...")
    try:
        result = client.images.generate(
            model=image_model,
            prompt=prompt,
            size="1536x1024",
            background="transparent",
            output_format="png",
        )
        b64 = result.data[0].b64_json
        overlay_path = out.with_name(out.stem + "_overlay.png")
        overlay_path.write_bytes(base64.b64decode(b64))
        compose_overlay_thumbnail(base_image, overlay_path, None, out)
        return True
    except Exception as exc:
        print(f"OpenAI overlay thumbnail generation failed: {exc}")
        return False


def compose_overlay_thumbnail(base_image: Path, overlay_image: Path, text: str | None, out: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; cannot compose overlay thumbnail.")
        shutil.copy2(base_image, out)
        return

    background = prepare_thumbnail_background(base_image).convert("RGBA")
    overlay = Image.open(overlay_image).convert("RGBA")
    target_w, target_h = background.size
    scale = max(target_w / overlay.width, target_h / overlay.height)
    resized = overlay.resize((int(overlay.width * scale), int(overlay.height * scale)))
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    overlay = resized.crop((left, top, left + target_w, top + target_h))

    combined = Image.alpha_composite(background, overlay)
    final = draw_thumbnail_text(combined, text).convert("RGB") if text else combined.convert("RGB")
    final.save(out, quality=94)


def save_transcript(transcript: list[TranscriptSegment], output_dir: Path) -> None:
    (output_dir / "transcript.json").write_text(
        json.dumps([asdict(segment) for segment in transcript], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_transcript(path: Path) -> list[TranscriptSegment] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TranscriptSegment(start=float(item["start"]), end=float(item["end"]), text=str(item["text"])) for item in data]


def load_candidates(path: Path) -> list[CandidateSegment] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[CandidateSegment] = []
    for item in data:
        candidates.append(
            CandidateSegment(
                id=int(item["id"]),
                start=float(item["start"]),
                end=float(item["end"]),
                score=float(item["score"]),
                visual_score=float(item["visual_score"]),
                speech_score=float(item["speech_score"]),
                audio_score=float(item["audio_score"]),
                reason=str(item.get("reason", "")),
                transcript=str(item.get("transcript", "")),
            )
        )
    return candidates


def plan_to_json(plan: EditPlan) -> dict[str, Any]:
    return {
        "title": plan.title,
        "description": plan.description,
        "thumbnail_text": plan.thumbnail_text,
        "tags": normalize_tags(plan.tags),
        "notes": plan.notes,
        "requested_title": plan.requested_title,
        "requested_description": plan.requested_description,
        "segments": [asdict(segment) for segment in plan.segments],
        "total_seconds": round(sum(segment.duration for segment in plan.segments), 2),
    }


def load_plan(path: Path) -> EditPlan | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        CandidateSegment(
            id=int(item["id"]),
            start=float(item["start"]),
            end=float(item["end"]),
            score=float(item["score"]),
            visual_score=float(item["visual_score"]),
            speech_score=float(item["speech_score"]),
            audio_score=float(item["audio_score"]),
            reason=str(item.get("reason", "")),
            transcript=str(item.get("transcript", "")),
        )
        for item in data.get("segments", [])
    ]
    return EditPlan(
        title=str(data.get("title") or "Passeio de moto"),
        description=str(data.get("description") or ""),
        thumbnail_text=str(data.get("thumbnail_text") or "PASSEIO DE MOTO"),
        tags=normalize_tags(data.get("tags")),
        segments=segments,
        notes=str(data.get("notes") or ""),
        requested_title=str(data.get("requested_title") or ""),
        requested_description=str(data.get("requested_description") or ""),
    )


def load_matching_plan(path: Path, target_seconds: float) -> EditPlan | None:
    if not path.exists():
        return None
    plan = load_plan(path)
    if plan is None:
        return None
    total = sum(segment.duration for segment in plan.segments)
    lower = target_seconds * 0.8
    upper = target_seconds * 1.2
    if lower <= total <= upper:
        return plan
    return None


def plan_matches_requested_metadata(plan: EditPlan, requested_title: str | None, requested_description: str | None) -> bool:
    normalized_title = (requested_title or "").strip()
    normalized_description = (requested_description or "").strip()
    if normalized_title and plan.requested_title.strip() != normalized_title:
        return False
    if normalized_description and plan.requested_description.strip() != normalized_description:
        return False
    return True


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatically edit motorcycle ride videos.")
    parser.add_argument("--input", required=True, type=Path, help="Folder containing camera videos.")
    parser.add_argument("--target-minutes", required=True, type=float, help="Desired final video length.")
    parser.add_argument("--title", default=None, help="Optional YouTube title. OpenAI will review/improve it when enabled.")
    parser.add_argument("--description", default=None, help="Optional YouTube description text. OpenAI will review/improve it when enabled.")
    parser.add_argument("--description-file", type=Path, default=None, help="Optional file containing YouTube description text.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output folder.")
    parser.add_argument("--output-video", type=Path, default=None, help="Final MP4 path.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Working folder for temporary files.")
    parser.add_argument("--language", default="pt", help="Whisper language code.")
    parser.add_argument("--whisper-model", default="medium", help="faster-whisper model size.")
    parser.add_argument("--ai-model", default=env_str("AI_MODEL", DEFAULT_AI_MODEL), help="OpenAI model for edit decisions.")
    parser.add_argument("--image-model", default=env_str("IMAGE_MODEL", DEFAULT_IMAGE_MODEL), help="OpenAI image model for thumbnail.")
    parser.add_argument("--candidate-limit", type=int, default=120, help="Max candidate segments sent to OpenAI.")
    parser.add_argument("--window-seconds", type=float, default=24.0, help="Candidate window size.")
    parser.add_argument("--stride-seconds", type=float, default=8.0, help="Candidate window stride.")
    parser.add_argument("--sample-every", type=float, default=2.0, help="Visual sample interval in seconds.")
    parser.add_argument("--render-preset", default=env_str("RENDER_PRESET", DEFAULT_RENDER_PRESET), help="libx264 preset: faster, fast, medium, slow.")
    parser.add_argument("--render-crf", type=int, default=env_int("RENDER_CRF", DEFAULT_RENDER_CRF), help="libx264 CRF. Lower is bigger/better; 21-23 is a good YouTube range.")
    parser.add_argument("--audio-bitrate", default=env_str("AUDIO_BITRATE", DEFAULT_AUDIO_BITRATE), help="AAC audio bitrate for final segments.")
    parser.add_argument("--force-render", action="store_true", help="Re-render selected segments and final video even if files already exist.")
    parser.add_argument("--force-analysis", action="store_true", help="Recompute local transcript, visual samples, and candidates.")
    parser.add_argument("--force-ai", action="store_true", help="Ask OpenAI for a new edit plan even if a compatible plan exists.")
    parser.add_argument("--force-thumbnail", action="store_true", help="Regenerate thumbnail even if one already exists.")
    parser.add_argument("--local-thumbnail", action="store_true", help="Use local thumbnail generation even when OpenAI is enabled.")
    parser.add_argument("--no-openai", action="store_true", help="Disable OpenAI selection and thumbnail.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    load_env_file(Path.cwd() / ".env")
    args = parse_args(argv)
    require_binary("ffmpeg")
    require_binary("ffprobe")

    input_dir = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_description = args.description
    if args.description_file:
        requested_description = args.description_file.expanduser().read_text(encoding="utf-8").strip()
    slug = target_slug(args.target_minutes)
    output_video = (args.output_video or (output_dir / f"moto_editado_{slug}.mp4")).expanduser().resolve()
    work_dir = (args.work_dir or (output_dir / "_work")).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    target_work_dir = work_dir / slug
    target_work_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = output_dir / "transcript.json"
    visual_samples_path = output_dir / "visual_samples.json"
    candidates_path = output_dir / "candidates.json"
    plan_json_path = output_dir / f"edit_plan_{slug}.json"
    plan_csv_path = output_dir / f"edit_plan_{slug}.csv"
    legacy_plan_json_path = output_dir / "edit_plan.json"

    if args.no_openai:
        os.environ.pop("OPENAI_API_KEY", None)

    videos = discover_videos(input_dir)
    print(f"Found {len(videos)} clips. Fixed overlap: {camera_overlap_seconds()}s.")

    source = build_clean_source(videos, work_dir, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run finished before analysis.")
        return 0

    duration = ffprobe_duration(source)
    audio = extract_audio(source, work_dir)

    transcript = None if args.force_analysis else load_transcript(transcript_path)
    if transcript is None:
        transcript = transcribe_audio(audio, args.whisper_model, args.language)
        save_transcript(transcript, output_dir)
    else:
        print(f"Reusing cached transcript: {transcript_path}")

    visual_samples = None
    if visual_samples_path.exists() and not args.force_analysis:
        visual_samples = json.loads(visual_samples_path.read_text(encoding="utf-8"))
        print(f"Reusing cached visual samples: {visual_samples_path}")
    if visual_samples is None:
        visual_samples = analyze_visuals(source, args.sample_every)
        visual_samples_path.write_text(json.dumps(visual_samples, indent=2), encoding="utf-8")

    target_seconds = max(60.0, args.target_minutes * 60.0)
    candidates = None if args.force_analysis else load_candidates(candidates_path)
    if candidates is None:
        candidates = build_candidates(
            duration=duration,
            transcript=transcript,
            visual_samples=visual_samples,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
            limit=args.candidate_limit,
        )
        candidates_path.write_text(
            json.dumps([asdict(candidate) for candidate in candidates], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        print(f"Reusing cached candidates: {candidates_path}")

    plan = None if args.force_ai else load_plan(plan_json_path)
    if plan is not None and not plan_matches_requested_metadata(plan, args.title, requested_description):
        print("Cached edit plan does not match provided title/description; asking for a new plan.")
        plan = None
    if plan is not None:
        print(f"Reusing cached edit plan: {plan_json_path}")
    elif not args.force_ai:
        plan = load_matching_plan(legacy_plan_json_path, target_seconds)
        if plan is not None and not plan_matches_requested_metadata(plan, args.title, requested_description):
            plan = None
        if plan is not None:
            print(f"Reusing matching legacy edit plan: {legacy_plan_json_path}")
    if not args.no_openai:
        if plan is None:
            plan = plan_with_openai(
                candidates,
                transcript,
                target_seconds,
                args.ai_model,
                input_dir.name,
                args.title,
                requested_description,
            )
    if plan is None:
        plan = heuristic_plan(candidates, target_seconds)
        if args.title:
            plan.title = args.title
            plan.requested_title = args.title
        if requested_description:
            plan.description = requested_description
            plan.requested_description = requested_description

    write_plan_outputs(plan, output_dir, plan_json_path=plan_json_path, plan_csv_path=plan_csv_path)
    render_final(
        source=source,
        plan=plan,
        output_video=output_video,
        work_dir=target_work_dir,
        render_preset=args.render_preset,
        render_crf=args.render_crf,
        audio_bitrate=args.audio_bitrate,
        force_render=args.force_render,
        dry_run=args.dry_run,
    )

    existing_asset = find_existing_thumbnail_asset(input_dir)
    thumb_ai = output_dir / f"thumbnail_openai_{slug}.png"
    thumb_local = output_dir / f"thumbnail_local_{slug}.jpg"
    if thumb_ai.exists() and not args.force_thumbnail and not args.no_openai and not args.local_thumbnail:
        generated = True
        print(f"Reusing cached OpenAI thumbnail: {thumb_ai}")
    elif thumb_local.exists() and not args.force_thumbnail and (args.no_openai or args.local_thumbnail):
        generated = False
        print(f"Reusing cached local thumbnail: {thumb_local}")
    else:
        generated = False
        base_thumb = existing_asset or extract_thumbnail_frame(source, plan, output_dir)
        if not args.no_openai and not args.local_thumbnail:
            generated = generate_openai_thumbnail(base_thumb, plan, thumb_ai, args.image_model)
        if not generated:
            local_thumbnail(base_thumb, plan.thumbnail_text, thumb_local)
    if thumb_ai.exists():
        shutil.copy2(thumb_ai, output_dir / "thumbnail_openai.png")
    if thumb_local.exists():
        shutil.copy2(thumb_local, output_dir / "thumbnail_local.jpg")

    print("\nDone.")
    print(f"Video: {output_video}")
    print(f"Plan: {output_dir / 'edit_plan.json'}")
    print(f"Thumbnail: {thumb_ai if generated else thumb_local}")
    print(f"Title suggestion: {plan.title}")
    print(f"Description: {plan.description[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
