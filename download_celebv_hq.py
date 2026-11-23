#!/usr/bin/env python3
"""Download/process CelebV-HQ and build the manifest used by the trainer.

The script intentionally keeps every available CelebV-HQ clip and every motion
category. Only clips that cannot provide a reliable 49-frame training window
for technical reasons are marked ``regular_eligible=false`` and routed to
image-conditioned teacher rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


METADATA_URL = (
    "https://raw.githubusercontent.com/CelebV-HQ/CelebV-HQ/"
    "main/celebvhq_info.json"
)
MANIFEST_NAME = "celebv_hq_manifest.csv"
MANIFEST_COLUMNS = [
    "video_file",
    "clip_id",
    "ytb_id",
    "caption",
    "actions",
    "emotion",
    "regular_eligible",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_sec",
    "start_sec",
    "end_sec",
    "bbox_top",
    "bbox_bottom",
    "bbox_left",
    "bbox_right",
    "version",
]

# Appearance defects do not provide a reliable pixel-space target. The clips
# are still retained and used for teacher rollout.
ROLLOUT_ONLY_APPEARANCE = {"blurry", "wearing_mask"}

ACTION_PHRASES = {
    "blow": "blows air",
    "chew": "chews",
    "close_eyes": "naturally closes the eyes",
    "cough": "coughs",
    "cry": "cries",
    "drink": "takes a drink",
    "eat": "eats",
    "frown": "shows a subtle frown",
    "gaze": "shifts their gaze",
    "glare": "holds an intent gaze",
    "head_wagging": "gently moves their head",
    "kiss": "makes a kissing motion",
    "laugh": "laughs",
    "listen_to_music": "moves subtly while listening",
    "look_around": "looks around",
    "make_a_face": "makes an expressive face",
    "nod": "gently nods",
    "play_instrument": "moves while playing an instrument",
    "read": "moves their eyes while reading",
    "shake_head": "gently shakes their head",
    "shout": "shouts",
    "sigh": "takes a subtle breath",
    "sing": "sings",
    "sleep": "rests with closed eyes",
    "smile": "shows a natural smile",
    "smoke": "smokes",
    "sneer": "changes their expression slightly",
    "sneeze": "sneezes",
    "sniff": "makes a small facial movement",
    "talk": "talks",
    "turn": "slightly turns their head",
    "weep": "weeps",
    "whisper": "whispers",
    "wink": "briefly winks",
    "yawn": "yawns",
}


@dataclass(frozen=True)
class VideoProbe:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the complete CelebV-HQ dataset for portrait student training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", default="celebv_hq")
    parser.add_argument("--metadata", default=None)
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only index already processed <out-dir>/videos/*.mp4 files.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep downloaded source videos after all their clips are processed.",
    )
    parser.add_argument("--proxy", default=None)
    parser.add_argument(
        "--cookies",
        default=None,
        help="Netscape cookies.txt passed to yt-dlp when YouTube asks to sign in.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Browser profile for yt-dlp, for example firefox, chrome, or edge.",
    )
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--training-frames", type=int, default=49)
    parser.add_argument("--training-fps", type=float, default=25.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm acceptance of the official CelebV-HQ non-commercial agreement.",
    )
    return parser.parse_args()


def run_checked(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def ensure_program(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required program is not available on PATH: {name}")


def yt_dlp_command() -> List[str]:
    # Prefer the Poetry/venv module so PATH need not contain a yt-dlp binary.
    try:
        import yt_dlp  # noqa: F401
    except ImportError as error:
        path_binary = shutil.which("yt-dlp")
        if path_binary is not None:
            return [path_binary]
        raise RuntimeError(
            "yt-dlp is not installed in this environment. "
            "Run: poetry install"
        ) from error
    return [sys.executable, "-m", "yt_dlp"]


def ensure_metadata(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"Downloading CelebV-HQ metadata -> {path}")
    urllib.request.urlretrieve(METADATA_URL, temporary)
    temporary.replace(path)
    return path


def load_metadata(path: Path) -> Tuple[List[str], List[str], Dict[str, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    meta_info = payload.get("meta_info", {})
    appearance_mapping = [str(value) for value in meta_info["appearance_mapping"]]
    action_mapping = [str(value) for value in meta_info["action_mapping"]]
    clips = payload.get("clips")
    if not isinstance(clips, dict) or not clips:
        raise RuntimeError(f"Invalid CelebV-HQ metadata: {path}")
    return appearance_mapping, action_mapping, clips


def active_labels(values: Sequence[object], mapping: Sequence[str]) -> List[str]:
    return [
        mapping[index]
        for index, value in enumerate(values[: len(mapping)])
        if float(value) > 0
    ]


def primary_emotion(attributes: dict) -> str:
    emotion = attributes.get("emotion", {})
    labels = emotion.get("labels", "neutral")
    if isinstance(labels, str):
        return labels
    if not isinstance(labels, list) or not labels:
        return "neutral"
    durations: Dict[str, float] = {}
    for segment in labels:
        if not isinstance(segment, dict):
            continue
        name = str(segment.get("emotion", "neutral"))
        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", start))
        durations[name] = durations.get(name, 0.0) + max(0.0, end - start)
    return max(durations, key=durations.get) if durations else "neutral"


def build_caption(actions: Sequence[str], emotion: str) -> str:
    phrases = [ACTION_PHRASES[action] for action in actions if action in ACTION_PHRASES]
    motion = ", ".join(phrases)
    if motion:
        motion = f"The person {motion}."
    else:
        motion = "The person has subtle natural facial and head motion."
    emotion_phrase = {
        "neutral": "a calm natural expression",
        "happy": "a gentle positive expression",
        "sadness": "a subdued expression",
        "anger": "a serious expression",
        "surprise": "a mildly surprised expression",
        "fear": "a tense expression",
        "contempt": "a restrained expression",
        "disgust": "a displeased expression",
    }.get(emotion, "a natural expression")
    return (
        "A close-up portrait video of one person with "
        f"{emotion_phrase}. {motion} Smooth realistic motion."
    )


def probe_video(path: Path) -> Optional[VideoProbe]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            return None
        stream = streams[0]
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        fps = 0.0
        for rate in (stream.get("avg_frame_rate"), stream.get("r_frame_rate")):
            if not isinstance(rate, str) or "/" not in rate:
                continue
            numerator, denominator = rate.split("/", 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                continue
            candidate = float(numerator) / denominator_value
            if math.isfinite(candidate) and candidate > 0:
                fps = candidate
                break
        duration = float(stream.get("duration") or 0.0)
        raw_frames = stream.get("nb_frames")
        frame_count = (
            int(raw_frames)
            if raw_frames is not None and str(raw_frames).isdigit()
            else 0
        )
        if frame_count <= 0 and duration > 0 and fps > 0:
            frame_count = int(round(duration * fps))
        values = (width, height, fps, duration, frame_count)
        if (
            width <= 0
            or height <= 0
            or fps <= 0
            or duration <= 0
            or frame_count <= 0
            or not all(math.isfinite(float(value)) for value in values)
        ):
            return None
        return VideoProbe(width, height, fps, frame_count, duration)
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def required_source_frames(
    source_fps: float,
    target_fps: float,
    output_frames: int,
) -> int:
    if target_fps * 0.92 <= source_fps < target_fps:
        return output_frames
    return math.ceil((output_frames - 1) * source_fps / target_fps) + 1


def is_regular_eligible(
    probe: VideoProbe,
    appearance: Sequence[str],
    training_frames: int,
    training_fps: float,
) -> bool:
    enough_frames = probe.frame_count >= required_source_frames(
        probe.fps,
        training_fps,
        training_frames,
    )
    compatible_fps = probe.fps >= training_fps * 0.92
    clean_appearance = not ROLLOUT_ONLY_APPEARANCE.intersection(appearance)
    return (
        enough_frames
        and compatible_fps
        and clean_appearance
    )


def square_crop_from_normalized_bbox(
    bbox: dict,
    width: int,
    height: int,
    expansion: float = 0.02,
) -> Tuple[int, int, int]:
    top = max(0.0, float(bbox["top"]) - expansion)
    bottom = min(1.0, float(bbox["bottom"]) + expansion)
    left = max(0.0, float(bbox["left"]) - expansion)
    right = min(1.0, float(bbox["right"]) + expansion)
    x0, x1 = left * width, right * width
    y0, y1 = top * height, bottom * height
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    side = min(float(min(width, height)), max(x1 - x0, y1 - y0))
    side = max(2.0, side)
    crop_x = int(round(min(max(0.0, center_x - side / 2.0), width - side)))
    crop_y = int(round(min(max(0.0, center_y - side / 2.0), height - side)))
    crop_side = int(round(min(side, width - crop_x, height - crop_y)))
    return crop_x, crop_y, crop_side


# Prefer a >=512p DASH merge, then any real video. Storyboards are not videos.
YT_DLP_FORMAT = "bv*[height>=512]+ba/b[height>=512]/bv*+ba/b"
# Cookies hide formats until a JS runtime solves the player challenge.
# web_safari still exposes HLS when the default clients return only storyboards.
YT_DLP_PLAYER_CLIENTS = (
    None,
    "youtube:player_client=web_safari,default",
)


def js_runtime_args() -> List[str]:
    deno = shutil.which("deno")
    if deno:
        return ["--js-runtimes", f"deno:{deno}"]
    node = shutil.which("node") or shutil.which("nodejs")
    if node:
        return ["--js-runtimes", f"node:{node}"]
    return []


def ensure_js_runtime() -> None:
    if js_runtime_args():
        return
    raise RuntimeError(
        "YouTube downloads need Node.js or Deno so yt-dlp can solve the "
        "player challenge. In WSL run: sudo apt-get update && "
        "sudo apt-get install -y nodejs"
    )


def download_source(
    ytb_id: str,
    raw_path: Path,
    proxy: Optional[str],
    cookies: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> Path:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(f"{raw_path.stem}.part{raw_path.suffix}")
    temporary.unlink(missing_ok=True)
    url = f"https://www.youtube.com/watch?v={ytb_id}"
    last_error: Optional[BaseException] = None
    for attempt, player_client in enumerate(YT_DLP_PLAYER_CLIENTS):
        temporary.unlink(missing_ok=True)
        command = [
            *yt_dlp_command(),
            *js_runtime_args(),
            "--remote-components",
            "ejs:github",
            "--no-playlist",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            "-f",
            YT_DLP_FORMAT,
            "--merge-output-format",
            "mp4",
            "-o",
            str(temporary),
        ]
        if player_client:
            command.extend(["--extractor-args", player_client])
        if proxy:
            command.extend(["--proxy", proxy])
        if cookies:
            command.extend(["--cookies", cookies])
        if cookies_from_browser:
            command.extend(["--cookies-from-browser", cookies_from_browser])
        command.append(url)
        try:
            run_checked(command)
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt + 1 < len(YT_DLP_PLAYER_CLIENTS):
                print(
                    f"  retry {ytb_id} with {YT_DLP_PLAYER_CLIENTS[attempt + 1]}",
                    file=sys.stderr,
                )
            continue
        if temporary.exists():
            temporary.replace(raw_path)
            return raw_path
        last_error = RuntimeError(f"yt-dlp did not create {temporary}")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"yt-dlp failed for {ytb_id}")


def process_clip(
    raw_path: Path,
    output_path: Path,
    clip: dict,
    source_probe: VideoProbe,
    target_size: int,
    crf: int,
    preset: str,
) -> None:
    start = float(clip["duration"]["start_sec"])
    end = float(clip["duration"]["end_sec"])
    if not 0 <= start < end:
        raise ValueError(f"Invalid clip interval: {start}..{end}")
    crop_x, crop_y, crop_side = square_crop_from_normalized_bbox(
        clip["bbox"],
        source_probe.width,
        source_probe.height,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.part{output_path.suffix}"
    )
    temporary.unlink(missing_ok=True)
    video_filter = (
        f"crop={crop_side}:{crop_side}:{crop_x}:{crop_y},"
        f"scale={target_size}:{target_size}:flags=lanczos,setsar=1"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",
        "-to",
        f"{end:.6f}",
        "-i",
        str(raw_path),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    try:
        run_checked(command)
        if probe_video(temporary) is None:
            raise RuntimeError(f"Processed clip is invalid: {temporary}")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def manifest_row(
    clip_id: str,
    clip: dict,
    probe: VideoProbe,
    appearance_mapping: Sequence[str],
    action_mapping: Sequence[str],
    training_frames: int,
    training_fps: float,
) -> Dict[str, object]:
    attributes = clip.get("attributes", {})
    appearance = active_labels(attributes.get("appearance", []), appearance_mapping)
    actions = active_labels(attributes.get("action", []), action_mapping)
    emotion = primary_emotion(attributes)
    duration = clip["duration"]
    bbox = clip["bbox"]
    return {
        "video_file": f"{clip_id}.mp4",
        "clip_id": clip_id,
        "ytb_id": str(clip["ytb_id"]),
        "caption": build_caption(actions, emotion),
        "actions": json.dumps(actions, ensure_ascii=False, separators=(",", ":")),
        "emotion": emotion,
        "regular_eligible": int(
            is_regular_eligible(
                probe,
                appearance,
                training_frames,
                training_fps,
            )
        ),
        "width": probe.width,
        "height": probe.height,
        "fps": f"{probe.fps:.6f}",
        "frame_count": probe.frame_count,
        "duration_sec": f"{probe.duration_sec:.6f}",
        "start_sec": duration["start_sec"],
        "end_sec": duration["end_sec"],
        "bbox_top": bbox["top"],
        "bbox_bottom": bbox["bottom"],
        "bbox_left": bbox["left"],
        "bbox_right": bbox["right"],
        "version": clip.get("version", ""),
    }


def load_existing_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise RuntimeError(
                f"Manifest schema differs from CelebV-HQ schema: {path}. "
                "Remove the manifest and run indexing again."
            )
        return {
            str(row["clip_id"]): dict(row)
            for row in reader
            if str(row.get("clip_id", "")).strip()
        }


def save_manifest(path: Path, rows: Dict[str, Dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for clip_id in sorted(rows):
            writer.writerow(rows[clip_id])
    temporary.replace(path)


def clips_by_source(clips: Dict[str, dict]) -> Dict[str, List[Tuple[str, dict]]]:
    grouped: Dict[str, List[Tuple[str, dict]]] = {}
    for clip_id, clip in clips.items():
        grouped.setdefault(str(clip["ytb_id"]), []).append((clip_id, clip))
    return grouped


def main() -> int:
    args = parse_args()
    if not args.accept_license:
        raise RuntimeError(
            "CelebV-HQ is restricted to non-commercial research. Read "
            "https://celebv-hq.github.io/ and rerun with --accept-license."
        )
    if args.training_frames <= 0 or args.training_fps <= 0:
        raise ValueError("training-frames and training-fps must be positive")

    ensure_program("ffprobe")
    if not args.index_only:
        ensure_program("ffmpeg")
        yt_dlp_command()  # fail early if yt-dlp is missing from the env
        ensure_js_runtime()

    root = Path(args.out_dir).resolve()
    videos_dir = root / "videos"
    raw_dir = root / "_raw"
    manifest_path = root / MANIFEST_NAME
    metadata_path = (
        Path(args.metadata).resolve()
        if args.metadata
        else root / "celebvhq_info.json"
    )
    root.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    ensure_metadata(metadata_path)
    appearance_mapping, action_mapping, clips = load_metadata(metadata_path)
    grouped = clips_by_source(clips)
    rows: Dict[str, Dict[str, object]] = {
        key: value
        for key, value in load_existing_manifest(manifest_path).items()
        if key in clips
    }

    failed_sources = 0
    for source_index, (ytb_id, source_clips) in enumerate(grouped.items(), start=1):
        existing_probes: Dict[str, VideoProbe] = {}
        for clip_id, clip in source_clips:
            current_path = videos_dir / f"{clip_id}.mp4"
            deprecated_name = clip.get("deprecated_name")
            if (
                not current_path.exists()
                and isinstance(deprecated_name, str)
                and deprecated_name
            ):
                deprecated_path = videos_dir / deprecated_name
                if deprecated_path.exists():
                    deprecated_path.replace(current_path)
            if current_path.exists():
                existing_probe = probe_video(current_path)
                if existing_probe is None:
                    current_path.unlink(missing_ok=True)
                    rows.pop(clip_id, None)
                else:
                    existing_probes[clip_id] = existing_probe
            else:
                rows.pop(clip_id, None)
        missing = [
            (clip_id, clip)
            for clip_id, clip in source_clips
            if not (videos_dir / f"{clip_id}.mp4").exists()
        ]
        raw_path = raw_dir / f"{ytb_id}.mp4"
        if missing and not args.index_only:
            try:
                source_probe = probe_video(raw_path) if raw_path.exists() else None
                if source_probe is None:
                    raw_path.unlink(missing_ok=True)
                    print(
                        f"[{source_index}/{len(grouped)}] download {ytb_id} "
                        f"for {len(missing)} clips"
                    )
                    download_source(
                        ytb_id,
                        raw_path,
                        args.proxy,
                        cookies=args.cookies,
                        cookies_from_browser=args.cookies_from_browser,
                    )
                    source_probe = probe_video(raw_path)
                if source_probe is None:
                    raw_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Downloaded source could not be probed: {raw_path}"
                    )
                for clip_id, clip in missing:
                    output_path = videos_dir / f"{clip_id}.mp4"
                    try:
                        process_clip(
                            raw_path,
                            output_path,
                            clip,
                            source_probe,
                            args.target_size,
                            args.crf,
                            args.preset,
                        )
                    except Exception as error:
                        print(
                            f"  failed clip {clip_id}: {error}",
                            file=sys.stderr,
                        )
            except Exception as error:
                failed_sources += 1
                print(f"  failed source {ytb_id}: {error}", file=sys.stderr)
            finally:
                if raw_path.exists() and not args.keep_raw:
                    raw_path.unlink(missing_ok=True)

        source_complete = True
        for clip_id, clip in source_clips:
            output_path = videos_dir / f"{clip_id}.mp4"
            if not output_path.exists():
                rows.pop(clip_id, None)
                source_complete = False
                continue
            probe = existing_probes.get(clip_id) or probe_video(output_path)
            if probe is None:
                output_path.unlink(missing_ok=True)
                rows.pop(clip_id, None)
                source_complete = False
                continue
            rows[clip_id] = manifest_row(
                clip_id,
                clip,
                probe,
                appearance_mapping,
                action_mapping,
                args.training_frames,
                args.training_fps,
            )
        save_manifest(manifest_path, rows)
        if source_complete:
            print(
                f"[{source_index}/{len(grouped)}] indexed {ytb_id}: "
                f"{len(source_clips)} clips"
            )

    regular_count = sum(
        str(row["regular_eligible"]).lower() in {"1", "true"}
        for row in rows.values()
    )
    print(
        f"manifest: {manifest_path}\n"
        f"available clips: {len(rows)}/{len(clips)}\n"
        f"regular-video eligible: {regular_count}\n"
        f"rollout-only: {len(rows) - regular_count}\n"
        f"failed sources this run: {failed_sources}"
    )
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
