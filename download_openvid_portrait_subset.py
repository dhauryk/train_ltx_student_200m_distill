#!/usr/bin/env python3
"""
Download a portrait-animation subset from OpenVid-1M with low disk usage.

How it works:
1) Read OpenVid caption CSV(s), keep only rows matching portrait keywords.
2) Read mapping shards (zip_file -> video_path) from OpenVid-1M-mapping.
3) Download one ZIP part at a time, extract only matched videos.
4) Probe extracted videos and keep only those passing min resolution/duration/FPS.
5) Delete processed ZIP and shard files to save disk space.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_POSITIVE_KEYWORDS = [
    "portrait",
    "close-up",
    "close up",
    "face",
    "person",
    "human",
    "man",
    "male",
    "boy",
    "woman",
    "female",
    "girl",
    "head",
    "hair",
    "blink",
    "blinking",
    "smile",
    "expression",
    "looking at camera",
    "head turn",
]

DEFAULT_NEGATIVE_KEYWORDS = [
    "landscape",
    "city",
    "street",
    "car",
    "driving",
    "times square",
    "rocket",
    "field",
    "animal",
    "dog",
    "cat",
    "crowd",
    "group of people",
    "full body",
    "wide shot",
]


@dataclass
class VideoProbe:
    width: int
    height: int
    fps: float
    duration_sec: float


@dataclass
class Stats:
    captions_seen: int = 0
    captions_matched: int = 0
    shards_total: int = 0
    shards_processed: int = 0
    rows_seen: int = 0
    rows_keyword_matched: int = 0
    videos_downloaded: int = 0
    videos_kept: int = 0
    videos_rejected_probe: int = 0
    videos_failed: int = 0
    videos_skipped_existing: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenVid-1M portrait subset by ZIP parts with immediate cleanup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", default="nkp37/OpenVid-1M", help="HF dataset id with ZIP parts.")
    parser.add_argument(
        "--mapping-repo-id",
        default="phil329/OpenVid-1M-mapping",
        help="HF dataset id with zip/video mapping CSV files.",
    )
    parser.add_argument("--hf-token", default=None, help="Optional HF token.")

    parser.add_argument(
        "--caption-files",
        nargs="+",
        default=["data/train/OpenVid-1M.csv"],
        help="Caption CSV file(s) inside --repo-id.",
    )
    parser.add_argument(
        "--mapping-pattern",
        default="video_mappings/OpenVid_part*.csv",
        help="Glob pattern for mapping shard files in --mapping-repo-id.",
    )

    parser.add_argument("--start-index", type=int, default=0, help="Mapping shard start index (inclusive).")
    parser.add_argument("--end-index", type=int, default=None, help="Mapping shard end index (exclusive).")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-videos-per-shard", type=int, default=None)

    parser.add_argument("--out-dir", default="openvid_portrait_subset")
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--manifest-name", default="subset_manifest.csv")
    parser.add_argument(
        "--state-file",
        default="processed_shards.json",
        help="State file with fully processed mapping shards (relative to --out-dir or absolute path).",
    )
    parser.add_argument("--min-free-gb", type=float, default=20.0)

    # from ltx_benchmark.py defaults: 512x512, 25 frames @ 25fps -> >=1.0 sec
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--min-fps", type=float, default=12.0)
    parser.add_argument("--aspect-min", type=float, default=0.8)
    parser.add_argument("--aspect-max", type=float, default=1.25)

    parser.add_argument("--positive-keywords", nargs="*", default=None)
    parser.add_argument("--negative-keywords", nargs="*", default=None)
    parser.add_argument("--positive-keywords-file", default=None)
    parser.add_argument("--negative-keywords-file", default=None)

    parser.add_argument("--keep-archives", action="store_true", help="Do not delete downloaded ZIP parts.")
    parser.add_argument("--keep-shards", action="store_true", help="Do not delete downloaded shard CSV files.")
    parser.add_argument("--dry-run", action="store_true", help="No ZIP/video download; captions+mapping only.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_keywords(words: Sequence[str]) -> List[str]:
    out: List[str] = []
    for w in words:
        w = w.strip().lower()
        if w and w not in out:
            out.append(w)
    return out


def load_keywords_from_file(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def prompt_matches(text: str, positives: Sequence[str], negatives: Sequence[str]) -> bool:
    low = text.lower()
    if positives and not any(word in low for word in positives):
        return False
    if negatives and any(word in low for word in negatives):
        return False
    return True


def infer_col(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def list_hf_files(repo_id: str, repo_type: str, hf_token: Optional[str]) -> List[str]:
    try:
        from huggingface_hub import list_repo_files
    except Exception as exc:
        raise RuntimeError("Please install huggingface_hub.") from exc

    return list_repo_files(repo_id=repo_id, repo_type=repo_type, token=hf_token)


def hf_download_file(
    repo_id: str,
    repo_type: str,
    filename: str,
    local_dir: Path,
    hf_token: Optional[str],
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("Please install huggingface_hub.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename=filename,
        local_dir=str(local_dir),
        token=hf_token,
    )
    return Path(path)


def resolve_archive_parts(zip_name: str, repo_files: Sequence[str]) -> List[str]:
    """Resolve a ZIP name to one file or split parts in HF repo."""
    file_set = set(repo_files)
    if zip_name in file_set:
        return [zip_name]

    stem = Path(zip_name).stem
    parent = str(Path(zip_name).parent)
    if parent == ".":
        parent = ""

    prefix = f"{stem}_part"
    candidates: List[str] = []
    for path in repo_files:
        name = Path(path).name
        if not name.startswith(prefix):
            continue
        if parent and str(Path(path).parent) != parent:
            continue
        candidates.append(path)
    return sorted(candidates)


def materialize_archive_from_hf(
    repo_id: str,
    zip_name: str,
    repo_files: Sequence[str],
    local_dir: Path,
    hf_token: Optional[str],
) -> Optional[Path]:
    """Download archive (single or split parts) and return local .zip path."""
    parts = resolve_archive_parts(zip_name, repo_files)
    if not parts:
        return None

    # Standard case: one complete ZIP exists on HF.
    if len(parts) == 1 and parts[0] == zip_name:
        return hf_download_file(
            repo_id=repo_id,
            repo_type="dataset",
            filename=zip_name,
            local_dir=local_dir,
            hf_token=hf_token,
        )

    # Split archive case: download chunks and concatenate.
    local_dir.mkdir(parents=True, exist_ok=True)
    target_zip = local_dir / Path(zip_name).name
    chunk_paths: List[Path] = []
    for part in parts:
        chunk_path = hf_download_file(
            repo_id=repo_id,
            repo_type="dataset",
            filename=part,
            local_dir=local_dir,
            hf_token=hf_token,
        )
        chunk_paths.append(chunk_path)

    with target_zip.open("wb") as out:
        for chunk in chunk_paths:
            with chunk.open("rb") as src:
                shutil.copyfileobj(src, out, length=16 * 1024 * 1024)

    # Chunks are temporary once reconstructed.
    for chunk in chunk_paths:
        chunk.unlink(missing_ok=True)

    return target_zip


def available_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def ensure_manifest_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_file",
                "caption",
                "width",
                "height",
                "fps",
                "duration_sec",
                "zip_file",
                "zip_member",
                "mapping_shard",
            ]
        )


def append_manifest(
    path: Path,
    video_file: str,
    caption: str,
    probe: VideoProbe,
    zip_file: str,
    zip_member: str,
    mapping_shard: str,
) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                video_file,
                caption,
                probe.width,
                probe.height,
                f"{probe.fps:.5f}",
                f"{probe.duration_sec:.5f}",
                zip_file,
                zip_member,
                mapping_shard,
            ]
        )


def probe_with_ffprobe(path: Path) -> Optional[VideoProbe]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(completed.stdout)
        streams = payload.get("streams", [])
        if not streams:
            return None
        s = streams[0]
        width = int(s.get("width") or 0)
        height = int(s.get("height") or 0)

        fps = 0.0
        rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
        if isinstance(rate, str) and "/" in rate:
            a, b = rate.split("/", 1)
            if float(b) != 0:
                fps = float(a) / float(b)
        duration = float(s.get("duration") or 0.0)
        if duration <= 0:
            nb_frames = s.get("nb_frames")
            if nb_frames and fps > 0:
                duration = float(nb_frames) / fps

        if width <= 0 or height <= 0 or duration <= 0:
            return None
        return VideoProbe(width=width, height=height, fps=fps, duration_sec=duration)
    except Exception:
        return None


def probe_with_opencv(path: Path) -> Optional[VideoProbe]:
    try:
        import cv2  # type: ignore
    except Exception:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if width <= 0 or height <= 0:
        return None
    duration = frame_count / fps if fps > 0 else 0.0
    if duration <= 0:
        return None
    return VideoProbe(width=width, height=height, fps=fps, duration_sec=duration)


def probe_video(path: Path) -> Optional[VideoProbe]:
    return probe_with_ffprobe(path) or probe_with_opencv(path)


def passes_probe_constraints(
    probe: VideoProbe,
    min_width: int,
    min_height: int,
    min_duration_sec: float,
    min_fps: float,
    aspect_min: float,
    aspect_max: float,
) -> bool:
    if probe.width < min_width or probe.height < min_height:
        return False
    if probe.duration_sec < min_duration_sec:
        return False
    if probe.fps > 0 and probe.fps < min_fps:
        return False
    aspect = probe.width / max(probe.height, 1)
    if aspect < aspect_min or aspect > aspect_max:
        return False
    return True


def stable_name(zip_file: str, zip_member: str) -> str:
    digest = hashlib.sha1(f"{zip_file}:{zip_member}".encode("utf-8")).hexdigest()  # nosec B324
    suffix = Path(zip_member).suffix.lower() or ".mp4"
    return f"{digest}{suffix}"


def load_caption_index(
    repo_id: str,
    caption_files: Sequence[str],
    hf_token: Optional[str],
    tmp_dir: Path,
    positives: Sequence[str],
    negatives: Sequence[str],
    keep_shards: bool,
    stats: Stats,
    verbose: bool,
) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for relpath in caption_files:
        log(f"Loading captions: {relpath}")
        local_csv = hf_download_file(
            repo_id=repo_id,
            repo_type="dataset",
            filename=relpath,
            local_dir=tmp_dir / "captions",
            hf_token=hf_token,
        )
        with local_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            video_col = infer_col(cols, ["video", "video_name", "filename", "file", "path"])
            text_col = infer_col(cols, ["caption", "prompt", "text", "description", "sentence"])
            if verbose:
                log(f"  columns: video={video_col}, text={text_col}")
            if not video_col or not text_col:
                raise RuntimeError(
                    f"Cannot detect required columns in {relpath}. "
                    f"Found columns: {cols[:20]}"
                )

            for row in reader:
                stats.captions_seen += 1
                video = str(row.get(video_col, "")).strip()
                text = str(row.get(text_col, "")).strip()
                if not video or not text:
                    continue
                if prompt_matches(text, positives, negatives):
                    stats.captions_matched += 1
                    index[video] = text

        if not keep_shards:
            local_csv.unlink(missing_ok=True)

    log(f"Caption candidates matched by keywords: {len(index)}")
    return index


def list_mapping_shards(repo_id: str, pattern: str, hf_token: Optional[str]) -> List[str]:
    import fnmatch

    all_files = list_hf_files(repo_id=repo_id, repo_type="dataset", hf_token=hf_token)
    return sorted([f for f in all_files if fnmatch.fnmatch(f, pattern)])


def load_processed_shards(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        items = payload.get("completed_mapping_shards", [])
        if isinstance(items, list):
            return {str(x) for x in items}
    except Exception:
        pass
    return set()


def save_processed_shards(state_path: Path, shards: set[str]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"completed_mapping_shards": sorted(shards)}
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    videos_dir = out_dir / "videos"
    tmp_dir = Path(args.tmp_dir).resolve() if args.tmp_dir else (out_dir / "_tmp")
    manifest_path = out_dir / args.manifest_name
    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = out_dir / state_path

    positives = normalize_keywords(
        load_keywords_from_file(args.positive_keywords_file)
        if args.positive_keywords_file
        else (args.positive_keywords or DEFAULT_POSITIVE_KEYWORDS)
    )
    negatives = normalize_keywords(
        load_keywords_from_file(args.negative_keywords_file)
        if args.negative_keywords_file
        else (args.negative_keywords or DEFAULT_NEGATIVE_KEYWORDS)
    )

    videos_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest_header(manifest_path)
    completed_shards = load_processed_shards(state_path)
    if completed_shards:
        log(f"Already completed shards in state: {len(completed_shards)}")

    stats = Stats()
    log(
        "Filter profile: "
        f"min={args.min_width}x{args.min_height}, duration>={args.min_duration_sec}s, "
        f"fps>={args.min_fps}, aspect={args.aspect_min:.2f}..{args.aspect_max:.2f}, "
        f"positive={len(positives)}, negative={len(negatives)}"
    )

    caption_index = load_caption_index(
        repo_id=args.repo_id,
        caption_files=args.caption_files,
        hf_token=args.hf_token,
        tmp_dir=tmp_dir,
        positives=positives,
        negatives=negatives,
        keep_shards=args.keep_shards,
        stats=stats,
        verbose=args.verbose,
    )
    if not caption_index:
        log("No caption matches found. Relax keyword filters.")
        return 0

    log("Listing archive files in dataset repo...")
    archive_repo_files = list_hf_files(args.repo_id, "dataset", args.hf_token)

    shards = list_mapping_shards(args.mapping_repo_id, args.mapping_pattern, args.hf_token)
    if not shards:
        log("No mapping shards found. Check --mapping-repo-id and --mapping-pattern.")
        return 2
    start = max(0, args.start_index)
    end = args.end_index if args.end_index is not None else len(shards)
    shards = shards[start:end]
    if args.max_shards is not None:
        shards = shards[: args.max_shards]
    pending_shards = [s for s in shards if s not in completed_shards]
    skipped_completed = len(shards) - len(pending_shards)
    if skipped_completed:
        log(f"Skipping already completed shards: {skipped_completed}")
    shards = pending_shards
    stats.shards_total = len(shards)
    log(f"Shards to process: {stats.shards_total}")

    for shard_i, shard_relpath in enumerate(shards, start=1):
        if args.max_videos is not None and stats.videos_kept >= args.max_videos:
            log("Reached --max-videos limit, stopping.")
            break
        free = available_free_gb(out_dir)
        if free < args.min_free_gb:
            log(f"Free disk {free:.2f} GB below --min-free-gb={args.min_free_gb:.2f}. Stopping.")
            break

        log(f"[{shard_i}/{stats.shards_total}] Mapping shard: {shard_relpath}")
        local_map = hf_download_file(
            repo_id=args.mapping_repo_id,
            repo_type="dataset",
            filename=shard_relpath,
            local_dir=tmp_dir / "mapping",
            hf_token=args.hf_token,
        )

        by_zip: Dict[str, List[Tuple[str, str, str]]] = {}
        shard_candidates = 0
        with local_map.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            zip_col = infer_col(cols, ["zip_file", "zip", "archive"])
            video_col = infer_col(cols, ["video", "video_name", "filename", "file"])
            path_col = infer_col(cols, ["video_path", "path", "filepath"])
            if args.verbose:
                log(f"  columns: zip={zip_col}, video={video_col}, path={path_col}")
            if not zip_col or not video_col:
                raise RuntimeError(f"Cannot detect mapping columns in {shard_relpath}.")

            for row in reader:
                stats.rows_seen += 1
                video_name = str(row.get(video_col, "")).strip()
                caption = caption_index.get(video_name)
                if not caption:
                    continue
                zip_name = str(row.get(zip_col, "")).strip()
                if not zip_name:
                    continue
                zip_member = str(row.get(path_col, "")).strip() if path_col else ""
                if not zip_member:
                    zip_member = video_name

                stats.rows_keyword_matched += 1
                shard_candidates += 1
                by_zip.setdefault(zip_name, []).append((video_name, zip_member, caption))

        shard_kept = 0
        shard_fully_processed = True
        for zip_name, members in by_zip.items():
            if args.max_videos is not None and stats.videos_kept >= args.max_videos:
                shard_fully_processed = False
                break
            if args.max_videos_per_shard is not None and shard_kept >= args.max_videos_per_shard:
                shard_fully_processed = False
                break
            if args.dry_run:
                shard_fully_processed = False
                continue

            free = available_free_gb(out_dir)
            if free < args.min_free_gb:
                log(f"  Free disk {free:.2f} GB below threshold, stopping shard early.")
                shard_fully_processed = False
                break

            log(f"  Download ZIP: {zip_name} (candidates: {len(members)})")
            local_zip = materialize_archive_from_hf(
                repo_id=args.repo_id,
                zip_name=zip_name,
                repo_files=archive_repo_files,
                local_dir=tmp_dir / "archives",
                hf_token=args.hf_token,
            )
            if local_zip is None:
                log(f"  ! Archive not found (single or split): {zip_name}. Skipping.")
                stats.videos_failed += len(members)
                continue

            try:
                with zipfile.ZipFile(local_zip, "r") as zf:
                    for _video_name, zip_member, caption in members:
                        if args.max_videos is not None and stats.videos_kept >= args.max_videos:
                            shard_fully_processed = False
                            break
                        if args.max_videos_per_shard is not None and shard_kept >= args.max_videos_per_shard:
                            shard_fully_processed = False
                            break

                        out_name = stable_name(zip_name, zip_member)
                        out_path = videos_dir / out_name
                        if out_path.exists():
                            stats.videos_skipped_existing += 1
                            continue

                        try:
                            with zf.open(zip_member) as src, out_path.open("wb") as dst:
                                shutil.copyfileobj(src, dst)
                            stats.videos_downloaded += 1
                        except Exception:
                            stats.videos_failed += 1
                            out_path.unlink(missing_ok=True)
                            continue

                        probe = probe_video(out_path)
                        if probe is None or not passes_probe_constraints(
                            probe=probe,
                            min_width=args.min_width,
                            min_height=args.min_height,
                            min_duration_sec=args.min_duration_sec,
                            min_fps=args.min_fps,
                            aspect_min=args.aspect_min,
                            aspect_max=args.aspect_max,
                        ):
                            stats.videos_rejected_probe += 1
                            out_path.unlink(missing_ok=True)
                            continue

                        stats.videos_kept += 1
                        shard_kept += 1
                        append_manifest(
                            path=manifest_path,
                            video_file=out_name,
                            caption=caption,
                            probe=probe,
                            zip_file=zip_name,
                            zip_member=zip_member,
                            mapping_shard=shard_relpath,
                        )
            finally:
                if not args.keep_archives:
                    local_zip.unlink(missing_ok=True)

        stats.shards_processed += 1
        if shard_fully_processed:
            completed_shards.add(shard_relpath)
            save_processed_shards(state_path, completed_shards)
            log("  shard state: marked as fully processed")
        else:
            log("  shard state: not fully processed (will be retried on restart)")
        log(
            f"  shard summary: candidates={shard_candidates}, "
            f"kept_this_shard={shard_kept}, total_kept={stats.videos_kept}"
        )
        if not args.keep_shards:
            local_map.unlink(missing_ok=True)

    log("\nDone.")
    log(f"Captions seen: {stats.captions_seen}")
    log(f"Captions keyword matched: {stats.captions_matched}")
    log(f"Shards processed: {stats.shards_processed}/{stats.shards_total}")
    log(f"Mapping rows seen: {stats.rows_seen}")
    log(f"Mapping keyword matched: {stats.rows_keyword_matched}")
    log(f"Videos extracted: {stats.videos_downloaded}")
    log(f"Videos kept: {stats.videos_kept}")
    log(f"Videos rejected by probe: {stats.videos_rejected_probe}")
    log(f"Videos skipped existing: {stats.videos_skipped_existing}")
    log(f"Videos failed: {stats.videos_failed}")
    log(f"Manifest: {manifest_path}")
    log(f"Videos dir: {videos_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
