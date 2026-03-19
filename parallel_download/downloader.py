#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

_BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/"
_MD5_MANIFEST_URL = "https://raw.githubusercontent.com/hd-epic/hd-epic-downloader/main/data/md5.txt"

# Participant-relevant categories. Digital-Twin is global and included.
# Hands-Masks is global but not participant-specific, so we exclude it by default.
_ALLOWED_TOP_LEVEL_BASE = {
    "videos",
    "slam-and-gaze",
    "audio-hdf5",
    "digital-twin",
    "root",
}
_DEFAULT_DATASET_DIRNAME = "HD-EPIC"


@dataclass(frozen=True)
class FileEntry:
    rel_path: str
    md5: str


def entry_category(rel_path: str) -> str:
    p = Path(rel_path)
    if len(p.parts) == 1:
        return "root"
    name = p.parts[0]
    if name == "Videos":
        return "videos"
    if name == "SLAM-and-Gaze":
        return "slam-and-gaze"
    if name == "Audio-HDF5":
        return "audio-hdf5"
    if name == "Digital-Twin":
        return "digital-twin"
    if name == "VRS":
        return "vrs"
    if name == "Hands-Masks":
        return "hands-masks"
    return name.lower()


def md5_checksum(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch_manifest_text(timeout_sec: int) -> str:
    req = urllib.request.Request(_MD5_MANIFEST_URL)
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        return resp.read().decode("utf-8")


def parse_manifest(text: str) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid md5 manifest line: {line}")
        md5, rel = parts
        rel = rel.strip()
        if not rel.startswith("./"):
            raise ValueError(f"Unexpected relative path format: {rel}")
        entries.append(FileEntry(rel_path=rel[2:], md5=md5))
    if not entries:
        raise ValueError("md5 manifest is empty")
    return entries


def participant_from_path(rel_path: str) -> str | None:
    m = re.search(r"/(P0[1-9])(?:/|$)", "/" + rel_path)
    return None if m is None else m.group(1)


def choose_entries(entries: Iterable[FileEntry], participant: str, include_vrs: bool) -> list[FileEntry]:
    allowed = set(_ALLOWED_TOP_LEVEL_BASE)
    if include_vrs:
        allowed.add("vrs")

    selected: list[FileEntry] = []
    for entry in entries:
        category = entry_category(entry.rel_path)
        if category not in allowed:
            continue

        if category in {"root", "digital-twin"}:
            selected.append(entry)
            continue

        part = participant_from_path(entry.rel_path)
        if part == participant:
            selected.append(entry)

    if not selected:
        raise ValueError("No files selected. Check filtering logic.")
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel HD-EPIC downloader with participant-focused selection, matching "
            "hd-epic-downloader directory organization."
        )
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("."),
        help=(
            "Output parent path. A fixed HD-EPIC folder is created inside it "
            "(default: current directory)."
        ),
    )
    parser.add_argument("--participant", type=str, default="P02", help="Participant ID, e.g. P02")
    parser.add_argument(
        "--include-vrs",
        action="store_true",
        help="Include VRS for the participant (default: disabled).",
    )
    parser.add_argument("--workers", type=int, default=12, help="Parallel download workers")
    parser.add_argument(
        "--segments-per-file",
        type=int,
        default=1,
        help="HTTP range segments per file (1 disables per-file segmentation).",
    )
    parser.add_argument(
        "--split-threshold-mib",
        type=int,
        default=256,
        help="Only segment files of at least this size (MiB).",
    )
    parser.add_argument("--timeout-sec", type=int, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=2, help="Retries per file")
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024, help="Read chunk size")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"P0[1-9]", args.participant):
        raise ValueError("participant must match P01..P09")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.segments_per_file < 1:
        raise ValueError("segments-per-file must be >= 1")
    if args.split_threshold_mib < 1:
        raise ValueError("split-threshold-mib must be >= 1")


def resolve_output_root(output_path: Path) -> Path:
    output_root = output_path.expanduser().resolve() / _DEFAULT_DATASET_DIRNAME
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def download_one(
    entry: FileEntry,
    output_root: Path,
    timeout_sec: int,
    retries: int,
    chunk_bytes: int,
    segments_per_file: int,
    split_threshold_bytes: int,
    progress_callback,
) -> tuple[str, str]:
    dst = output_root / entry.rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and md5_checksum(dst) == entry.md5:
        return ("skipped", entry.rel_path)

    url = _BASE_URL + urllib.parse.quote(entry.rel_path)
    tmp = dst.with_suffix(dst.suffix + ".part")

    def _probe_remote() -> tuple[int | None, bool]:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                content_len = resp.getheader("content-length")
                accept_ranges = (resp.getheader("accept-ranges") or "").lower()
                size = int(content_len) if content_len is not None else None
                return size, ("bytes" in accept_ranges)
        except Exception:
            return None, False

    def _download_single_stream() -> None:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, tmp.open("wb") as out:
            content_len = resp.getheader("content-length")
            if content_len is not None:
                progress_callback(entry, "set_total", int(content_len))

            while True:
                buf = resp.read(chunk_bytes)
                if not buf:
                    break
                out.write(buf)
                progress_callback(entry, "bytes", len(buf))

    def _download_segment(part_path: Path, start: int, end: int) -> None:
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp, part_path.open("wb") as out:
            status = getattr(resp, "status", None)
            if status != 206:
                raise RuntimeError(f"Server did not honor Range request for {entry.rel_path} (status={status})")

            while True:
                buf = resp.read(chunk_bytes)
                if not buf:
                    break
                out.write(buf)
                progress_callback(entry, "bytes", len(buf))

    def _download_segmented(remote_size: int) -> None:
        progress_callback(entry, "set_total", remote_size)

        seg_count = min(segments_per_file, remote_size)
        base = remote_size // seg_count
        rem = remote_size % seg_count

        parts: list[Path] = []
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for i in range(seg_count):
            seg_size = base + (1 if i < rem else 0)
            start = cursor
            end = cursor + seg_size - 1
            cursor = end + 1
            ranges.append((start, end))
            parts.append(tmp.with_name(tmp.name + f".seg{i:03d}"))

        try:
            with ThreadPoolExecutor(max_workers=seg_count) as seg_pool:
                seg_futs = [
                    seg_pool.submit(_download_segment, part, rng[0], rng[1])
                    for part, rng in zip(parts, ranges)
                ]
                for seg_f in as_completed(seg_futs):
                    seg_f.result()

            with tmp.open("wb") as out:
                for part in parts:
                    with part.open("rb") as inp:
                        while True:
                            buf = inp.read(chunk_bytes)
                            if not buf:
                                break
                            out.write(buf)
        finally:
            for part in parts:
                if part.exists():
                    part.unlink()

    last_exc: Exception | None = None
    for _ in range(retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()

            remote_size, range_ok = _probe_remote()
            can_segment = (
                segments_per_file > 1
                and range_ok
                and remote_size is not None
                and remote_size >= split_threshold_bytes
            )

            if can_segment:
                _download_segmented(remote_size)
            else:
                _download_single_stream()

            got = md5_checksum(tmp)
            if got != entry.md5:
                raise RuntimeError(
                    f"MD5 mismatch for {entry.rel_path}: expected {entry.md5}, got {got}"
                )

            os.replace(tmp, dst)
            return ("downloaded", entry.rel_path)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    raise RuntimeError(f"Failed {entry.rel_path} after {retries + 1} attempts: {last_exc}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    split_threshold_bytes = args.split_threshold_mib * 1024 * 1024

    output_root = resolve_output_root(args.output_path)

    manifest = parse_manifest(fetch_manifest_text(args.timeout_sec))
    selected = choose_entries(manifest, args.participant, include_vrs=args.include_vrs)

    mode_msg = "with VRS" if args.include_vrs else "without VRS"
    print(f"Selected {len(selected)} files for {args.participant} ({mode_msg}).")
    print(f"Downloading into: {output_root}")

    lock = threading.Lock()
    done = 0
    downloaded = 0
    skipped = 0
    failed = 0

    categories = sorted({entry_category(e.rel_path) for e in selected})
    category_position = {cat: idx for idx, cat in enumerate(categories)}
    bars = {
        cat: tqdm(
            total=0,
            desc=f"{cat:12s}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            position=idx,
            leave=True,
        )
        for cat, idx in category_position.items()
    }
    files_bar = tqdm(
        total=len(selected),
        desc="files",
        unit="file",
        dynamic_ncols=True,
        position=len(categories),
        leave=True,
    )

    per_file_totals: dict[str, int] = {}

    def progress_callback(entry: FileEntry, event: str, value: int) -> None:
        cat = entry_category(entry.rel_path)
        if cat not in bars:
            return
        with lock:
            if event == "set_total":
                key = entry.rel_path
                previous = per_file_totals.get(key, 0)
                if value > previous:
                    bars[cat].total = bars[cat].total + (value - previous)
                    per_file_totals[key] = value
                    bars[cat].refresh()
            elif event == "bytes":
                bars[cat].update(value)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    download_one,
                    entry,
                    output_root,
                    args.timeout_sec,
                    args.retries,
                    args.chunk_bytes,
                    args.segments_per_file,
                    split_threshold_bytes,
                    progress_callback,
                ): entry
                for entry in selected
            }

            total = len(futures)
            for fut in as_completed(futures):
                entry = futures[fut]
                try:
                    status, _ = fut.result()
                    with lock:
                        done += 1
                        if status == "downloaded":
                            downloaded += 1
                        elif status == "skipped":
                            skipped += 1
                        files_bar.update(1)
                        tqdm.write(f"[{done}/{total}] {status}: {entry.rel_path}")
                except Exception as exc:  # noqa: BLE001
                    with lock:
                        done += 1
                        failed += 1
                        files_bar.update(1)
                        tqdm.write(f"[{done}/{total}] failed: {entry.rel_path} -> {exc}")
    finally:
        for bar in bars.values():
            bar.close()
        files_bar.close()

    print("-")
    print(f"Finished. downloaded={downloaded}, skipped={skipped}, failed={failed}, total={len(selected)}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
