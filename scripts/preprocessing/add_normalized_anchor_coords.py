#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WIDTH = 1408.0
HEIGHT = 1408.0


def normalize_pixel(pixel: Any) -> list[float] | None:
    if not isinstance(pixel, (list, tuple)) or len(pixel) < 2:
        return None

    if pixel[0] is None or pixel[1] is None:
        return None

    return [
        float(pixel[0]) / WIDTH,
        float(pixel[1]) / HEIGHT,
    ]


def add_normalized_anchor_coords(data: dict[str, Any]) -> int:
    """
    Add object_y_normalized_projected_pixel to step 5b and 5c answer_metadata.

    5b uses:
        object_y_projected_pixel

    5c uses:
        object_y_pixel
    """
    updated_count = 0

    for trajectory_id, traj in data.items():
        branch_groups = traj.get("branch_groups", {}) or {}

        for group_name, steps in branch_groups.items():
            if not isinstance(steps, list):
                continue

            for step in steps:
                step_id = str(step.get("step")).strip()

                if step_id not in {"5b", "5c"}:
                    continue

                meta = step.get("answer_metadata", {})
                if not isinstance(meta, dict):
                    continue

                # Do not overwrite if it already exists.
                if "object_y_normalized_projected_pixel" in meta:
                    continue

                if step_id == "5b":
                    pixel = meta.get("object_y_projected_pixel")
                else:  # step_id == "5c"
                    pixel = meta.get("object_y_pixel")

                normalized = normalize_pixel(pixel)
                if normalized is None:
                    continue

                meta["object_y_normalized_projected_pixel"] = normalized
                updated_count += 1

    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add normalized anchor coordinates to 5b and 5c in staged OOS VQA JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated_count = add_normalized_anchor_coords(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Added normalized anchor coordinates to {updated_count} step(s).")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()