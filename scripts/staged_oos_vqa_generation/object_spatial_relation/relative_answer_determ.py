from __future__ import annotations

from dataclasses import dataclass
import math



RELATIVE_CHOICES = [
    "Front-left",
    "Front-right",
    "Back-left",
    "Back-right",
]


@dataclass(frozen=True)
class RelativeAnswer:
    """Resolved relative answer labels from an anchored displacement vector."""
    vector: list[float]
    choices: list[str]
    correct_idx: int
    acceptable_idxs: list[int]


def _choice_index(label: str) -> int:
    return RELATIVE_CHOICES.index(label)


def classify_relative_quadrant_robust(
    a_relative_to_b_egocentric: list[float],
    *,
    center_margin: float = 0.05,
    depth_margin: float = 0.05,
    angle_margin_deg: float = 5.0,
) -> tuple[int | None, str | None, dict]:
    """
    Match the staged_oos_question_generator.py quadrant rule.

    Uses:
      x -> left/right
      z -> front/back

    Rejects ambiguous cases near:
      - x = 0
      - z = 0
      - side-view angular boundary
    """
    if a_relative_to_b_egocentric is None or len(a_relative_to_b_egocentric) < 3:
        return None, None, {"reason": "no_coordinates"}

    x, _, z = [float(v) for v in a_relative_to_b_egocentric[:3]]

    if abs(z) < depth_margin:
        return None, None, {"reason": "near_z_boundary", "x": x, "z": z}

    if abs(x) < center_margin:
        return None, None, {"reason": "near_x_boundary", "x": x, "z": z}

    yaw_deg = math.degrees(math.atan2(x, z))
    if abs(abs(yaw_deg) - 90.0) < angle_margin_deg:
        return None, None, {"reason": "near_diagonal_boundary", "yaw_deg": yaw_deg, "x": x, "z": z}

    if z > 0:
        if x < 0:
            return 0, "Front-left", {"x": x, "z": z}
        return 1, "Front-right", {"x": x, "z": z}
    else:
        if x < 0:
            return 2, "Back-left", {"x": x, "z": z}
        return 3, "Back-right", {"x": x, "z": z}


def determine_relative_answer_from_vector(
    a_relative_to_b_egocentric: list[float],
    center_margin: float = 0.05,
    depth_margin: float = 0.05,
    angle_margin_deg: float = 5.0,
) -> RelativeAnswer:
    correct_idx, label, debug = classify_relative_quadrant_robust(
        a_relative_to_b_egocentric,
        center_margin=center_margin,
        depth_margin=depth_margin,
        angle_margin_deg=angle_margin_deg,
    )

    if correct_idx is None or label is None:
        raise ValueError(f"Relative vector is ambiguous for quadrant classification: {debug}")

    return RelativeAnswer(
        vector=[float(v) for v in a_relative_to_b_egocentric],
        choices=RELATIVE_CHOICES,
        correct_idx=correct_idx,
        acceptable_idxs=[correct_idx],
    )


