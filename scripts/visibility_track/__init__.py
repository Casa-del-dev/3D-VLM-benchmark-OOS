"""HD-EPIC visibility-track pipeline.

Top-level orchestration lives in `generate_visibility_track.py`. Individual
stages are kept as importable modules under sibling subpackages:

    open_close_track/    — fixture open/closed intervals from narrations
    in_view_track/       — per-object in-view sampling via 3D projection
    object_detection/    — Grounding DINO ROI-based visibility refinement
"""
