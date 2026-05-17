from __future__ import annotations

from typing import Any


def build_tracking_prompt(sample: dict[str, Any], template: str) -> str:
    return template.format(
        sequence_id=sample.get("sequence_id", ""),
        frame_id=sample.get("frame_id", ""),
        frame_idx=sample.get("frame_idx", ""),
        num_targets=len(sample.get("target_boxes", [])),
    )
