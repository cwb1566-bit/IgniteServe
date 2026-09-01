# SPDX-License-Identifier: Apache-2.0
"""Parse low-overhead py-spy sampling profiles into boundary candidates."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def parse_speedscope_profiles(
    paths: Iterable[Path],
    *,
    file_markers: tuple[str, ...] = ("/vllm/", "torch/_dynamo", "torch/_inductor"),
    excluded_file_markers: tuple[str, ...] = (),
    excluded_qualnames: tuple[str, ...] = (),
    required_stack_qualnames: tuple[str, ...] = (),
    profile_name_markers: tuple[str, ...] = (),
    max_profile_s: float | None = None,
) -> list[dict[str, Any]]:
    stats: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        frames = payload.get("shared", {}).get("frames", [])
        for profile in payload.get("profiles", []):
            if profile.get("type") != "sampled":
                continue
            samples = profile.get("samples", [])
            weights = profile.get("weights") or [1.0] * len(samples)
            profile_name = str(profile.get("name", ""))
            if profile_name_markers and not any(
                    marker in profile_name for marker in profile_name_markers):
                continue
            elapsed = 0.0
            for stack, weight_value in zip(samples, weights):
                weight = float(weight_value)
                if max_profile_s is not None and elapsed >= max_profile_s:
                    break
                elapsed += weight
                stack_names = [str(frames[int(frame_index)].get("name", ""))
                               for frame_index in stack]
                if required_stack_qualnames and not any(
                        marker in name for marker in required_stack_qualnames
                        for name in stack_names):
                    continue
                valid_indices = []
                for frame_index in stack:
                    frame = frames[int(frame_index)]
                    filename = str(frame.get("file", ""))
                    qualname = str(frame.get("name", ""))
                    if file_markers and not any(
                            marker in filename for marker in file_markers):
                        continue
                    if excluded_file_markers and any(
                            marker in filename for marker in excluded_file_markers):
                        continue
                    if excluded_qualnames and any(
                            marker in qualname for marker in excluded_qualnames):
                        continue
                    valid_indices.append(int(frame_index))
                for frame_index in set(valid_indices):
                    frame = frames[frame_index]
                    key = (str(frame.get("name", "")),
                           str(frame.get("file", "")),
                           int(frame.get("line", 0) or 0))
                    stat = stats.setdefault(key, {
                        "qualname": key[0],
                        "filename": key[1],
                        "lineno": key[2],
                        "sampled_inclusive_s": 0.0,
                        "sampled_self_s": 0.0,
                        "sample_count": 0,
                        "profiles": set(),
                    })
                    stat["sampled_inclusive_s"] += weight
                    stat["sample_count"] += 1
                    stat["profiles"].add(profile_name)
                if valid_indices:
                    leaf = frames[valid_indices[-1]]
                    key = (str(leaf.get("name", "")),
                           str(leaf.get("file", "")),
                           int(leaf.get("line", 0) or 0))
                    stats[key]["sampled_self_s"] += weight
    candidates = []
    for stat in stats.values():
        stat["profiles"] = sorted(stat["profiles"])
        candidates.append(stat)
    candidates.sort(
        key=lambda item: (
            item["sampled_inclusive_s"], item["sampled_self_s"]),
        reverse=True)
    return candidates
