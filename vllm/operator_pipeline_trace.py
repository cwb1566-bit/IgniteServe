# SPDX-License-Identifier: Apache-2.0
"""Opt-in profiler ranges for operator-pipeline experiments."""

import os
from contextlib import nullcontext
from typing import ContextManager

import torch


def operator_pipeline_trace_enabled() -> bool:
    return os.environ.get("VLLM_OPERATOR_PIPELINE_TRACE", "").lower() in {
        "1", "true", "yes", "on"
    }


def operator_pipeline_range(label: str) -> ContextManager:
    if not operator_pipeline_trace_enabled():
        return nullcontext()
    return torch.profiler.record_function(f"vllm.operator_pipeline::{label}")
