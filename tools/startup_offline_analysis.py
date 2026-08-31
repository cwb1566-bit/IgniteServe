#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI for collecting and validating IgniteServe startup contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm.v1.engine.startup_analysis.generator import StartupContractGenerator
from vllm.v1.engine.startup_analysis.nsight import (
    append_nsight_resource_events,
    extract_nsight_stage_resources,
)
from vllm.v1.engine.startup_analysis.validator import (
    generate_edge_removal_candidates,
    run_contract_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate")
    generate.add_argument("--trace", action="append", type=Path, required=True)
    generate.add_argument("--seed-plan", type=Path)
    generate.add_argument("--call-profile", action="append", type=Path,
                          default=[])
    generate.add_argument("--without-seed-edges", action="store_true")
    generate.add_argument("--output", type=Path, required=True)

    nsight = commands.add_parser("import-nsight")
    nsight.add_argument("--sqlite", type=Path, required=True)
    nsight.add_argument("--trace", type=Path, required=True)
    nsight.add_argument("--output-trace", type=Path, required=True)
    nsight.add_argument("--output-summary", type=Path, required=True)

    perturb = commands.add_parser("perturb")
    perturb.add_argument("--contract", type=Path, required=True)
    perturb.add_argument("--output-dir", type=Path, required=True)
    perturb.add_argument("--maximum-confidence", type=float, default=0.89)

    validate = commands.add_parser("validate")
    validate.add_argument("--candidate", action="append", type=Path,
                          required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--timeout-s", type=float, default=600.0)
    validate.add_argument("--expected-fingerprint")
    validate.add_argument("--isolate-compile-cache", action="store_true")
    validate.add_argument("run_command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate":
        generator = StartupContractGenerator(
            args.trace,
            seed_plan=args.seed_plan,
            include_seed_edges=not args.without_seed_edges,
            call_profile_paths=args.call_profile,
        )
        contract = generator.write(args.output)
        print(json.dumps(contract["summary"], sort_keys=True))
        return 0
    if args.command == "import-nsight":
        analysis = extract_nsight_stage_resources(args.sqlite)
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(
            json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
        append_nsight_resource_events(args.trace, analysis, args.output_trace)
        print(json.dumps({
            "stage_count": analysis["stage_count"],
            "analysis_elapsed_s": analysis["analysis_elapsed_s"],
        }, sort_keys=True))
        return 0
    if args.command == "perturb":
        paths = generate_edge_removal_candidates(
            args.contract, args.output_dir,
            maximum_confidence=args.maximum_confidence)
        print(json.dumps({"candidate_count": len(paths),
                          "paths": [str(path) for path in paths]},
                         sort_keys=True))
        return 0
    if args.command == "validate":
        command = list(args.run_command)
        if command and command[0] == "--":
            command.pop(0)
        if not command:
            raise SystemExit("validate requires a command after --")
        report = run_contract_validation(
            command, args.candidate, args.output_dir,
            timeout_s=args.timeout_s,
            expected_fingerprint=args.expected_fingerprint,
            isolate_compile_cache=args.isolate_compile_cache)
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
