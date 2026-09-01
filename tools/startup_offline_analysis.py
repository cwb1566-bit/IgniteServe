#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CLI for collecting and validating IgniteServe startup contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm.v1.engine.startup_analysis.discovery import StartupDiscoveryBuilder
from vllm.v1.engine.startup_analysis.enricher import StartupEvidenceEnricher
from vllm.v1.engine.startup_analysis.extractor import StartupActionExtractor
from vllm.v1.engine.startup_analysis.finalizer import (
    FinalizationBlocked,
    finalize_startup_contract,
    generate_node_coarsening_suggestions,
)
from vllm.v1.engine.startup_analysis.generator import StartupContractGenerator
from vllm.v1.engine.startup_analysis.nsight import (
    append_nsight_resource_events,
    extract_nsight_stage_resources,
)
from vllm.v1.engine.startup_analysis.validator import (
    generate_edge_removal_candidates,
    run_contract_validation,
)
from vllm.v1.engine.startup_analysis.workflow import (
    apply_edge_validation_report,
    verify_serial_equivalence,
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

    discover = commands.add_parser("discover")
    discover.add_argument("--trace", action="append", type=Path,
                          required=True)
    discover.add_argument("--call-profile", action="append", type=Path,
                          default=[])
    discover.add_argument("--nsight-summary", action="append", type=Path,
                          default=[])
    discover.add_argument("--source", type=Path)
    discover.add_argument("--root-stage", default="engine_startup.root")
    discover.add_argument("--output", type=Path, required=True)

    extract = commands.add_parser("extract")
    extract.add_argument("--source", type=Path, required=True)
    extract.add_argument("--discovery", type=Path, required=True)
    extract.add_argument("--class-name", default="EngineCore")
    extract.add_argument("--method-name", default="_initialize_engine_core")
    extract.add_argument("--output-root", type=Path, required=True)

    enrich = commands.add_parser("enrich")
    enrich.add_argument("--contract", type=Path, required=True)
    enrich.add_argument("--variant-trace", action="append", required=True,
                        metavar="VARIANT=PATH")
    enrich.add_argument("--variant-match", action="append", default=[],
                        metavar="VARIANT=JSON")
    enrich.add_argument("--variant-nsight", action="append", default=[],
                        metavar="VARIANT=PATH")
    enrich.add_argument("--review-dir", type=Path)
    enrich.add_argument("--output", type=Path, required=True)

    coarsen = commands.add_parser("coarsen")
    coarsen.add_argument("--contract", type=Path, required=True)
    coarsen.add_argument("--review-file", type=Path, required=True)
    coarsen.add_argument("--maximum-combined-cost-s", type=float,
                         default=0.05)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--contract", type=Path, required=True)
    finalize.add_argument("--adapter", type=Path, required=True)
    finalize.add_argument("--review-dir", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    finalize.add_argument("--compatibility-json", type=Path)

    serial = commands.add_parser("verify-serial")
    serial.add_argument("--contract", type=Path, required=True)
    serial.add_argument("--baseline-result", action="append", required=True,
                        metavar="VARIANT=PATH")
    serial.add_argument("--candidate-result", action="append", required=True,
                        metavar="VARIANT=PATH")
    serial.add_argument("--candidate-trace", action="append", required=True,
                        metavar="VARIANT=PATH")
    serial.add_argument(
        "--execution-kind",
        choices=("source_instrumentation", "generated_adapter"),
        default="source_instrumentation",
        help=("Whether the candidate results came from temporary source "
              "wrappers or the generated adapter. Only generated_adapter "
              "evidence can satisfy finalization."),
    )
    serial.add_argument("--report", type=Path, required=True)
    serial.add_argument("--output-contract", type=Path, required=True)

    apply_validation = commands.add_parser("apply-edge-validation")
    apply_validation.add_argument("--contract", type=Path, required=True)
    apply_validation.add_argument("--validation-report", type=Path,
                                  required=True)
    apply_validation.add_argument("--review-file", type=Path, required=True)
    apply_validation.add_argument("--output", type=Path, required=True)

    nsight = commands.add_parser("import-nsight")
    nsight.add_argument("--sqlite", type=Path, required=True)
    nsight.add_argument("--trace", type=Path, required=True)
    nsight.add_argument("--output-trace", type=Path, required=True)
    nsight.add_argument("--output-summary", type=Path, required=True)

    perturb = commands.add_parser("perturb")
    perturb.add_argument("--contract", type=Path, required=True)
    perturb.add_argument("--output-dir", type=Path, required=True)
    perturb.add_argument("--maximum-confidence", type=float, default=0.89)
    perturb.add_argument("--variant")

    validate = commands.add_parser("validate")
    validate.add_argument("--candidate", action="append", type=Path,
                          required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--timeout-s", type=float, default=600.0)
    validate.add_argument("--expected-fingerprint")
    validate.add_argument("--isolate-compile-cache", action="store_true")
    validate.add_argument("--repeats", type=int, default=1)
    validate.add_argument("--baseline-elapsed-s", action="append", type=float,
                          default=[])
    validate.add_argument("--maximum-regression-fraction", type=float,
                          default=0.05)
    validate.add_argument("run_command", nargs=argparse.REMAINDER)
    return parser


def _group_paths(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected VARIANT=PATH, got {value!r}")
        name, path = value.split("=", 1)
        result.setdefault(name, []).append(Path(path))
    return result


def _group_json(values: list[str]) -> dict[str, dict]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"expected VARIANT=JSON, got {value!r}")
        name, payload = value.split("=", 1)
        result[name] = json.loads(payload)
    return result


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
    if args.command == "discover":
        payload = StartupDiscoveryBuilder(
            args.trace,
            call_profile_paths=args.call_profile,
            nsight_summary_paths=args.nsight_summary,
            source_path=args.source,
            root_stage=args.root_stage,
        ).write(args.output)
        print(json.dumps({
            "call_boundary_count": len(payload["call_boundaries"]),
            "root_observations": payload["root_cost_s"]["observations"],
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    if args.command == "extract":
        outputs = StartupActionExtractor(
            args.source, args.discovery,
            class_name=args.class_name,
            method_name=args.method_name,
        ).generate(args.output_root)
        print(json.dumps({key: str(value) for key, value in outputs.items()},
                         sort_keys=True))
        return 0
    if args.command == "enrich":
        payload = StartupEvidenceEnricher(
            args.contract,
            _group_paths(args.variant_trace),
            matches_by_variant=_group_json(args.variant_match),
            nsight_by_variant=_group_paths(args.variant_nsight),
        ).write(args.output, review_dir=args.review_dir)
        print(json.dumps({
            "variants": [item["name"] for item in payload["variants"]],
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    if args.command == "coarsen":
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        suggestions = generate_node_coarsening_suggestions(
            contract,
            maximum_combined_cost_s=args.maximum_combined_cost_s)
        args.review_file.parent.mkdir(parents=True, exist_ok=True)
        args.review_file.write_text(json.dumps({
            "schema": "igniteserve-manual-review-v1",
            "kind": "node_coarsening",
            "items": suggestions,
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"suggestion_count": len(suggestions)},
                         sort_keys=True))
        return 0
    if args.command == "finalize":
        compatibility = None
        if args.compatibility_json:
            compatibility = json.loads(
                args.compatibility_json.read_text(encoding="utf-8"))
        try:
            outputs = finalize_startup_contract(
                args.contract, args.adapter, args.review_dir,
                args.output_dir, compatibility_override=compatibility)
        except FinalizationBlocked as exc:
            print(json.dumps({
                "status": "blocked",
                "blocker_count": len(exc.blockers),
                "report": str(exc.report),
            }, sort_keys=True))
            return 2
        print(json.dumps({"status": "final",
                          **{key: str(value) for key, value in outputs.items()}},
                         sort_keys=True))
        return 0
    if args.command == "verify-serial":
        report = verify_serial_equivalence(
            args.contract,
            _group_paths(args.baseline_result),
            _group_paths(args.candidate_result),
            _group_paths(args.candidate_trace),
            args.report,
            execution_kind=args.execution_kind,
        )
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        contract.setdefault("validation", {})["serial_equivalence"] = {
            "status": "passed" if report["passed"] else "failed",
            "report": str(args.report),
            "execution_kind": args.execution_kind,
        }
        args.output_contract.parent.mkdir(parents=True, exist_ok=True)
        args.output_contract.write_text(
            json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"passed": report["passed"],
                          "report": str(args.report)}, sort_keys=True))
        return 0 if report["passed"] else 3
    if args.command == "apply-edge-validation":
        payload = apply_edge_validation_report(
            args.contract, args.validation_report, args.output,
            review_file=args.review_file)
        summary = payload.get("validation", {}).get(
            "edge_perturbations", {})
        print(json.dumps(summary, sort_keys=True))
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
            maximum_confidence=args.maximum_confidence,
            variant_name=args.variant)
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
            isolate_compile_cache=args.isolate_compile_cache,
            repeats=args.repeats,
            baseline_elapsed_s=args.baseline_elapsed_s,
            maximum_regression_fraction=args.maximum_regression_fraction)
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
