# SPDX-License-Identifier: Apache-2.0
"""Conservative source discovery and evidence-bound finalization, without GPUs."""
import argparse
import ast
import json
from pathlib import Path

from .contract import digest, identity, variant_nodes


def analyze(source, class_name, method_name):
    tree = ast.parse(Path(source).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == method_name)
    regions = []
    # Do not pretend a call trace proves extraction or alias safety.
    for index, statement in enumerate(method.body):
        calls = [ast.unparse(n.func) for n in ast.walk(statement)
                 if isinstance(n, ast.Call)]
        reads = sorted({n.id for n in ast.walk(statement)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
        writes = sorted({n.id for n in ast.walk(statement)
                         if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)})
        regions.append(dict(id=index, start=statement.lineno,
                            end=statement.end_lineno, calls=calls,
                            reads=reads, writes=writes, review="required"))
    return dict(source_sha256=digest(source), entry=f"{class_name}.{method_name}",
                regions=regions, note="Calls/control/aliases require adapter review")


def finalize(candidate, adapter, review, validation, root, output):
    payload = json.loads(Path(candidate).read_text())
    decisions = json.loads(Path(review).read_text())
    evidence = json.loads(Path(validation).read_text())
    expected = identity(payload)
    if digest(adapter) != payload["adapter_sha256"]:
        raise ValueError("Adapter changed since candidate generation")
    for name, value in payload["sources"].items():
        if digest(Path(root) / name) != value:
            raise ValueError(f"Reviewed source changed: {name}")
    if decisions.get("identity") != expected or evidence.get("identity") != expected:
        raise ValueError("Review/validation is for a different candidate")
    required = {n["name"] for v in payload["variants"] for n in variant_nodes(payload, v)}
    if set(decisions.get("actions", {})) != required or any(
            v != "approved" for v in decisions["actions"].values()):
        raise ValueError("Every action requires explicit review")
    if decisions.get("constraints") != "approved" or not decisions.get("rationale"):
        raise ValueError("State, control, affinity and opaque constraints need review")
    if not evidence.get("serial_equivalence") or not evidence.get("scheduled_equivalence"):
        raise ValueError("Both serial and scheduled generated-adapter verification required")
    if set(evidence.get("variants", [])) != {v["name"] for v in payload["variants"]}:
        raise ValueError("Every variant needs verification")
    payload["status"] = "final"
    payload["validation"] = {"identity": expected, "review": decisions,
                             "verification": evidence}
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("analyze")
    for key in ("source", "class-name", "method-name", "output"):
        discover.add_argument("--" + key, required=True)
    final = sub.add_parser("finalize")
    for key in ("candidate", "adapter", "review", "validation", "root", "output"):
        final.add_argument("--" + key, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "analyze":
        output = args.pop("output")
        Path(output).write_text(json.dumps(analyze(**args), indent=2) + "\n")
    else:
        finalize(**args)


if __name__ == "__main__":
    main()
