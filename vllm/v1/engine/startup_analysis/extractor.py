# SPDX-License-Identifier: Apache-2.0
"""Conservative source-to-contract extraction for unseeded startup code."""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identifier(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not value or value[0].isdigit() or keyword.iskeyword(value):
        value = f"stage_{value}"
    return value


def _attribute_name(node: ast.AST) -> Optional[str]:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _assignment_names(node: ast.AST) -> Optional[list[str]]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        names = []
        for element in node.elts:
            child = _assignment_names(element)
            if child is None:
                return None
            names.extend(child)
        return names
    return None


class _UseDef(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.calls: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        target = self.writes if isinstance(node.ctx, (ast.Store, ast.Del)) else self.reads
        target.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = _attribute_name(node)
        if name and name.startswith("self."):
            target = self.writes if isinstance(
                node.ctx, (ast.Store, ast.Del)) else self.reads
            target.add(name)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _attribute_name(node.func)
        if name is None and isinstance(node.func, ast.Name):
            name = node.func.id
        if name:
            self.calls.append(name)
        if not (name and name.startswith("self.")):
            self.visit(node.func)
        for argument in node.args:
            self.visit(argument)
        for keyword_node in node.keywords:
            self.visit(keyword_node.value)


class _ActionTransformer(ast.NodeTransformer):
    def __init__(self, state_names: set[str], input_names: set[str]) -> None:
        self.state_names = state_names
        self.input_names = input_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "self":
            return ast.copy_location(ast.Name(id="engine_core", ctx=node.ctx), node)
        if isinstance(node.ctx, ast.Load) and node.id in self.state_names:
            key = f"input.{node.id}" if node.id in self.input_names else node.id
            return ast.copy_location(ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="context", ctx=ast.Load()),
                    attr="require", ctx=ast.Load()),
                args=[ast.Constant(key)], keywords=[]), node)
        return node


def _simple_call(statement: ast.stmt) -> tuple[Optional[ast.Call], list[str]]:
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return statement.value, []
    if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.value, ast.Call)):
        outputs = _assignment_names(statement.targets[0])
        if outputs is not None:
            return statement.value, outputs
    if (isinstance(statement, ast.AnnAssign)
            and isinstance(statement.value, ast.Call)):
        outputs = _assignment_names(statement.target)
        if outputs is not None:
            return statement.value, outputs
    return None, []


def _closed_call(call: ast.Call, known_names: set[str]) -> bool:
    """Return whether a call can be rebound without capturing a local frame."""
    target = _attribute_name(call.func)
    if not target or not target.startswith("self."):
        return False
    for node in ast.walk(call):
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp,
                             ast.DictComp, ast.GeneratorExp,
                             ast.Await, ast.Yield, ast.YieldFrom,
                             ast.NamedExpr)):
            return False
        if (isinstance(node, ast.Name) and node.id != "self"
                and isinstance(node.ctx, ast.Load)
                and node.id not in known_names):
            return False
        if isinstance(node, ast.Call) and node is not call:
            return False
    return True


@dataclass
class _Region:
    index: int
    statements: list[ast.stmt]
    executable: bool
    action_name: str
    reads: set[str] = field(default_factory=set)
    writes: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    available_names: set[str] = field(default_factory=set)
    estimated_cost_s: float = 0.0

    @property
    def lineno(self) -> int:
        return int(getattr(self.statements[0], "lineno", 0))

    @property
    def end_lineno(self) -> int:
        return int(getattr(self.statements[-1], "end_lineno", self.lineno))


class StartupActionExtractor:
    """Generate conservative candidate stages and explicit review artifacts."""

    def __init__(
        self,
        source_path: Path,
        discovery_path: Path,
        *,
        class_name: str = "EngineCore",
        method_name: str = "_initialize_engine_core",
    ) -> None:
        self.source_path = Path(source_path)
        self.discovery_path = Path(discovery_path)
        self.class_name = class_name
        self.method_name = method_name
        self.source = self.source_path.read_text(encoding="utf-8")
        self.discovery = _json(self.discovery_path)

    def _method(self) -> ast.FunctionDef | ast.AsyncFunctionDef:
        tree = ast.parse(self.source, filename=str(self.source_path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == self.class_name:
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and item.name == self.method_name):
                        return item
        raise ValueError(
            f"Cannot find {self.class_name}.{self.method_name} in {self.source_path}")

    def _input_names(self, method: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        arguments = [*method.args.posonlyargs, *method.args.args,
                     *method.args.kwonlyargs]
        result = {argument.arg for argument in arguments if argument.arg != "self"}
        if method.args.vararg:
            result.add(method.args.vararg.arg)
        if method.args.kwarg:
            result.add(method.args.kwarg.arg)
        return result

    def _cost(self, calls: Iterable[str]) -> float:
        suffixes = {item.rsplit(".", 1)[-1] for item in calls}
        candidates = []
        for item in [*self.discovery.get("call_boundaries", []),
                     *self.discovery.get("sampled_call_boundaries", [])]:
            qualname = str(item.get("qualname", "")).rsplit(".", 1)[-1]
            if qualname in suffixes:
                candidates.append(float(item.get(
                    "median_s", item.get("sampled_inclusive_s", 0.0)) or 0.0))
        return max(candidates, default=0.0)

    def _regions(self) -> tuple[list[_Region], set[str]]:
        method = self._method()
        input_names = self._input_names(method)
        known_names = set(input_names)
        regions: list[_Region] = []
        opaque: list[ast.stmt] = []

        def append_opaque() -> None:
            if not opaque:
                return
            index = len(regions)
            region = _Region(
                index=index,
                statements=list(opaque),
                executable=False,
                action_name=f"opaque_region_{index:03d}",
                available_names=set(known_names),
            )
            regions.append(region)
            use_def = _UseDef()
            for statement in opaque:
                use_def.visit(statement)
            known_names.update(name for name in use_def.writes
                               if not name.startswith("self."))
            opaque.clear()

        for statement in method.body:
            if (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)):
                continue
            call, outputs = _simple_call(statement)
            if call is None or not _closed_call(call, known_names):
                opaque.append(statement)
                continue
            append_opaque()
            target = _attribute_name(call.func) or "call"
            action_name = _identifier(target.rsplit(".", 1)[-1])
            action_name = f"{action_name}_{getattr(statement, 'lineno', 0)}"
            regions.append(_Region(
                index=len(regions),
                statements=[statement],
                executable=True,
                action_name=action_name,
                outputs=outputs,
                available_names=set(known_names),
            ))
            known_names.update(outputs)
        append_opaque()

        for region in regions:
            use_def = _UseDef()
            for statement in region.statements:
                use_def.visit(statement)
            region.reads = use_def.reads
            region.writes = use_def.writes
            region.calls = use_def.calls
            region.estimated_cost_s = self._cost(region.calls)
        return regions, input_names

    def generate(self, output_root: Path) -> dict[str, Path]:
        output_root = Path(output_root)
        candidate_dir = output_root / "candidate"
        review_dir = output_root / "manual_review"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        review_dir.mkdir(parents=True, exist_ok=True)
        regions, input_names = self._regions()

        produced_by: dict[str, str] = {}
        nodes = []
        edges = []
        action_review = []
        alias_review = []
        previous: Optional[str] = None
        for region in regions:
            required = []
            for key in sorted(region.reads):
                if key in produced_by:
                    required.append(key)
                elif key in input_names:
                    required.append(f"input.{key}")
            produced = sorted(region.writes)
            dependencies = [previous] if previous else []
            for key in sorted(region.reads):
                producer = produced_by.get(key)
                if producer is None:
                    continue
                if producer not in dependencies:
                    dependencies.append(producer)
                edges.append({
                    "source": producer,
                    "target": region.action_name,
                    "kind": "static-use-def",
                    "state": key,
                    "confidence": 0.85,
                    "protected": True,
                })
            if previous:
                edges.append({
                    "source": previous,
                    "target": region.action_name,
                    "kind": "original-order",
                    "confidence": 0.5,
                    "protected": False,
                })
            for key in produced:
                if key in produced_by:
                    alias_review.append({
                        "id": f"multiple-writer:{key}:{region.action_name}",
                        "mandatory": True,
                        "status": "pending",
                        "state": key,
                        "previous_writer": produced_by[key],
                        "writer": region.action_name,
                    })
                produced_by[key] = region.action_name
            for key in sorted(region.reads & region.writes):
                alias_review.append({
                    "id": f"in-place-or-alias:{key}:{region.action_name}",
                    "mandatory": True,
                    "status": "pending",
                    "state": key,
                    "writer": region.action_name,
                    "reason": "state is both read and written in one source region",
                })
            source = "\n".join(
                ast.get_source_segment(self.source, statement) or ""
                for statement in region.statements)
            node = {
                "name": region.action_name,
                "action_id": region.action_name,
                "dependencies": list(dict.fromkeys(dependencies)),
                "required_state": sorted(set(required)),
                "produced_state": produced,
                "resources": {},
                "estimated_cost_s": region.estimated_cost_s,
                "confidence": 0.8 if region.executable else 0.2,
                "category": "source_extracted" if region.executable else "opaque",
                "enabled": True,
                "source": {
                    "path": str(self.source_path),
                    "lineno": region.lineno,
                    "end_lineno": region.end_lineno,
                    "text": source,
                },
                "executable": region.executable,
            }
            nodes.append(node)
            if not region.executable:
                action_review.append({
                    "id": f"action:{region.action_name}",
                    "mandatory": True,
                    "status": "pending",
                    "node": region.action_name,
                    "reason": "region cannot be extracted as a closed bound-method call",
                    "source": node["source"],
                })
            previous = region.action_name

        source_hash = hashlib.sha256(self.source.encode("utf-8")).hexdigest()
        compatibility = dict(self.discovery.get("compatibility", {}))
        compatibility["source_sha256"] = source_hash
        contract = {
            "schema": "igniteserve-startup-contract-v2",
            "status": "candidate",
            "engine": "vllm",
            "entrypoint": f"{self.class_name}.{self.method_name}",
            "compatibility": compatibility,
            "adapter": {"path": "startup_adapter.py", "sha256": None},
            "initial_state": sorted(f"input.{name}" for name in input_names),
            "resource_capacities": {},
            "validation": {
                "serial_equivalence": {"status": "pending"},
                "edge_perturbations": {"status": "pending"},
            },
            "variants": [{
                "name": "discovered-default",
                "match": {},
                "nodes": nodes,
                "edges": edges,
            }],
            "source_discovery": str(self.discovery_path),
        }
        contract_path = candidate_dir / "startup_contract.candidate.v0.json"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True),
                                 encoding="utf-8")
        adapter_path = candidate_dir / "startup_adapter.candidate.py"
        adapter_path.write_text(self._adapter_source(regions, input_names),
                                encoding="utf-8")
        stage_map_path = candidate_dir / "candidate_stage_map.json"
        stage_map_path.write_text(json.dumps({
            "schema": "igniteserve-line-stage-map-v1",
            "source_path": str(self.source_path.resolve()),
            "function": f"{self.class_name}.{self.method_name}",
            "regions": [{
                "stage": region.action_name,
                "lineno": region.lineno,
                "end_lineno": region.end_lineno,
            } for region in regions],
        }, indent=2, sort_keys=True), encoding="utf-8")
        action_path = review_dir / "action_bindings.json"
        action_path.write_text(json.dumps({
            "schema": "igniteserve-manual-review-v1",
            "kind": "action_bindings",
            "items": action_review,
        }, indent=2, sort_keys=True), encoding="utf-8")
        alias_path = review_dir / "state_aliases.json"
        alias_path.write_text(json.dumps({
            "schema": "igniteserve-manual-review-v1",
            "kind": "state_aliases",
            "items": alias_review,
        }, indent=2, sort_keys=True), encoding="utf-8")
        for name, kind in (
            ("opaque_constraints.json", "opaque_constraints"),
            ("failed_edge_tests.json", "failed_edge_tests"),
            ("node_coarsening.json", "node_coarsening"),
        ):
            (review_dir / name).write_text(json.dumps({
                "schema": "igniteserve-manual-review-v1",
                "kind": kind,
                "items": [],
            }, indent=2, sort_keys=True), encoding="utf-8")
        (review_dir / "README.md").write_text(
            "# Manual review gate\n\n"
            "1. Resolve every pending item in `action_bindings.json` by "
            "implementing the corresponding adapter stub and setting "
            "`status` to `resolved`.\n"
            "2. Decide alias/in-place-write items in `state_aliases.json`; "
            "add explicit state keys or dependencies where needed.\n"
            "3. Confirm allocator, opaque C++/CUDA, process, and collective "
            "constraints in `opaque_constraints.json`.\n"
            "4. After serial equivalence passes, run isolated edge tests. "
            "Review every inconclusive result in `failed_edge_tests.json`.\n"
            "5. `node_coarsening.json` is advisory: accepting a merge requires "
            "editing both the contract and adapter; rejection is safe.\n\n"
            "`finalize` emits no runnable contract while any mandatory item "
            "is pending or any enabled adapter stub remains.\n",
            encoding="utf-8")
        return {
            "contract": contract_path,
            "adapter": adapter_path,
            "stage_map": stage_map_path,
            "review": review_dir,
        }

    def _adapter_source(self, regions: list[_Region],
                        input_names: set[str]) -> str:
        lines = [
            "# Generated by IgniteServe startup offline analysis.",
            "from __future__ import annotations",
            "",
            "class UnresolvedStartupAction(RuntimeError):",
            "    pass",
            "",
        ]
        for region in regions:
            function = f"_action_{_identifier(region.action_name)}"
            lines.append(f"def {function}(engine_core, context):")
            if not region.executable:
                lines.append(
                    f"    raise UnresolvedStartupAction({region.action_name!r})")
            else:
                statement = region.statements[0]
                transformed = _ActionTransformer(
                    region.available_names - {"self"}, input_names).visit(
                        ast.fix_missing_locations(ast.parse(
                            ast.get_source_segment(self.source, statement)
                            or "pass")))
                transformed = ast.fix_missing_locations(transformed)
                if region.outputs:
                    call_statement = transformed.body[0]
                    assert isinstance(call_statement, (ast.Assign, ast.AnnAssign))
                    value = ast.unparse(call_statement.value)
                    lines.append(f"    value = {value}")
                    if len(region.outputs) == 1:
                        lines.append(
                            f"    context.put({region.outputs[0]!r}, value)")
                    else:
                        for index, output in enumerate(region.outputs):
                            lines.append(
                                f"    context.put({output!r}, value[{index}])")
                    lines.append("    return value")
                else:
                    code = ast.unparse(transformed).strip()
                    for source_line in code.splitlines():
                        lines.append(f"    {source_line}")
            lines.append("")
        lines.extend([
            "def build_action_registry(engine_core):",
            "    return {",
        ])
        for region in regions:
            function = f"_action_{_identifier(region.action_name)}"
            lines.append(
                f"        {region.action_name!r}: "
                f"lambda context, fn={function}: fn(engine_core, context),")
        lines.extend(["    }", ""])
        return "\n".join(lines)
