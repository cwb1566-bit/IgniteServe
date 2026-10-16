# SPDX-License-Identifier: Apache-2.0
"""AST-based stage instrumentation for an otherwise sequential method."""

from __future__ import annotations

import ast
import copy
import inspect
import json
from pathlib import Path
from types import MethodType
from typing import Any, Callable


class SourceInstrumentationError(RuntimeError):
    pass


def _find_method(tree: ast.Module, class_name: str,
                 method_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and item.name == method_name):
                    return item
    raise SourceInstrumentationError(
        f"cannot find {class_name}.{method_name} in source AST")


def build_instrumented_bound_method(
    bound_method: Callable[..., Any],
    stage_map_path: Path,
    recorder: Any,
) -> Callable[..., Any]:
    """Wrap mapped top-level source regions without globally tracing Python."""
    stage_map_path = Path(stage_map_path)
    payload = json.loads(stage_map_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "igniteserve-line-stage-map-v1":
        raise SourceInstrumentationError("unsupported stage-map schema")
    function_name = str(payload.get("function", ""))
    if "." not in function_name:
        raise SourceInstrumentationError("stage-map function is not qualified")
    class_name, method_name = function_name.rsplit(".", 1)
    original = getattr(bound_method, "__func__", bound_method)
    if original.__name__ != method_name:
        raise SourceInstrumentationError(
            f"stage map targets {method_name}, got {original.__name__}")
    source_path = Path(payload["source_path"]).resolve()
    actual_path = Path(inspect.getsourcefile(original) or "").resolve()
    if source_path != actual_path:
        raise SourceInstrumentationError(
            f"source mismatch: stage-map={source_path} runtime={actual_path}")
    source = source_path.read_text(encoding="utf-8")
    method = copy.deepcopy(_find_method(
        ast.parse(source, filename=str(source_path)), class_name, method_name))
    method.decorator_list = []

    regions = []
    for item in payload.get("regions", []):
        regions.append((int(item["lineno"]), int(item["end_lineno"]),
                        str(item["stage"])))
    if not regions:
        raise SourceInstrumentationError("stage map has no source regions")

    def stage_for(statement: ast.stmt) -> str | None:
        start = int(getattr(statement, "lineno", 0))
        end = int(getattr(statement, "end_lineno", start))
        matches = [name for region_start, region_end, name in regions
                   if region_start <= start and end <= region_end]
        if len(matches) > 1:
            raise SourceInstrumentationError(
                f"statement lines {start}-{end} match multiple regions")
        return matches[0] if matches else None

    body: list[ast.stmt] = []
    active_name = None
    active_statements: list[ast.stmt] = []

    def flush() -> None:
        nonlocal active_name, active_statements
        if not active_statements:
            return
        if active_name is None:
            body.extend(active_statements)
        else:
            scope = ast.With(
                items=[ast.withitem(
                    context_expr=ast.Call(
                        func=ast.Name(id="_igniteserve_candidate_stage",
                                      ctx=ast.Load()),
                        args=[ast.Constant(active_name)], keywords=[]),
                    optional_vars=None)],
                body=active_statements,
                type_comment=None,
            )
            ast.copy_location(scope, active_statements[0])
            body.append(scope)
        active_name = None
        active_statements = []

    for statement in method.body:
        name = stage_for(statement)
        if active_statements and name != active_name:
            flush()
        active_name = name
        active_statements.append(statement)
    flush()
    method.body = body
    module = ast.fix_missing_locations(ast.Module(
        body=[method], type_ignores=[]))
    runtime_globals = dict(original.__globals__)
    runtime_globals["_igniteserve_candidate_stage"] = recorder.stage
    namespace: dict[str, Any] = {}
    exec(compile(module, str(source_path), "exec"), runtime_globals, namespace)
    generated = namespace[method_name]
    generated.__defaults__ = original.__defaults__
    generated.__kwdefaults__ = original.__kwdefaults__
    generated.__qualname__ = original.__qualname__
    return MethodType(generated, bound_method.__self__)
