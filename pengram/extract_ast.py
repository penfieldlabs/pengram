# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Deterministic code extraction via tree-sitter.

Produces ``{"nodes": [...], "edges": [...]}`` where edges use the
:mod:`pengram.vocabulary` structural relationship types (``calls``,
``imports``, ``extends``, ``implements_interface``, ``decorates``,
``overrides``). Every edge is tagged ``confidence="EXTRACTED"``.

Languages supported:

Python, JavaScript, TypeScript (+ JSX/TSX), Go, Rust, Java, C, C++, Ruby,
C#, Kotlin, Scala, PHP, Swift, Lua, Zig, PowerShell, Elixir, Objective-C,
Julia, Dart, Verilog/SystemVerilog, Vue, Svelte.

Grammars are loaded lazily via ``tree_sitter_languages``. Files whose
grammar is unavailable are skipped — the pipeline logs and continues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._ui import warn as _ui_warn
from .vocabulary import CONFIDENCE_EXTRACTED

_TREE_SITTER_HINT = (
    "tree-sitter and tree-sitter-language-pack are required for AST extraction. "
    "Install with: pip install 'pengram[code]'"
)


# ---------------------------------------------------------------------------
# ID normalization
# ---------------------------------------------------------------------------

_ID_SAFE_RE = re.compile(r"[^a-zA-Z0-9]+")


def make_id(*parts: str) -> str:
    """Return a stable, filesystem-safe node id."""
    joined = "_".join(p for p in parts if p)
    cleaned = _ID_SAFE_RE.sub("_", joined).strip("_").lower()
    return cleaned


# ---------------------------------------------------------------------------
# Per-language configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageConfig:
    """Minimal per-language node-type vocabulary.

    Tree-sitter grammars use similar node-type names across most mainstream
    languages (``function_definition``, ``class_declaration``, ``call``,
    etc.). The sets below let one generic walker handle all of them.
    """

    name: str  # grammar name as recognised by tree_sitter_languages
    class_types: frozenset[str] = field(default_factory=frozenset)
    function_types: frozenset[str] = field(default_factory=frozenset)
    import_types: frozenset[str] = field(default_factory=frozenset)
    call_types: frozenset[str] = field(default_factory=frozenset)
    extends_fields: tuple[str, ...] = ()
    implements_fields: tuple[str, ...] = ()
    decorator_types: frozenset[str] = field(default_factory=frozenset)
    name_field: str = "name"


# NOTE: sets are kept deliberately broad — different grammar versions rename
# node types (e.g. "function_declaration" vs "function_definition") and we
# want both to work.

_PYTHON = LanguageConfig(
    name="python",
    class_types=frozenset({"class_definition"}),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"import_statement", "import_from_statement"}),
    call_types=frozenset({"call"}),
    extends_fields=("superclasses",),
    decorator_types=frozenset({"decorator"}),
)

_JS = LanguageConfig(
    name="javascript",
    class_types=frozenset({"class_declaration", "class"}),
    function_types=frozenset(
        {
            "function_declaration",
            "method_definition",
            "arrow_function",
            "function",
            "generator_function_declaration",
        }
    ),
    import_types=frozenset({"import_statement", "call_expression"}),
    call_types=frozenset({"call_expression"}),
    extends_fields=("superclass",),
)

_TS = LanguageConfig(
    name="typescript",
    class_types=frozenset({"class_declaration"}),
    function_types=frozenset(
        {
            "function_declaration",
            "method_definition",
            "arrow_function",
            "function_signature",
        }
    ),
    import_types=frozenset({"import_statement"}),
    call_types=frozenset({"call_expression"}),
    extends_fields=("superclass",),
    implements_fields=("implements",),
)

_TSX = LanguageConfig(
    name="tsx",
    class_types=_TS.class_types,
    function_types=_TS.function_types,
    import_types=_TS.import_types,
    call_types=_TS.call_types,
    extends_fields=_TS.extends_fields,
    implements_fields=_TS.implements_fields,
)

_GO = LanguageConfig(
    name="go",
    class_types=frozenset({"type_declaration"}),
    function_types=frozenset({"function_declaration", "method_declaration"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"call_expression"}),
)

_RUST = LanguageConfig(
    name="rust",
    class_types=frozenset({"struct_item", "enum_item", "trait_item"}),
    function_types=frozenset({"function_item"}),
    import_types=frozenset({"use_declaration"}),
    call_types=frozenset({"call_expression", "macro_invocation"}),
)

_JAVA = LanguageConfig(
    name="java",
    class_types=frozenset({"class_declaration", "interface_declaration", "enum_declaration"}),
    function_types=frozenset({"method_declaration", "constructor_declaration"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"method_invocation"}),
    extends_fields=("superclass",),
    implements_fields=("interfaces",),
)

_C = LanguageConfig(
    name="c",
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"preproc_include"}),
    call_types=frozenset({"call_expression"}),
)

_CPP = LanguageConfig(
    name="cpp",
    class_types=frozenset({"class_specifier", "struct_specifier"}),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"preproc_include"}),
    call_types=frozenset({"call_expression"}),
)

_RUBY = LanguageConfig(
    name="ruby",
    class_types=frozenset({"class", "module"}),
    function_types=frozenset({"method", "singleton_method"}),
    import_types=frozenset({"call"}),  # require/require_relative are method calls
    call_types=frozenset({"call", "method_call"}),
    extends_fields=("superclass",),
)

_CSHARP = LanguageConfig(
    name="csharp",
    class_types=frozenset({"class_declaration", "interface_declaration", "record_declaration"}),
    function_types=frozenset({"method_declaration", "constructor_declaration"}),
    import_types=frozenset({"using_directive"}),
    call_types=frozenset({"invocation_expression"}),
    extends_fields=("bases",),
)

_KOTLIN = LanguageConfig(
    name="kotlin",
    class_types=frozenset({"class_declaration", "object_declaration"}),
    function_types=frozenset({"function_declaration"}),
    import_types=frozenset({"import_header"}),
    call_types=frozenset({"call_expression"}),
)

_SCALA = LanguageConfig(
    name="scala",
    class_types=frozenset({"class_definition", "object_definition", "trait_definition"}),
    function_types=frozenset({"function_definition", "function_declaration"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"call_expression"}),
)

_PHP = LanguageConfig(
    name="php",
    class_types=frozenset({"class_declaration", "interface_declaration", "trait_declaration"}),
    function_types=frozenset({"function_definition", "method_declaration"}),
    import_types=frozenset({"namespace_use_declaration"}),
    call_types=frozenset({"function_call_expression", "member_call_expression"}),
)

_SWIFT = LanguageConfig(
    name="swift",
    class_types=frozenset({"class_declaration", "protocol_declaration", "struct_declaration"}),
    function_types=frozenset({"function_declaration", "init_declaration"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"call_expression"}),
)

_LUA = LanguageConfig(
    name="lua",
    function_types=frozenset({"function_declaration", "function_definition"}),
    import_types=frozenset({"function_call"}),  # require("x")
    call_types=frozenset({"function_call"}),
)

_ZIG = LanguageConfig(
    name="zig",
    class_types=frozenset({"struct_declaration"}),
    function_types=frozenset({"fn_proto", "function_definition"}),
    import_types=frozenset({"builtin_call_expression"}),
    call_types=frozenset({"call_expression"}),
)

_POWERSHELL = LanguageConfig(
    name="powershell",
    function_types=frozenset({"function_statement"}),
    call_types=frozenset({"command"}),
)

_ELIXIR = LanguageConfig(
    name="elixir",
    class_types=frozenset({"call"}),  # defmodule is a call
    function_types=frozenset({"call"}),  # def is a call
    call_types=frozenset({"call"}),
)

_OBJC = LanguageConfig(
    name="objc",
    class_types=frozenset({"class_interface", "class_implementation"}),
    function_types=frozenset({"method_definition"}),
    import_types=frozenset({"preproc_include", "module_import"}),
    call_types=frozenset({"message_expression", "call_expression"}),
)

_JULIA = LanguageConfig(
    name="julia",
    class_types=frozenset({"struct_definition", "abstract_definition"}),
    function_types=frozenset({"function_definition", "short_function_definition"}),
    import_types=frozenset({"import_statement", "using_statement"}),
    call_types=frozenset({"call_expression"}),
)

_DART = LanguageConfig(
    name="dart",
    class_types=frozenset({"class_definition", "mixin_declaration", "enum_declaration"}),
    function_types=frozenset({"function_signature", "method_signature"}),
    import_types=frozenset({"import_specification"}),
    call_types=frozenset({"function_expression_invocation"}),
)

_VERILOG = LanguageConfig(
    name="verilog",
    class_types=frozenset({"module_declaration", "class_declaration"}),
    function_types=frozenset({"function_declaration", "task_declaration"}),
    call_types=frozenset({"system_tf_call", "subroutine_call"}),
)

# Extension → config.
_EXT_CONFIG: dict[str, LanguageConfig] = {
    ".py": _PYTHON,
    ".js": _JS,
    ".jsx": _JS,
    ".mjs": _JS,
    ".cjs": _JS,
    ".ts": _TS,
    ".tsx": _TSX,
    ".go": _GO,
    ".rs": _RUST,
    ".java": _JAVA,
    ".c": _C,
    ".h": _C,
    ".cpp": _CPP,
    ".cc": _CPP,
    ".cxx": _CPP,
    ".hpp": _CPP,
    ".hh": _CPP,
    ".rb": _RUBY,
    ".cs": _CSHARP,
    ".kt": _KOTLIN,
    ".kts": _KOTLIN,
    ".scala": _SCALA,
    ".php": _PHP,
    ".swift": _SWIFT,
    ".lua": _LUA,
    ".zig": _ZIG,
    ".ps1": _POWERSHELL,
    ".ex": _ELIXIR,
    ".exs": _ELIXIR,
    ".m": _OBJC,
    ".mm": _OBJC,
    ".jl": _JULIA,
    ".dart": _DART,
    ".v": _VERILOG,
    ".sv": _VERILOG,
    ".vue": _JS,
    ".svelte": _JS,
}


SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_EXT_CONFIG.keys())


# ---------------------------------------------------------------------------
# Grammar loading
# ---------------------------------------------------------------------------


def _require_tree_sitter():
    try:
        from tree_sitter_language_pack import get_parser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_TREE_SITTER_HINT) from exc
    return get_parser


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name_of(node: Any, source: bytes, field_name: str = "name") -> str | None:
    try:
        n = node.child_by_field_name(field_name)
    except Exception:
        n = None
    if n is not None:
        return _text(n, source)
    # Fallback: first identifier child.
    for child in node.children:
        if child.type == "identifier":
            return _text(child, source)
    return None


def _import_targets(node: Any, source: bytes, config: LanguageConfig) -> list[str]:
    """Extract module names from an import-like node, language-agnostically."""
    # Strategy: collect all string literals and dotted identifiers inside the node.
    targets: list[str] = []
    for child in node.children:
        if child.type in {"string", "string_literal"}:
            raw = _text(child, source).strip("'\"` ")
            if raw:
                targets.append(raw)
        elif child.type in {"dotted_name", "scoped_identifier", "identifier", "package_name"}:
            targets.append(_text(child, source))
    # For grammars where the interesting target is deeper (e.g. Ruby's require call),
    # recursively search one extra level.
    if not targets:
        for child in node.children:
            for gc in getattr(child, "children", []) or []:
                if gc.type in {"string", "string_literal"}:
                    raw = _text(gc, source).strip("'\"` ")
                    if raw:
                        targets.append(raw)
    return targets


def _walk(root: Any, visitor) -> None:
    """Depth-first traversal of a tree-sitter tree."""
    stack = [root]
    while stack:
        node = stack.pop()
        visitor(node)
        # Push children in reverse so traversal is left-to-right.
        children = list(getattr(node, "children", []) or [])
        for child in reversed(children):
            stack.append(child)


def extract_code(path: Path) -> dict[str, list]:
    """Extract nodes and edges from a source file.

    Returns a dict with ``nodes`` and ``edges`` keys. Empty lists are
    returned for unsupported extensions or when tree-sitter is unavailable.
    """
    ext = path.suffix.lower()
    config = _EXT_CONFIG.get(ext)
    if config is None:
        return {"nodes": [], "edges": []}
    try:
        get_parser = _require_tree_sitter()
    except ImportError:
        raise
    try:
        parser = get_parser(config.name)
    except Exception as exc:
        _ui_warn(f"No tree-sitter grammar available for {ext} ({config.name}): {exc}")
        return {"nodes": [], "edges": []}

    try:
        source = path.read_bytes()
    except OSError as exc:
        _ui_warn(f"Could not read {path}: {exc}")
        return {"nodes": [], "edges": []}

    try:
        tree = parser.parse(source)
    except Exception as exc:
        _ui_warn(f"tree-sitter parse failed for {path}: {exc}")
        return {"nodes": [], "edges": []}

    str_path = str(path)
    file_id = make_id(str_path)
    nodes: list[dict[str, Any]] = [
        {
            "id": file_id,
            "label": path.name,
            "kind": "file",
            "source_file": str_path,
            "language": config.name,
            "confidence": CONFIDENCE_EXTRACTED,
        }
    ]
    edges: list[dict[str, Any]] = []

    def _add_edge(source: str, target: str, relation: str, node) -> None:
        edges.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": CONFIDENCE_EXTRACTED,
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
            }
        )

    def _handle(node) -> None:
        t = node.type
        if t in config.import_types:
            for target in _import_targets(node, source, config):
                module_name = target.split("/")[-1].split(".")[-1]
                if not module_name:
                    continue
                target_id = make_id(module_name)
                _add_edge(file_id, target_id, "imports", node)
            return
        if t in config.class_types:
            name = _name_of(node, source, config.name_field)
            if not name:
                return
            class_id = make_id(str_path, name)
            nodes.append(
                {
                    "id": class_id,
                    "label": name,
                    "kind": "class",
                    "source_file": str_path,
                    "language": config.name,
                    "confidence": CONFIDENCE_EXTRACTED,
                }
            )
            # Attach the class back to its containing file via `uses` so the
            # relation stays within STRUCTURAL_TYPES.
            _add_edge(file_id, class_id, "uses", node)
            for field_name in config.extends_fields:
                try:
                    parent_node = node.child_by_field_name(field_name)
                except Exception:
                    parent_node = None
                if parent_node is not None:
                    parent_name = _text(parent_node, source).strip()
                    if parent_name:
                        _add_edge(class_id, make_id(parent_name), "extends", node)
            for field_name in config.implements_fields:
                try:
                    impl_node = node.child_by_field_name(field_name)
                except Exception:
                    impl_node = None
                if impl_node is not None:
                    impl_name = _text(impl_node, source).strip()
                    if impl_name:
                        _add_edge(class_id, make_id(impl_name), "implements_interface", node)
            return
        if t in config.function_types:
            name = _name_of(node, source, config.name_field)
            if not name:
                return
            fn_id = make_id(str_path, name)
            nodes.append(
                {
                    "id": fn_id,
                    "label": f"{name}()",
                    "kind": "function",
                    "source_file": str_path,
                    "language": config.name,
                    "confidence": CONFIDENCE_EXTRACTED,
                }
            )
            _add_edge(file_id, fn_id, "uses", node)
            return
        if t in config.decorator_types:
            deco_name = _name_of(node, source) or _text(node, source).strip("@ \n")
            if deco_name:
                _add_edge(file_id, make_id(deco_name), "decorates", node)
            return
        if t in config.call_types:
            try:
                callee_node = node.child_by_field_name("function")
            except Exception:
                callee_node = None
            callee = _text(callee_node, source) if callee_node is not None else None
            if callee:
                # Reduce e.g. ``foo.bar.baz`` → ``baz`` for call targets.
                short = callee.split(".")[-1].strip("()")
                if short:
                    _add_edge(file_id, make_id(short), "calls", node)
            return

    _walk(tree.root_node, _handle)
    return {"nodes": nodes, "edges": edges}


__all__ = [
    "LanguageConfig",
    "SUPPORTED_EXTENSIONS",
    "extract_code",
    "make_id",
]
