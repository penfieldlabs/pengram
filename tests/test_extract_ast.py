# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.extract_ast."""

from __future__ import annotations

from pathlib import Path

import pytest

from pengram import extract_ast
from pengram.vocabulary import STRUCTURAL_TYPES

# tree-sitter-language-pack is a dev/CI dependency (see pyproject [dev]) so
# the real-grammar integration tests below always run. There is intentionally
# no skip-guard — skipping hides coverage gaps. If you're running a minimal
# install and one of these fails to import, install the dev extra.


def test_make_id_stable() -> None:
    assert extract_ast.make_id("Foo", "Bar") == extract_ast.make_id("Foo", "Bar")


def test_make_id_normalizes() -> None:
    assert extract_ast.make_id("My Class/Name!") == "my_class_name"


def test_make_id_combines_parts() -> None:
    result = extract_ast.make_id("/path/to/file.py", "MyClass")
    assert "path" in result
    assert "myclass" in result


def test_unknown_extension_returns_empty(tmp_path: Path) -> None:
    f = tmp_path / "foo.unknown"
    f.write_text("whatever")
    assert extract_ast.extract_code(f) == {"nodes": [], "edges": []}


def test_python_extracts_classes_and_functions(tmp_path: Path) -> None:
    src = tmp_path / "hello.py"
    src.write_text(
        "import os\n"
        "\n"
        "class Greeter:\n"
        "    def say(self, name):\n"
        "        print(name)\n"
        "\n"
        "def top_level():\n"
        "    return 1\n"
        "\n"
        "top_level()\n"
    )
    result = extract_ast.extract_code(src)
    labels = {n["label"] for n in result["nodes"]}
    assert "Greeter" in labels
    assert "say()" in labels
    assert "top_level()" in labels
    assert "hello.py" in labels


def test_python_import_edges(tmp_path: Path) -> None:
    src = tmp_path / "hello.py"
    src.write_text("import os\nimport sys\n")
    result = extract_ast.extract_code(src)
    import_edges = [e for e in result["edges"] if e["relation"] == "imports"]
    assert import_edges, "expected at least one imports edge"
    for edge in import_edges:
        assert edge["confidence"] == "EXTRACTED"


def test_python_call_edges(tmp_path: Path) -> None:
    src = tmp_path / "hello.py"
    src.write_text("def foo(): pass\nfoo()\n")
    result = extract_ast.extract_code(src)
    call_edges = [e for e in result["edges"] if e["relation"] == "calls"]
    assert call_edges


def test_js_extracts_functions(tmp_path: Path) -> None:
    src = tmp_path / "hello.js"
    src.write_text("function greet(name) { return name; }\nclass Foo { bar() {} }\n")
    result = extract_ast.extract_code(src)
    labels = {n["label"] for n in result["nodes"]}
    assert "Foo" in labels
    assert any("greet" in label for label in labels)


def test_all_edges_use_structural_vocabulary(tmp_path: Path) -> None:
    src = tmp_path / "x.py"
    src.write_text("import os\nclass A:\n    pass\nclass B(A):\n    def m(self):\n        pass\n")
    result = extract_ast.extract_code(src)
    for edge in result["edges"]:
        assert edge["relation"] in STRUCTURAL_TYPES, (
            f"edge {edge['relation']} not in STRUCTURAL_TYPES"
        )


def test_node_ids_are_stable(tmp_path: Path) -> None:
    src = tmp_path / "x.py"
    src.write_text("def f(): pass\n")
    r1 = extract_ast.extract_code(src)
    r2 = extract_ast.extract_code(src)
    ids1 = sorted(n["id"] for n in r1["nodes"])
    ids2 = sorted(n["id"] for n in r2["nodes"])
    assert ids1 == ids2


def test_extends_edge_python(tmp_path: Path) -> None:
    src = tmp_path / "x.py"
    src.write_text("class A: pass\nclass B(A): pass\n")
    result = extract_ast.extract_code(src)
    extends = [e for e in result["edges"] if e["relation"] == "extends"]
    assert extends


def test_file_node_always_present(tmp_path: Path) -> None:
    src = tmp_path / "empty.py"
    src.write_text("")
    result = extract_ast.extract_code(src)
    file_nodes = [n for n in result["nodes"] if n.get("kind") == "file"]
    assert len(file_nodes) == 1


def test_missing_grammar_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # For languages whose grammar is incompatible (e.g. csharp version mismatch),
    # the extractor must degrade gracefully.
    src = tmp_path / "x.cs"
    src.write_text("class A {}")
    result = extract_ast.extract_code(src)
    # Either empty (grammar unavailable) or a parse with nodes — but never raise.
    assert "nodes" in result and "edges" in result


# ---------------------------------------------------------------------------
# Tests using a fake tree-sitter module so coverage is exercised even when
# the real grammars are not installed.
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(
        self,
        type_: str,
        text: str = "",
        children: list[_FakeNode] | None = None,
        fields: dict[str, _FakeNode] | None = None,
        start_line: int = 0,
    ) -> None:
        self.type = type_
        self._text = text.encode("utf-8")
        self.children = children or []
        self._fields = fields or {}
        self.start_point = (start_line, 0)
        # Start/end bytes are computed by the fake tree relative to the source.
        self.start_byte = 0
        self.end_byte = len(self._text)

    def child_by_field_name(self, name: str):
        return self._fields.get(name)


class _FakeTree:
    def __init__(self, root: _FakeNode) -> None:
        self.root_node = root


class _FakeParser:
    def __init__(self, root: _FakeNode) -> None:
        self._root = root

    def parse(self, source: bytes) -> _FakeTree:
        return _FakeTree(self._root)


def _install_fake_parser(monkeypatch: pytest.MonkeyPatch, root: _FakeNode) -> None:
    """Install a fake ``tree_sitter_language_pack`` that returns ``root``."""
    import sys
    import types

    fake_module = types.ModuleType("tree_sitter_language_pack")
    fake_module.get_parser = lambda name: _FakeParser(root)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake_module)


def _node_with_text(type_: str, text: str, source: bytes, start: int, **kwargs) -> _FakeNode:
    node = _FakeNode(type_, text, **kwargs)
    node.start_byte = start
    node.end_byte = start + len(text.encode("utf-8"))
    return node


def test_extract_code_with_fake_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build a tiny fake Python AST: one class node with a name child.
    source = b"class Foo: pass\n"
    name = _FakeNode("identifier", "Foo")
    name.start_byte = 6
    name.end_byte = 9
    class_node = _FakeNode(
        "class_definition",
        "",
        fields={"name": name},
        start_line=0,
    )
    class_node.start_byte = 0
    class_node.end_byte = len(source)
    root = _FakeNode("module", "", children=[class_node])

    src = tmp_path / "fake.py"
    src.write_bytes(source)
    _install_fake_parser(monkeypatch, root)

    result = extract_ast.extract_code(src)
    labels = {n["label"] for n in result["nodes"]}
    assert "Foo" in labels
    assert "fake.py" in labels


def test_extract_code_emits_call_edge_with_fake_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"foo()\n"
    fn_ident = _FakeNode("identifier", "foo")
    fn_ident.start_byte = 0
    fn_ident.end_byte = 3
    call = _FakeNode(
        "call",
        "",
        fields={"function": fn_ident},
    )
    call.start_byte = 0
    call.end_byte = len(source)
    root = _FakeNode("module", "", children=[call])

    src = tmp_path / "calls.py"
    src.write_bytes(source)
    _install_fake_parser(monkeypatch, root)

    result = extract_ast.extract_code(src)
    relations = {e["relation"] for e in result["edges"]}
    assert "calls" in relations


def test_extract_code_import_edge_with_fake_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"import os\n"
    module_name = _node_with_text("dotted_name", "os", source, start=7)
    imp = _FakeNode("import_statement", "", children=[module_name])
    imp.start_byte = 0
    imp.end_byte = len(source)
    root = _FakeNode("module", "", children=[imp])

    src = tmp_path / "imports.py"
    src.write_bytes(source)
    _install_fake_parser(monkeypatch, root)

    result = extract_ast.extract_code(src)
    imports = [e for e in result["edges"] if e["relation"] == "imports"]
    assert imports


def test_extract_code_function_node_with_fake_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"def foo(): pass\n"
    name = _node_with_text("identifier", "foo", source, start=4)
    fn = _FakeNode("function_definition", "", fields={"name": name})
    fn.start_byte = 0
    fn.end_byte = len(source)
    root = _FakeNode("module", "", children=[fn])

    src = tmp_path / "funcs.py"
    src.write_bytes(source)
    _install_fake_parser(monkeypatch, root)

    result = extract_ast.extract_code(src)
    fns = [n for n in result["nodes"] if n.get("kind") == "function"]
    assert fns and fns[0]["label"] == "foo()"


def test_extract_code_decorator_with_fake_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = b"@dec\n"
    ident = _node_with_text("identifier", "dec", source, start=1)
    deco = _FakeNode("decorator", "", fields={"name": ident})
    deco.start_byte = 0
    deco.end_byte = len(source)
    root = _FakeNode("module", "", children=[deco])

    src = tmp_path / "deco.py"
    src.write_bytes(source)
    _install_fake_parser(monkeypatch, root)

    result = extract_ast.extract_code(src)
    decorates = [e for e in result["edges"] if e["relation"] == "decorates"]
    assert decorates


def test_extract_code_import_error_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When tree-sitter is entirely missing, extract_code raises ImportError.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "tree_sitter_language_pack":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import sys

    monkeypatch.delitem(sys.modules, "tree_sitter_language_pack", raising=False)

    src = tmp_path / "x.py"
    src.write_text("pass\n")
    with pytest.raises(ImportError, match="tree-sitter"):
        extract_ast.extract_code(src)


def test_extract_code_parser_not_found_returns_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the grammar raises (e.g. version mismatch), result is empty, no raise."""
    import sys
    import types

    fake = types.ModuleType("tree_sitter_language_pack")

    def bad_get_parser(name: str):
        raise RuntimeError("grammar unavailable")

    fake.get_parser = bad_get_parser  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)

    src = tmp_path / "x.py"
    src.write_text("def f(): pass\n")
    result = extract_ast.extract_code(src)
    assert result == {"nodes": [], "edges": []}


def test_extract_code_parse_failure_returns_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    class _BoomParser:
        def parse(self, src: bytes):
            raise RuntimeError("boom")

    fake = types.ModuleType("tree_sitter_language_pack")
    fake.get_parser = lambda name: _BoomParser()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tree_sitter_language_pack", fake)

    src = tmp_path / "x.py"
    src.write_text("def f(): pass\n")
    result = extract_ast.extract_code(src)
    assert result == {"nodes": [], "edges": []}


def test_extract_code_unreadable_file_returns_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the source file can't be read, return empty rather than crashing."""
    root = _FakeNode("module", "", children=[])
    _install_fake_parser(monkeypatch, root)

    src = tmp_path / "vanish.py"
    src.write_text("pass\n")

    # Monkeypatch read_bytes to raise OSError.
    def bad_read(self):  # type: ignore[no-untyped-def]
        raise OSError("file vanished")

    monkeypatch.setattr(Path, "read_bytes", bad_read)
    result = extract_ast.extract_code(src)
    assert result == {"nodes": [], "edges": []}
