# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Tests for pengram.export_html."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx

from pengram.export_html import export_html, render_html


def _make_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("a", label="Alice", kind="concept")
    g.add_node("b", label="Bob", kind="concept")
    g.add_edge("a", "b", relation="references", confidence="EXTRACTED")
    return g


def test_render_html_contains_vis_cdn() -> None:
    html = render_html(_make_graph(), {0: ["a", "b"]})
    assert "vis-network" in html


def test_render_html_valid_doctype() -> None:
    html = render_html(_make_graph())
    assert html.startswith("<!DOCTYPE html>")
    assert "<html" in html


def test_render_html_embeds_data() -> None:
    g = _make_graph()
    html = render_html(g, {0: ["a", "b"]})
    # Extract the DATA constant.
    match = re.search(r"const DATA = (\{.*?\});", html, flags=re.DOTALL)
    assert match is not None
    data = json.loads(match.group(1))
    ids = {n["id"] for n in data["nodes"]}
    assert ids == {"a", "b"}
    assert data["edges"][0]["from"] == "a"
    assert data["edges"][0]["to"] == "b"


def test_render_html_legend_contains_communities() -> None:
    g = _make_graph()
    html = render_html(g, {0: ["a"], 1: ["b"]})
    match = re.search(r"const DATA = (\{.*?\});", html, flags=re.DOTALL)
    data = json.loads(match.group(1))
    legend_cids = {entry["cid"] for entry in data["legend"]}
    assert legend_cids == {0, 1}


def test_export_html_writes_file(tmp_path: Path) -> None:
    out = export_html(_make_graph(), tmp_path, communities={0: ["a", "b"]})
    assert out.exists()
    assert out.name == "graph.html"
    content = out.read_text()
    assert "<!DOCTYPE html>" in content


def test_render_html_escapes_title() -> None:
    html = render_html(_make_graph(), title="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_render_html_uses_viewport_height_not_percentage() -> None:
    """Regression for Bug 6: percentage-based #graph sizing in a CSS grid
    caused a 0×0 canvas. Template must use a viewport-based height."""
    html = render_html(_make_graph())
    assert "#graph { height: 100vh;" in html
    # The previous broken rule must no longer be present.
    assert "#graph { width: 100%; height: 100%;" not in html


def test_render_html_pins_vis_version() -> None:
    """Regression for Bug 6: unpinned vis.js CDN could break on major bumps."""
    html = render_html(_make_graph())
    assert "vis-network@" in html  # any explicit version, not the unpinned alias


def test_render_html_waits_for_dom() -> None:
    """Regression for Bug 6: init must wait for DOMContentLoaded so the
    grid container has real dimensions before vis.js reads them."""
    html = render_html(_make_graph())
    assert "DOMContentLoaded" in html


def test_render_html_has_error_handler() -> None:
    """Regression for Bug 6: init failures must surface in the DOM, not just console."""
    html = render_html(_make_graph())
    assert "showError" in html
    assert "vis.js failed to load" in html
