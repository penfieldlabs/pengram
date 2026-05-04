# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Penfield Labs
"""Self-contained interactive graph visualisation as a single HTML file.

Uses vis.js (loaded from CDN) for rendering. Node colors encode community
assignment; edge styles vary by relationship type. Sidebar has search,
selected-node detail, and community legend.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from .export_common import write_text_if_changed
from .vocabulary import SEMANTIC_TYPES, STRUCTURAL_TYPES

_VIS_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

# Stable-ish pastel palette. Covers up to 20 distinct communities before
# cycling.
_COMMUNITY_COLORS: list[str] = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
    "#86b5d9",
    "#f4a582",
    "#d6604d",
    "#92c5de",
    "#a1d76a",
    "#fde0ef",
    "#c2a5cf",
    "#fddbc7",
    "#d1e5f0",
    "#f7f7f7",
]

# Semantic types get warm colours; structural types cool colours.
_EDGE_COLORS: dict[str, str] = {
    "supports": "#2a9d8f",
    "contradicts": "#e76f51",
    "disputes": "#e76f51",
    "references": "#8e8e8e",
    "parent_of": "#264653",
    "child_of": "#264653",
    "part_of": "#264653",
    "composed_of": "#264653",
    "sibling_of": "#264653",
    "causes": "#e9c46a",
    "influenced_by": "#e9c46a",
    "prerequisite_for": "#e9c46a",
    "implements": "#4a7c59",
    "tests": "#4a7c59",
    "documents": "#4a7c59",
    "example_of": "#4a7c59",
    "responds_to": "#a259ff",
    "inspired_by": "#a259ff",
    "supersedes": "#b5179e",
    "updates": "#b5179e",
    "evolution_of": "#b5179e",
    "follows": "#808080",
    "precedes": "#808080",
    "depends_on": "#1d3557",
    "calls": "#1d3557",
    "imports": "#457b9d",
    "uses": "#a8dadc",
    "extends": "#588157",
    "implements_interface": "#588157",
    "instantiates": "#a3b18a",
    "overrides": "#344e41",
    "decorates": "#bc4749",
}


def _color_for_community(cid: int) -> str:
    return _COMMUNITY_COLORS[cid % len(_COMMUNITY_COLORS)]


def _edge_color(relation: str) -> str:
    return _EDGE_COLORS.get(relation, "#cccccc")


# Node shape per kind — helps readers skim the graph at a glance.
# Kinds not listed here fall back to ``dot`` in ``_build_payload``.
_KIND_SHAPE: dict[str, str] = {
    "concept": "dot",
    "category": "triangle",
    "document": "box",
    "transcript": "box",
    "file": "square",
    "class": "ellipse",
    "function": "dot",
    "code": "square",
}


def _build_payload(
    g: nx.Graph,
    communities: dict[int, list[str]] | None,
) -> dict[str, Any]:
    community_of: dict[str, int] = {}
    for cid, nodes in (communities or {}).items():
        for node in nodes:
            community_of[node] = cid
    # Node size scales with mentions (semantic nodes) or degree (code nodes).
    degree = dict(g.degree()) if len(g) else {}
    node_list: list[dict[str, Any]] = []
    for node_id in g.nodes:
        node = g.nodes[node_id]
        cid = community_of.get(node_id)
        kind = node.get("kind", "")
        mentions = int(node.get("mentions", 1) or 1)
        deg = int(degree.get(node_id, 1) or 1)
        size = 10 + min(25, max(mentions, deg) * 2)
        node_list.append(
            {
                "id": node_id,
                "label": node.get("label", node_id),
                "title": node.get("summary", node.get("label", node_id)),
                "group": cid,
                "color": _color_for_community(cid) if cid is not None else "#cccccc",
                "shape": _KIND_SHAPE.get(kind, "dot"),
                "size": size,
                "kind": kind,
                "mentions": mentions,
            }
        )
    edge_list: list[dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        relation = data.get("relation", "")
        edge_list.append(
            {
                "from": u,
                "to": v,
                "label": relation,  # hidden by default, shown on hover (see template)
                "relation": relation,
                "confidence": data.get("confidence", ""),
                "color": {"color": _edge_color(relation), "opacity": 0.7},
                "arrows": "to" if isinstance(g, nx.DiGraph) else "",
            }
        )
    legend = [
        {"cid": cid, "size": len(nodes), "color": _color_for_community(cid)}
        for cid, nodes in sorted((communities or {}).items())
    ]
    return {"nodes": node_list, "edges": edge_list, "legend": legend}


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__PENGRAM_TITLE__</title>
<script src="__PENGRAM_CDN__"></script>
<style>
  html, body { margin: 0; height: 100vh; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1f2933; background: #fafafa; }
  #app { display: grid; grid-template-columns: 320px 1fr; height: 100vh; }
  #sidebar { padding: 1rem 1.1rem; border-right: 1px solid #e4e7eb; overflow-y: auto; background: #ffffff; }
  #graph { height: 100vh; background: #fafafa; }
  #status { font-size: .8rem; color: #b44; min-height: 1.2em; margin-top: .25rem; }
  h1 { font-size: 1.1rem; margin: 0 0 .25rem; font-weight: 600; letter-spacing: -.01em; }
  h2 { font-size: .78rem; margin: 1.1rem 0 .35rem; color: #616e7c; text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }
  input[type="search"] { width: 100%; padding: .45rem .65rem; border: 1px solid #cbd2d9; border-radius: 6px; font-size: .9rem; box-sizing: border-box; }
  input[type="search"]:focus { outline: none; border-color: #4e79a7; box-shadow: 0 0 0 2px rgba(78, 121, 167, .15); }
  .legend { list-style: none; padding: 0; margin: 0; }
  .legend li { display: flex; align-items: center; gap: .5rem; margin: .3rem 0; font-size: .82rem; color: #3e4c59; }
  .swatch { width: 11px; height: 11px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
  pre { background: #f5f7fa; padding: .5rem .6rem; border-radius: 4px; font-size: .75rem; line-height: 1.4; white-space: pre-wrap; word-break: break-word; margin: .4rem 0; max-height: 10rem; overflow: auto; }
  #detail { font-size: .85rem; color: #3e4c59; }
  #detail strong { color: #1f2933; font-size: .95rem; }
  .relation { font-size: .72rem; padding: .05rem .35rem; border-radius: 3px; background: #e4e7eb; color: #323f4b; margin-right: .3rem; }
  .edge-row { padding: .15rem 0; border-bottom: 1px solid #f0f2f5; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <h1>__PENGRAM_TITLE__</h1>
    <div id="status"></div>
    <h2>Search</h2>
    <input type="search" id="q" placeholder="Find a node..." autocomplete="off" />
    <h2>Selected</h2>
    <div id="detail">Click a node to inspect it.</div>
    <h2>Communities</h2>
    <ul id="legend" class="legend"></ul>
  </aside>
  <div id="graph"></div>
</div>
<script>
  const DATA = __PENGRAM_DATA__;
  const statusEl = document.getElementById("status");
  const showError = (msg) => { statusEl.textContent = msg; console.error(msg); };

  const boot = () => {
    try {
      if (typeof vis === "undefined") {
        showError("vis.js failed to load from CDN. Check network / offline status.");
        return;
      }
      const container = document.getElementById("graph");
      if (!container) { showError("Graph container missing."); return; }
      const nodes = new vis.DataSet(DATA.nodes);
      const edges = new vis.DataSet(DATA.edges);
      const options = {
        interaction: { hover: true, tooltipDelay: 200, hideEdgesOnDrag: true },
        nodes: {
          shape: "dot", size: 14,
          font: { size: 12, color: "#1f2933", strokeWidth: 3, strokeColor: "#fafafa" },
          borderWidth: 1.5,
          shadow: { enabled: true, size: 6, x: 0, y: 2, color: "rgba(0,0,0,0.08)" }
        },
        edges: {
          width: 1, selectionWidth: 2,
          smooth: { type: "continuous", roundness: 0.4 },
          font: { size: 10, align: "middle", color: "#52606d", strokeWidth: 2, strokeColor: "#fafafa" },
          // Hide labels until hover so a 600-edge graph doesn't look like noise.
          labelHighlightBold: false,
          scaling: { label: { enabled: false } }
        },
        physics: {
          solver: "forceAtlas2Based",
          forceAtlas2Based: { gravitationalConstant: -35, springLength: 110, springConstant: 0.08, damping: 0.55 },
          stabilization: { enabled: true, iterations: 200, updateInterval: 25, fit: true },
          timestep: 0.4
        }
      };
      // Strip edge labels from the default view; restore on hover only.
      edges.forEach(e => { e._origLabel = e.label; e.label = ""; edges.update(e); });

      const network = new vis.Network(container, { nodes, edges }, options);
      network.once("stabilizationIterationsDone", () => { network.setOptions({ physics: false }); });

      network.on("hoverEdge", (params) => {
        const e = edges.get(params.edge);
        if (e && e._origLabel) edges.update({ id: e.id, label: e._origLabel });
      });
      network.on("blurEdge", (params) => {
        const e = edges.get(params.edge);
        if (e) edges.update({ id: e.id, label: "" });
      });

      const legendEl = document.getElementById("legend");
      for (const entry of DATA.legend) {
        const li = document.createElement("li");
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = entry.color;
        li.appendChild(swatch);
        li.appendChild(document.createTextNode(`Community ${entry.cid} (${entry.size})`));
        legendEl.appendChild(li);
      }

      const detailEl = document.getElementById("detail");
      const renderDetail = (id) => {
        const node = nodes.get(id);
        if (!node) { detailEl.textContent = "Click a node to inspect it."; return; }
        const related = edges.get({ filter: (e) => e.from === id || e.to === id });
        const wrap = document.createElement("div");
        const strong = document.createElement("strong");
        strong.textContent = node.label;
        wrap.appendChild(strong);
        if (node.kind) {
          const small = document.createElement("div");
          small.style.color = "#7b8794";
          small.style.fontSize = ".78rem";
          small.textContent = node.kind;
          wrap.appendChild(small);
        }
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(node, null, 2);
        wrap.appendChild(pre);
        for (const e of related) {
          const row = document.createElement("div");
          row.className = "edge-row";
          const tag = document.createElement("span");
          tag.className = "relation";
          tag.textContent = e._origLabel || e.relation || "";
          row.appendChild(tag);
          row.appendChild(document.createTextNode(
            (e.from === id ? "→ " : "← ") + (e.from === id ? e.to : e.from)
          ));
          wrap.appendChild(row);
        }
        detailEl.innerHTML = "";
        detailEl.appendChild(wrap);
      };

      network.on("selectNode", (params) => { if (params.nodes[0]) renderDetail(params.nodes[0]); });
      network.on("deselectNode", () => { detailEl.textContent = "Click a node to inspect it."; });

      document.getElementById("q").addEventListener("input", (e) => {
        const q = e.target.value.trim().toLowerCase();
        if (!q) { network.unselectAll(); detailEl.textContent = "Click a node to inspect it."; return; }
        const match = nodes.get().find(n => (n.label || "").toLowerCase().includes(q));
        if (match) {
          network.selectNodes([match.id]);
          network.focus(match.id, { scale: 1.4, animation: { duration: 400, easingFunction: "easeInOutQuad" } });
          renderDetail(match.id);
        }
      });
    } catch (err) {
      showError("Graph init failed: " + (err && err.message ? err.message : err));
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
</script>
</body>
</html>
"""


def render_html(
    g: nx.Graph,
    communities: dict[int, list[str]] | None = None,
    *,
    title: str = "PENgram",
) -> str:
    """Return a self-contained HTML document visualising ``g``.

    Uses sentinel placeholders (``__PENGRAM_TITLE__`` etc.) and a single
    regex pass so user-supplied values (``title``, the serialized
    ``data`` JSON) can never be re-interpreted as other placeholders.
    """
    payload = _build_payload(g, communities)
    substitutions = {
        "__PENGRAM_TITLE__": html.escape(title),
        "__PENGRAM_CDN__": _VIS_CDN,
        "__PENGRAM_DATA__": json.dumps(payload),
    }
    pattern = re.compile("|".join(re.escape(k) for k in substitutions))
    return pattern.sub(lambda m: substitutions[m.group(0)], _HTML_TEMPLATE)


def export_html(
    g: nx.Graph,
    output_dir: Path,
    *,
    communities: dict[int, list[str]] | None = None,
    title: str = "PENgram",
    filename: str = "graph.html",
) -> Path:
    """Write an interactive graph.html to ``output_dir`` and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    write_text_if_changed(path, render_html(g, communities, title=title))
    return path


# The three vocabulary sets are exposed so the legend and tests can iterate.
__all__ = [
    "export_html",
    "render_html",
    "SEMANTIC_TYPES",
    "STRUCTURAL_TYPES",
]
