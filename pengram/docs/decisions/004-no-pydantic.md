# ADR-004: No Pydantic — lean install

**Status:** Accepted  
**Date:** 2026-04-25  
**Context:** Package dependencies, `pyproject.toml`

## Decision

PENgram does not use Pydantic (or attrs, msgspec, or any schema-validation
library) for its internal data structures. Core dependencies are limited to
`networkx` and `pyyaml`.

## Rationale

PENgram is a CLI tool that users install into ad-hoc environments — sometimes
a bare venv, sometimes alongside an existing ML stack. Every transitive
dependency is a potential version conflict. Pydantic v1/v2 migration pain in
the broader ecosystem is a concrete example of this risk.

The extraction and linking pipeline passes dicts and dataclasses between
functions. Schema validation at the boundary (`validate.py`) uses plain
asserts and type checks. This is sufficient: the data shapes are stable, the
producers and consumers live in the same codebase, and the validation surface
is small enough to cover with unit tests.

## Alternatives considered

- **Pydantic:** strong runtime validation, but adds ~10 MB of transitive deps
  and a v1/v2 compatibility surface that conflicts with `openai`, `graspologic`,
  and other optional deps.
- **attrs / dataclasses + cattrs:** lighter, but still an extra dep for a
  problem adequately solved by dict access and `validate.py`.
- **msgspec:** fast, but niche — users would need to learn its API to
  contribute.
