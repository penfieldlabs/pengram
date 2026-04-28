# ADR-006: PengramError exception hierarchy

## Status
Accepted

## Context
The codebase had 4+ bare `RuntimeError` and `ValueError` raises across
`transcribe.py`, `build.py`, and other modules. Callers wanting to
catch "any PENgram error" had to enumerate individual exception types,
and error handling in tests was fragile (matching on `RuntimeError`
when the real issue was a config problem).

`SecurityError(ValueError)` and `LLMError(RuntimeError)` already
existed but shared no common ancestor.

## Decision
Introduce `pengram.errors` with a `PengramError(Exception)` base.
All PENgram-specific exceptions inherit from it:

- `ConfigError(PengramError, ValueError)` — invalid configuration
- `ExtractionError(PengramError, ValueError)` — schema validation
- `SecurityError(PengramError, ValueError)` — input validation (in security.py)
- `LLMError(PengramError, RuntimeError)` — LLM call failures (in llm.py)

Multiple inheritance preserves backward compatibility: existing
`except ValueError` or `except RuntimeError` handlers still catch
these errors.

## Consequences
- `except PengramError` catches any library-raised error cleanly.
- Bare `RuntimeError`/`ValueError` raises replaced with specific
  subclasses that carry actionable context.
- The `errors.py` module has no internal imports, so it can't
  create circular dependencies.
