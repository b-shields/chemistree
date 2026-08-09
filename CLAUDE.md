# Development Requirements

## Environment

- Conda env `chemistree` (Python 3.14). Activate with `conda activate chemistree`.
- Dependencies are managed by Poetry. Install with `poetry install`.
- Run tools inside the env: `poetry run pytest`, `poetry run black .`, etc.

## Dev workflow

- `pre-commit install` once after cloning; hooks run black, ruff, and mypy.
- Add runtime deps with `poetry add <pkg>`, dev deps with `poetry add --group dev <pkg>`.
- Commit `poetry.lock` so the environment stays reproducible.

## Commits

- Write commit messages about *what changed and why* — never about who or what
  authored the change. No tool attribution, no co-author trailers, no "generated
  by" lines.
- Commit frequently, in small logical units.
- Never commit code that hasn't passed its unit tests. pre-commit does not run
  pytest — run it yourself first.

## Coding guidelines

- Follow PEP 8; formatting is enforced by black (line length 88).
- Write clean, small functions that do one thing. Prefer pure functions.
- Use type hints on all public functions.
- Google-style docstrings on **every** function and method — public or private.
  A helper still has an API to explain; skipping its docstring because the name
  starts with `_` is exactly the inconsistency to avoid. Keep them compact and
  focused on what matters:
  - Lead with a one-line summary. That alone is the whole docstring when the
    signature already speaks for itself.
  - Public functions and methods document their `Args:` and `Returns:` (and
    `Raises:` when they raise) — the public API is worth spelling out in full.
  - A private helper, or a no-argument method/property, may stay a one-line
    summary; add sections there only to clarify something non-obvious. Never
    restate types or the obvious.
  - Dunder/protocol methods (`__init__`, `__repr__`, `__post_init__`, …) are
    covered by the class docstring and don't need their own.

  ```python
  def merge(tree: FragmentTree, other: FragmentTree) -> FragmentTree:
      """Merge two fragment trees into one.

      Args:
          tree: Base tree to merge into.
          other: Tree whose fragments are grafted onto ``tree``.

      Raises:
          ValueError: If the trees share incompatible root fragments.
      """
  ```
- The `_` prefix means module-private: a `_name` must not be imported by another
  module. The moment another module needs it, drop the `_` — it is now public API.

- Comments either (a) explain a non-obvious *why*, or (b) signpost the steps of a
  longer function with a brief step-marker header. Keep both terse — a line or two,
  never a paragraph. Let clear names and structure carry the rest.
- Prefer explicit over clever. Readability first.
- Keep imports at module top, sorted (ruff handles ordering).

## Testing

- **Develop with TDD.** Write the test first from the desired behavior, watch it
  fail, then implement until it passes.
- Tests assert a function's output against a known **ground truth** — a value you
  can verify by hand or from an authoritative source. Test behavior, not
  implementation.
- Keep tests clean and minimal: one clear thing per test, no incidental setup.
- Don't test what other packages already guarantee. A broken import or an
  rdkit-level failure surfaces on its own when pytest runs — asserting it adds
  noise, not coverage.
- Tests live in `tests/`, mirroring the `src/chemistree/` layout. Use pytest.
  Keep them fast and deterministic.
