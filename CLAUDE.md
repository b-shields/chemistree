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
- Google-style docstrings on public modules, classes, and functions:

  ```python
  def merge(tree: FragmentTree, other: FragmentTree) -> FragmentTree:
      """Merge two fragment trees into one.

      Args:
          tree: Base tree to merge into.
          other: Tree whose fragments are grafted onto ``tree``.

      Returns:
          A new tree containing fragments from both inputs.

      Raises:
          ValueError: If the trees share incompatible root fragments.
      """
  ```

- Comments explain *why*, not *what*. Use them sparingly — a line or two, never a paragraph. Let clear names and structure carry the rest.
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
