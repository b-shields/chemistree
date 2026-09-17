"""Resume logic for the benchmark runner: api-error rows are retried, good ones kept."""

from __future__ import annotations

from benchmarks.run import resume_plan


def _row(case_id, replicate, *, api_error=False, status="ok"):
    """A minimal result row for the resume tests."""
    return {
        "id": case_id,
        "replicate": replicate,
        "api_error": api_error,
        "status": status,
    }


def test_resume_plan_keeps_real_results_and_lists_them_done():
    """A non-api-error row is kept and its (id, replicate) counts as complete."""
    rows = [_row("a", 1), _row("a", 2)]
    keepers, completed = resume_plan(rows)
    assert keepers == rows
    assert completed == {("a", 1), ("a", 2)}


def test_resume_plan_drops_api_error_rows_so_they_rerun():
    """An api-error row is dropped and its (id, replicate) is left to run again."""
    rows = [_row("a", 1), _row("a", 2, api_error=True, status="no_final_smiles")]
    keepers, completed = resume_plan(rows)
    assert keepers == [_row("a", 1)]
    assert completed == {("a", 1)}  # (a, 2) is not done -> the runner will redo it


def test_resume_plan_keeps_genuine_non_api_failures():
    """A timeout or missed contract without an api error is a real result -- kept."""
    rows = [_row("a", 1, status="timeout"), _row("a", 2, status="no_final_smiles")]
    keepers, completed = resume_plan(rows)
    assert completed == {("a", 1), ("a", 2)}


def test_resume_plan_dedupes_repeated_keys():
    """Two kept rows for one (id, replicate) collapse to a single completed key."""
    rows = [_row("a", 1), _row("a", 1)]
    keepers, completed = resume_plan(rows)
    assert len(keepers) == 1
    assert completed == {("a", 1)}


def test_resume_plan_empty():
    """No prior rows -> nothing kept, nothing complete (a fresh run)."""
    assert resume_plan([]) == ([], set())
