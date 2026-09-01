# chemistree benchmarks

Compares an agent **with** chemistree against the same model **without** it, on
chemical-structure tasks. See [`PLAN.md`](PLAN.md) for the full design (tracks, arms,
scoring, fairness). This README covers how to run the harness and the current results.

## Arms

Same model, same case; only the tool surface differs (see `arms.py`):

- **chemistree** — the chemistree MCP tools only (`bind`, `describe`, `swap`, …).
- **generalist** — `Bash` (rdkit, and smina for 3D) to do it by hand; the honest competitor.
- **naked** — no tools; reasons over the SMILES in the prompt.

## Running

Each case runs one isolated `claude -p` (its own MCP-server child for the chemistree
arm), so cases never share context and every token is attributed to that case. The agent
must end with a `FINAL_SMILES:` line; the runner parses it and records the run's token
usage, cost, turns, and — for the chemistree arm — a per-tool trace.

```bash
conda activate chemistree
python -m benchmarks.run --items benchmarks/sample_2d.jsonl --arm chemistree
python -m benchmarks.run --items benchmarks/sample_2d.jsonl --arm naked --dry-run   # print commands only
```

Flags: `--arm {chemistree,generalist,naked}`, `--model` (default `haiku`), `--out`,
`--limit`, `--timeout`.

## 2D test (pipeline sanity)

Three hand-made single-edit cases (`sample_2d.jsonl`), model **haiku**. These are toy
edits chosen to exercise the whole pipeline end to end — **not** evidence for the thesis.
All three arms solve all three, so quality does not separate them here; the numbers below
are the token/turn cost of getting there. The raw per-case rows are saved under
[`results/test_2d/`](results/test_2d/) (one JSONL per arm).

Per-arm summary (n = 3):

| Arm | Accuracy | Mean context footprint (tok) | Mean output (tok) | Mean turns | Total cost |
|-----|:--------:|-----------------------------:|------------------:|-----------:|-----------:|
| naked      | 3/3 | 13,285 | 507   | 1.0 | $0.035 |
| generalist | 3/3 | 47,008 | 1,094 | 2.7 | $0.084 |
| chemistree | 3/3 | 95,046 | 1,292 | 6.3 | $0.095 |

Context footprint per case (all outputs correct):

| Case | naked | generalist | chemistree |
|------|------:|-----------:|-----------:|
| grow-methyl (add a methyl to benzene)     | 13,282 | 108,152 | 122,087 |
| remove-chloro (drop the Cl)               | 13,285 | 16,435  | 71,485  |
| swap-cf3 (methyl → trifluoromethyl)       | 13,288 | 16,438  | 91,567  |

### Reading this

On trivial edits, **naked wins on efficiency** — one turn, ~13k tokens — because the task
is so simple that a tool surface (schemas, `bind` → `describe_group` → edit → `smiles`
round-trips) costs context without repaying it. The generalist is bimodal: it reasons
directly on the easy two (~16k, one turn) and only pays when it actually shells out to
rdkit (grow-methyl, 108k). chemistree is the most expensive here because it faithfully
inspects and reads back through several tool calls.

This is the expected result for toy cases, and we report it plainly. chemistree's value is
hypothesized to appear where the naked/generalist approaches degrade — multi-substituted
ring edits addressed by stable id, and 3D structure-based optimization where a valid pose
and a scoring signal matter (Track B). Those are the tests that can actually support or
refute the thesis; this section only confirms the harness measures cleanly.

Next: the ChemCoTBench editing loader + official scorer (Track A) and the DUD-Z 3D track
(Track B) — see `PLAN.md` §8.
