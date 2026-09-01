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

### smina (3D only)

The 3D track scores with **smina**, which cannot go in the `chemistree` env (its openbabel
dependency conflicts with Python 3.14), so install it in **its own env** and point the
oracle at the binary:

```bash
conda create -n smina -c conda-forge smina -y
export SMINA_BIN="$(conda run -n smina which smina)"   # the runner and generalist arm read this
```

`claude` must also be on `PATH` (the runner spawns it).

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

## 3D test (scaffold recovery)

One case: abl1 (`tests/data/abl1/`, `sample_3d.jsonl`), model **haiku**. The crystal ligand
is stripped to its **Murcko scaffold** (coords kept, so it stays posed — `tasks/strip_murcko.py`),
and each arm is asked to *re-elaborate it into a potent binder*. Scored two ways, both by
smina Vinardo: **binding** = redock Δ vs the scaffold baseline (redocked scaffold **−11.1**,
crystal −12.1), and **recovery** = ECFP4 Tanimoto to the crystal ligand (scaffold floor
**0.328**, perfect rebuild 1.0). Raw rows: [`results/test_3d/`](results/test_3d/).

| Arm | Δaffinity (binding) | Recovery (→crystal) | Turns | Footprint (tok) | Cost | What it did |
|-----|:-------------------:|:-------------------:|:-----:|----------------:|-----:|-------------|
| naked      | −0.5 | 0.33 → **0.38** | 1  | 13,409    | $0.026 | guessed a generic F + Cl |
| generalist | — (failed to dock) | 0.33 → **0.20** (worse) | 9  | 219,314   | $0.140 | mangled the scaffold via SMILES text |
| chemistree | **−0.9** | 0.33 → 0.35 | 61 | 1,942,769 | $0.352 | added OH toward the pocket; a *different* valid binder |

### Reading this (n = 1 — illustrative, not conclusive)

The recovery metric does its job: **no arm can win by bloating**. What the run shows:

- **chemistree** improved binding the most (−0.9, matching the crystal) and kept a **valid,
  well-posed** molecule — but it explored a *different* chemotype (added hydroxyls toward
  the pocket, not the crystal's chlorines), so Tanimoto recovery barely moved. It also
  **badly over-explored** — 61 tool calls, 1.9M tokens, $0.35 — a clear argument for an
  edit/turn budget.
- **the generalist broke the molecule.** Editing SMILES as text with no structural feedback,
  it produced a final that would not even dock (`Δ=None`) and grew *less* like the crystal
  (recovery 0.33 → 0.20). That fragility is exactly what a structure-aware representation is
  meant to prevent.
- **naked** cheaply guessed a generic F + Cl — highest recovery of the three, but still near
  the floor and by luck, not pocket reasoning.

Caveats that keep this from meaning much yet: **n = 1**, **haiku**, a **weak-signal target**
(abl1's stripped groups are small halogens worth ~1 kcal), and recovery-to-a-*single*-crystal
is harsh — it penalizes chemistree for finding a different good binder. Report binding and
recovery side by side and read them together.

Next: more DUD-Z targets (bigger R-groups), an edit/turn budget to rein in cost, and maybe a
sonnet pass — plus the DUD-Z 2D case generator (Track A). See `PLAN.md` §8.
