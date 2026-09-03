# chemistree benchmarks

Compares an agent **with** chemistree against the same model **without** it, on
chemical-structure tasks. See [`PLAN.md`](PLAN.md) for the full design (tracks, arms,
scoring, fairness, and the DUD-Z expansion planned next). This README covers setup, how to
run the harness, and the current abl1 results.

## Arms

Same model, same case; only the tool surface differs (see `arms.py`):

- **chemistree** — the chemistree MCP tools only (`bind`, `describe`, `swap`, `matches`,
  `residues_near`, …).
- **generalist** — `Bash` (rdkit, and smina for 3D) to do it by hand; the honest competitor.
- **naked** — no tools; reasons over the SMILES in the prompt. 2D only — with no 3D access it
  is dropped from both 3D tracks.

## Setup

1. **The `chemistree` env** — see the [top-level README](../README.md#setup)
   (`conda env create -f conda.yml`, `poetry install`). Run the harness from the repo root.
2. **The `smina` env (3D only)** — smina cannot go in the `chemistree` env (its openbabel
   dependency conflicts with Python 3.14), so it lives in its own env, and the oracle reads
   the binary from `SMINA_BIN`:
   ```bash
   conda create -n smina -c conda-forge smina -y
   export SMINA_BIN="$(conda run -n smina which smina)"   # runner + generalist arm read this
   ```
3. **`claude` on `PATH`** — the runner spawns one headless `claude -p` per case.

## Running

Each case runs one isolated `claude -p` (its own MCP-server child for the chemistree arm),
so cases never share context and every token is attributed to that case. The **task prompt
is byte-identical across arms** — only the appended system prompt differs (how the arm
reaches the molecule, plus optional guidance). Each result row records both `prompt` and
`system_prompt`, so parity is auditable. A 2D/decoration agent ends with a `FINAL_SMILES:`
line; a probe agent ends with `FINAL_ANSWER:`. The runner parses it and records token
usage, cost, turns, and — for the chemistree arm — a per-tool trace.

The whole abl1 mock-up (all arms, 2D + 3D) runs from one script:

```bash
SMINA_BIN="$(conda run -n smina which smina)" bash benchmarks/test_abl1.sh
```

Single arm / case set:

```bash
conda activate chemistree
python -m benchmarks.run --items benchmarks/cases/abl1_2d.jsonl --arm chemistree
python -m benchmarks.run --items benchmarks/cases/abl1_2d.jsonl --arm naked --dry-run  # print commands only
```

Flags: `--arm {chemistree,generalist,naked}`, `--model` (default `haiku`), `--out`,
`--limit`, `--timeout`, `--no-guidance` (drop the strategy layer — the base-prompt ablation).

**Clearing results?** Not needed. `run.py` opens `--out` in write mode, so each file is
**overwritten** every run, and `test_abl1.sh` also `rm -f`s `results/test_2d` and
`results/test_3d` at the start. Just re-run.

**Timeout & what we measure.** The per-case timeout is a **generous hang-guard** (600 s), not
a metric: tools run locally in milliseconds, so wall time is model thinking + API round-trips
— and *machine sleep* (a closed laptop) also counts against it. So **efficiency is measured by
cost/tokens** (immune to latency and sleep) and accuracy by correctness; a timeout means
genuinely stuck. Run on a machine that will not sleep — a suspended laptop corrupts wall time
and kills long runs.

## The abl1 worked example

One DUD-Z target, **abl1** (`tests/data/abl1/`), taken all the way through both tracks on
model **haiku**. The crystal ligand is
`Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)ccc1F`. Rows land uncompressed under
[`results/test_2d/`](results/test_2d/) and [`results/test_3d/`](results/test_3d/) (regenerated
by `test_abl1.sh`; **currently empty, pending the clean workstation run**). **A timeout drops
the row — read a missing row as a failure**, but confirm it is a genuine stall, not a sleep
artifact.

`notebooks/agents-view-edits.ipynb` walks all five 2D cases showing the exact agent-facing
context (group listing, per-group detail, `matches`, and the ring-position feedback) for each
editing method.

### Prompt parity

Every arm gets the **same task prompt**; only the system prompt differs. For `grow-halogen`:

```
Add a chlorine to the aniline ring at the open ring carbon that is ortho to the fluorine.
The molecule as SMILES: Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)ccc1F
End your reply with a line exactly of the form:
FINAL_SMILES: <smiles>
```

System prompt per arm (mechanics ± guidance): **naked** — none; **generalist** — "You have a
Bash tool with rdkit (python) available."; **chemistree** — "Edit the molecule with the
chemistree tools. Load it: call bind …", plus the shared medchem + chemistree tool guidance.

## Track A — 2D editing

Five NL→SMILES edits over the full crystal ligand (`cases/abl1_2d.jsonl`), one computed gold
each, canonical-SMILES match. Categories: **substitution** (×3), **growing**, **core hopping**.

_**Results: pending the clean workstation run.** Regenerate with `test_abl1.sh` and fill a
per-arm summary (accuracy, mean footprint/output tokens, turns, cost) and a per-case
correctness table here._

What to watch: whether **chemistree solves `swap-to-oxazole` and `core-hop-quinazoline`** — the
two hard cases (a heteroatom shuffle keeping both substituents; a fused-ring core hop) that the
`matches` + ring-numbering + prompt work targeted, and the only core-hop any arm has solved in
testing. Report cost/tokens as the efficiency axis (thinking-token cost, not verbosity, is why
the chemistree arm is priciest), and confirm any timeout is a genuine stall, not a sleep
artifact.

## Track B1 — 3D scaffold recovery

The crystal ligand is stripped to its **Murcko scaffold** (coords kept, so it stays posed —
`tasks/strip_murcko.py`) and each arm re-elaborates it into a potent binder
(`cases/abl1_3d_decoration.jsonl`). Scored by smina Vinardo — **redock** the final SMILES (all
arms) Δ vs the redocked scaffold baseline, plus **score-only the chemistree `write_pose` SDF**
(chemistree only) vs the scored scaffold pose (**−9.9**) — and **recovery**, ECFP4 Tanimoto to
the crystal (floor **0.328**). Each arm runs **with and without** guidance (the ablation). Naked
is dropped (no 3D access).

_**Results: pending the clean workstation run.** Fill a table of Δbinding (redock), the
chemistree pose-score (scaffold→final), recovery (→crystal), turns, footprint, and cost, for
each arm × guidance (guided | ablation)._

What to watch: whether **chemistree finishes a valid, improved, well-posed binder** where the
generalist (editing SMILES as text) does not; whether the `write_pose` score-only column beats
the blind redock (a minimum the redock search misses); and read **recovery and binding
together** — improving binding via a different chemotype moves recovery only modestly. Watch the
turn/token cost as the edit-budget argument.

## Track B2 — 3D understanding probes

Deterministic Q&A over the **full posed** crystal ligand (`cases/abl1_3d_probes.jsonl`), answer
matched to a computed ground truth. chemistree vs generalist.

_**Results: pending the clean workstation run.** Fill a per-probe correctness table
(chemistree vs generalist)._

Probes: `nearest-residue-cl` (closest residue to the chlorines → ASP149) and `cl-contact-count`
(residues within 4.0 Å of a chlorine → 9). `cl-contact-count` was a fix target: `contacts`
reports only the closest atom per residue and undercounts (chemistree once answered 6);
reframing `residues_near` as the count/list tool and redirecting from `contacts` routes the
agent to the right tool. Watch that it now returns 9.

## Caveats

**n = 1** target for the 3D tracks, **haiku**, a single stochastic sample (variance shuffles
which 2D case slips), and a **weak-signal** case (abl1's stripped groups are small halogens
worth ~1 kcal). This worked example validates the harness, the metrics, and the tool set end to
end; it does not yet support or refute the thesis. **Next:** the rest of DUD-Z (bigger R-groups,
n > 1), a sonnet pass, and an edit/turn budget. See [`PLAN.md`](PLAN.md).
