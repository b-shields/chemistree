# chemistree benchmarks

Compares an agent **with** chemistree against the same model **without** it, on
chemical-structure tasks. See [`PLAN.md`](PLAN.md) for the full design (tracks, arms,
scoring, fairness, and the DUD-Z expansion planned next). This README covers setup, how to
run the harness, the full 10-target DUD-Z results, and the abl1 worked example.

## Arms

Same model, same case; only the tool surface differs (see `arms.py`):

- **chemistree** — the chemistree MCP tools only (`bind`, `describe`, `swap`, `matches`,
  `residues_near`, …).
- **generalist** — `Bash` (rdkit, and smina for 3D) to do it by hand; the honest competitor.
- **naked** — no tools; reasons over the SMILES in the prompt. 2D only — with no 3D access it
  is dropped from both 3D tracks.

**What chemistree gives its agent — and the others don't.** This is the comparison the
benchmark measures, so we state it plainly. The chemistree arm's tools carry curated domain
knowledge and structure: a vendored set of named heteroaromatic rings (name ⇄ canonical SMILES
plus IUPAC numbering), a fragment-tree representation with stable atom and group ids, per-edit
ring-position feedback, and a `matches` check. The agent builds a ring by name
(`swap 0 quinazoline 3@2 4@6`) and is told where each substituent landed. The shared medchem
prompt deliberately no longer lists ring SMILES, so the **generalist** and **naked** arms rely on
the model's own chemistry knowledge to write those rings — the same knowledge chemistree supplies
through its tools. The task prompt is still byte-identical across arms; the difference is exactly
this tool-borne knowledge, which is the effect under test.

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

The whole benchmark runs from one parallel script, **`run_benchmark.sh`**. For each target it
launches the full arm × track matrix at once (2D: naked/generalist/chemistree; 3D probes and
decoration: generalist/chemistree, decoration also with the `--no-guidance` ablation), batched
per target, then moves to the next. Defaults are **n = 3 samples on haiku**.

```bash
# 1. Smoke-test on abl1 first (reuses the abl1 case files):
bash benchmarks/run_benchmark.sh abl1

# 2. Then the full 10-target DUD-Z set:
bash benchmarks/run_benchmark.sh

# preview the exact job list without running or spending anything:
DRY=1 bash benchmarks/run_benchmark.sh
```

Env knobs (all optional): `REPEAT` (samples/case, default 3), `MODEL` (default `haiku`;
`sonnet` for the stronger pass), `TIMEOUT` (600 s hang-guard), `MAXPAR` (concurrent-run cap,
default 10; needs bash 4.3+), `TRACE=1` (per-tool traces), `SMINA_BIN`, `OUT` (default
`results/full`).

Results land under `benchmarks/results/full/<target>/` — one file per arm × track, each with
one row per case × sample (tagged with a `replicate` index):

```
results/full/<target>/
  2d_{naked,generalist,chemistree}.jsonl
  probes_{generalist,chemistree}.jsonl
  decorate_{generalist,chemistree}[_noguid].jsonl
```

Old results are cleared per target on rerun (`run.py` overwrites each `--out`; the script
`rm -rf`s each target dir first), so just re-run — no manual clearing.

**Regenerating the cases.** The case files, Murcko scaffold seeds, and probe answers are all
produced by `python -m benchmarks.gen_cases` (reads `data/manifest.jsonl`). Golds and probe
answers are computed and self-checked with RDKit / numpy, **independent of the chemistree tools
under test**. Review every case (prompt + input + expected output) in
[`notebooks/check-test-structures.ipynb`](../notebooks/check-test-structures.ipynb).

Single arm / case set:

```bash
python -m benchmarks.run --items benchmarks/cases/egfr_2d.jsonl --arm chemistree
python -m benchmarks.run --items benchmarks/cases/egfr_2d.jsonl --arm naked --dry-run  # print only
```

Flags: `--arm {chemistree,generalist,naked}`, `--model` (default `haiku`), `--out`, `--limit`,
`--repeat` (samples/case), `--timeout`, `--no-guidance` (drop the strategy layer — the ablation).

**Timeout & what we measure.** The per-case timeout is a **generous hang-guard** (600 s), not
a metric: tools run locally in milliseconds, so wall time is model thinking + API round-trips
— and *machine sleep* (a closed laptop) also counts against it. So **efficiency is measured by
cost/tokens** (immune to latency and sleep) and accuracy by correctness; a timeout means
genuinely stuck. Run on a machine that will not sleep — a suspended laptop corrupts wall time
and kills long runs.

## Full DUD-Z suite (N = 1)

Ten DUD-Z targets spanning ten target classes (egfr, aa2ar, andr, hivpr, fa10, hdac8, parp1,
hs90a, ada, nram), model **haiku**, **N = 1 sample** per case — one pass over the whole matrix.
Rows land uncompressed under [`results/full/`](results/full/), regenerated by
`bash benchmarks/run_benchmark.sh`. The tables below are the 2026-09-03 workstation run. N = 1
is a breadth pass: read the **direction** across targets here, and the **magnitude** from the
abl1 n = 3 worked example below. The 3D means aggregate one sample per target, so treat small
differences as noise; the large ones (2D accuracy, 3D completion) are the signal.

### Track A — 2D editing (22 cases across 10 targets)

One computed gold per case, canonical-SMILES match. 0 timeouts across all 66 rows.

| Arm | Accuracy | Mean footprint | Mean cost |
|-----|:--------:|:--------------:|:---------:|
| naked | 10/22 (45%) | 15k | $0.066 |
| generalist | 17/22 (77%) | 95k | $0.081 |
| chemistree | **21/22 (95%)** | 145k | $0.074 |

**chemistree wins cleanly**, and is the only arm to solve five cases that *both* others miss:
`aa2ar/sub-methoxy`, `andr/sub-amine`, `hdac8/core-thiophene`, `hdac8/grow-oxetane`, and
`nram/core-tetrazole`. Its lone miss is `hivpr/core-naphthalene` (which generalist got). The
extra footprint buys the accuracy at no cost premium — chemistree is actually *cheaper* per case
than the generalist here (fewer failed Bash round-trips).

### Track B1 — 3D decoration (10 targets)

Each target's crystal ligand is stripped to its Murcko scaffold and re-elaborated; scored by
smina Vinardo redock Δ vs the scaffold baseline, ECFP4 recovery to the crystal, and (chemistree
only) the `write_pose` SDF scored `--score_only`. **Completion** = the arm returned a molecule
the oracle could dock at all.

| Arm | Guidance | Completed | Δbinding | Recovery | Pose score | Mean turns | Mean cost |
|-----|----------|:---------:|:--------:|:--------:|:----------:|:----------:|:---------:|
| chemistree | guided | **10/10** | −0.54 | 0.32 | −7.6 | 53 | $0.32 |
| chemistree | ablation | **10/10** | −0.44 | 0.38 | −7.2 | 37 | $0.20 |
| generalist | guided | 5/10 | −0.48 | 0.26 | — | 22 | $0.26 |
| generalist | ablation | 8/10 | −0.30 | 0.27 | — | 15 | $0.17 |

**The headline is reliability, not magnitude.** chemistree returns a valid dockable molecule on
**10/10** targets in both conditions; the generalist drops **half** of them under guidance
(5/10) and 2 without it — it produces SMILES the oracle can't dock when it has to manipulate the
scaffold by hand. Among the molecules that *do* complete, the binding and recovery deltas are
small and comparable across arms; at N = 1 per target, do not over-read them — the abl1 n = 3
run is where the magnitude (chemistree guided −1.5 kcal, recovery 0.45) is measured.

### Track B2 — 3D probes (20 cases)

Deterministic Q&A, answer matched to a computed truth: generalist **15/20**, chemistree
**16/20** — essentially tied. Probes exercise reading a fixed pose, where a Bash+rdkit agent is
already competent, so they don't separate the arms.

### Net

chemistree dominates 2D editing (95% vs 77% vs 45%) and is the *only* arm that reliably finishes
3D decoration (10/10 vs 5/10 completion). 3D probes are a wash. The two large effects are exactly
where the tool-borne knowledge and stable ids should help — constructing and manipulating
structure — and the wash is where they shouldn't.

## The abl1 worked example

One DUD-Z target, **abl1** (`tests/data/abl1/`), taken all the way through both tracks on
model **haiku**, **n = 3 samples** per case. The crystal ligand is
`Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)ccc1F`. Rows land uncompressed under
[`results/full/abl1/`](results/full/abl1/), regenerated by `bash benchmarks/run_benchmark.sh
abl1`. The tables below are the 2026-09-03 workstation run. **A timeout drops the row — read a
missing row as a failure**, but confirm it is a genuine stall, not a sleep artifact.

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

Per arm (n = 3 samples × 5 cases = 15 rows):

| Arm | Accuracy | Mean footprint | Mean output tok | Mean turns | Mean cost |
|-----|:--------:|:--------------:|:---------------:|:----------:|:---------:|
| naked | 9/15 | 15.7k | 12.2k | 1.0 | $0.068 |
| generalist | 11/15 | 76.9k | 7.0k | 3.3 | $0.063 |
| chemistree | **13/15** | 256k | 11.8k | 10.2 | $0.120 |

Per case (correct samples / 3):

| Case | Category | naked | generalist | chemistree |
|------|----------|:-----:|:----------:|:----------:|
| sub-halogen | substitution | 3/3 | 3/3 | 3/3 |
| sub-ring-to-pyridine | substitution | 2/3 | 3/3 | 2/3 |
| swap-to-oxazole | substitution | 1/3 | 2/3 | 2/3 |
| grow-halogen | growing | 3/3 | 3/3 | 3/3 |
| core-hop-quinazoline | core hopping | 0/3 | 0/3 | **3/3** |

Findings: **chemistree is the only arm to solve `core-hop-quinazoline`** (3/3 vs 0/3 for both
others) — the fused-ring core hop the ring-by-name + `matches` + ring-numbering work targeted.
`swap-to-oxazole` (the heteroatom shuffle keeping both substituents) stays hard for every arm
(≤ 2/3). The trade is context and turns: chemistree spends ~3.3× the footprint and ~10
turns/case (thinking-token cost, not verbosity — its output tokens match naked's). One naked
`core-hop` sample **timed out** and counts as a failure. Efficiency is cost/tokens, not wall
time; confirm any timeout is a genuine stall, not a sleep artifact.

## Track B1 — 3D scaffold recovery

The crystal ligand is stripped to its **Murcko scaffold** (coords kept, so it stays posed —
`tasks/strip_murcko.py`) and each arm re-elaborates it into a potent binder
(`cases/abl1_3d_decoration.jsonl`). Scored by smina Vinardo — **redock** the final SMILES (all
arms) Δ vs the redocked scaffold baseline, plus **score-only the chemistree `write_pose` SDF**
(chemistree only) vs the scored scaffold pose (**−9.9**) — and **recovery**, ECFP4 Tanimoto to
the crystal (floor **0.328**). Each arm runs **with and without** guidance (the ablation). Naked
is dropped (no 3D access).

Means over n = 3 (Δbinding negative = better; pose score is the chemistree `write_pose` SDF
scored `--score_only`, scaffold baseline **−9.9**; recovery is ECFP4 Tanimoto to the crystal,
scaffold floor **0.328**):

| Arm | Guidance | Δbinding (redock) | Recovery (→final) | Pose score (scaffold→final) | Mean turns | Mean footprint | Mean cost |
|-----|----------|:-----------------:|:-----------------:|:---------------------------:|:----------:|:--------------:|:---------:|
| chemistree | guided | **−1.5** | **0.45** | −9.9 → **−11.9** | 58 | 2.09M | $0.35 |
| chemistree | ablation | −0.6 | 0.36 | −9.9 → −10.4 | 31 | 652k | $0.14 |
| generalist | guided | −0.6 | 0.38 | — | 17 | 482k | $0.20 |
| generalist | ablation | +0.3 | 0.31 | — | 12 | 361k | $0.16 |

Findings: **chemistree + guidance is the only arm that reliably improves binding** — all three
samples give a negative Δbinding (mean −1.5 kcal), while the **generalist ablation on average
makes binding worse** (+0.3) and the generalist guided has a sample that worsens it (+1.9). The
chemistree `write_pose` score-only column (**−11.9**) beats both its own blind redock and the
scaffold pose baseline (−9.9) — the built pose captures binding the redock search does not
always recover. Guidance matters most for chemistree (−1.5 vs −0.6 without it), and pulls
recovery up with it (0.45 vs 0.36; best single sample **0.556** against the 0.328 scaffold
floor). Read recovery and binding together — improving binding via a different chemotype moves
recovery only modestly. The price is the edit budget: chemistree guided runs ~58 turns and a
~2.1M-token footprint per case ($0.35), the most of any cell.

## Track B2 — 3D understanding probes

Deterministic Q&A over the **full posed** crystal ligand (`cases/abl1_3d_probes.jsonl`), answer
matched to a computed ground truth. chemistree vs generalist.

Per probe (correct samples / 3):

| Probe | Truth | generalist | chemistree |
|-------|:-----:|:----------:|:----------:|
| nearest-residue-cl | ASP149 | 3/3 | 3/3 |
| cl-contact-count | 9 | 3/3 | 3/3 |

Both arms answer both probes correctly on every sample (~$0.033/case; chemistree ~6.5 turns vs
generalist ~4.5). `cl-contact-count` was a fix target: `contacts` reports only the closest atom
per residue and undercounts (chemistree once answered 6); reframing `residues_near` as the
count/list tool and redirecting from `contacts` routes the agent to the right tool. It now
returns **9** across all three chemistree samples — the fix holds.

## Caveats

The abl1 example is **haiku**, **n = 3 stochastic samples** (variance still shuffles which 2D
case slips), on a **weak-signal** target (abl1's stripped groups are small halogens worth
~1 kcal). It validates the harness, the metrics, and the tool set end to end, and supplies the
per-case magnitude the N = 1 full suite above cannot. Read the two together: the suite shows the
direction holds across ten target classes, abl1 shows how large the effect is when sampled. The
full suite is still **N = 1 per case** — the 3D means are one sample per target, so only the
large effects (2D accuracy, 3D completion) are trustworthy yet. **Next:** an n ≥ 3 pass over the
full suite for 3D magnitude, a sonnet pass, and an edit/turn budget. See [`PLAN.md`](PLAN.md).
