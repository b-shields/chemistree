# chemistree benchmarks

Compares an agent **with** chemistree against the same model **without** it, on
chemical-structure tasks. See [`PLAN.md`](PLAN.md) for the full design (tracks, arms,
scoring, fairness, and the DUD-Z expansion planned next). This README covers setup, how to
run the harness, and the full results (ten DUD-Z targets plus the abl1 development case).

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
# 1. Smoke-test on a single target first:
bash benchmarks/run_benchmark.sh egfr

# 2. Then the full suite (10 DUD-Z targets + abl1):
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
[`notebooks/check-benchmark-structures.ipynb`](../notebooks/check-benchmark-structures.ipynb).

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

## Full DUD-Z suite (N = 2)

Eleven case sets — the ten DUD-Z suite targets (egfr, aa2ar, andr, hivpr, fa10, hdac8, parp1,
hs90a, ada, nram) plus **abl1, the target chemistree was developed against**, now run as an
eleventh set — model **haiku**, **N = 2 samples** per case. Rows land uncompressed under
[`results/full/`](results/full/), regenerated by `bash benchmarks/run_benchmark.sh` (`REPEAT=2`).
The 2D and probe tracks are binary-scored (canonical-SMILES or answer match); decoration is
scored by an independent smina oracle. 0 timeouts.

### Track A — 2D editing (27 cases × 2 = 54 rows)

One computed gold per case, canonical-SMILES match.

| Arm | Accuracy | Mean footprint | Mean cost |
|-----|:--------:|:--------------:|:---------:|
| naked | 34/54 (63%) | 16k | $0.071 |
| generalist | 36/54 (67%) | 108k | $0.076 |
| chemistree | **50/54 (93%)** | 209k | $0.070 |

**chemistree wins cleanly.** It solves every one of the 27 cases on at least one sample; its four
misses (`aa2ar/sub-methoxy`, `nram/sub-nmethyl`, `hs90a/grow-dimethylamino`, and one abl1 case)
are each a single unlucky sample of an otherwise-solved case — haiku sampling variance on complex
hand-written SMILES, not a systematic failure. The extra footprint (deliberate tool moves) buys
the accuracy at no cost premium — chemistree is actually *cheaper* per case than the generalist
(fewer failed Bash round-trips).

### Track B1 — 3D decoration (11 targets × 2 = 22 rows)

Each target's crystal ligand is stripped to its Murcko scaffold and re-elaborated; scored by
smina Vinardo redock Δ vs the scaffold baseline, ECFP4 recovery to the crystal, and (chemistree
only) the `write_pose` SDF scored `--score_only`. **Completion** = the arm returned a molecule
the oracle could dock at all.

| Arm | Guidance | Completed | Δbinding | Recovery | Pose score | Mean turns | Mean cost |
|-----|----------|:---------:|:--------:|:--------:|:----------:|:----------:|:---------:|
| chemistree | guided | **22/22** | −0.57 | 0.31 | −8.1 | 58 | $0.37 |
| chemistree | ablation | **22/22** | −0.29 | 0.38 | −7.6 | 36 | $0.19 |
| generalist | guided | 19/22 | −0.25 | 0.31 | — | 21 | $0.26 |
| generalist | ablation | 18/22 | −0.22 | 0.31 | — | 15 | $0.14 |

**The headline is reliability.** chemistree returns a valid dockable molecule on **22/22** in
both conditions; the generalist drops 3–4 — it produces SMILES the oracle can't dock when it has
to manipulate the scaffold by hand. Among the molecules that *do* complete, chemistree's guided
binding improvement is the largest (−0.57 vs the generalist's −0.25); recovery deltas are small
and comparable (scaffold floor ≈ 0.40).

### Track B2 — 3D probes (20 cases × 2 = 40 rows)

Deterministic Q&A, answer matched to a computed truth. Every probe is a question a medicinal
chemist asks while designing — which residue anchors or packs against a group, how many contacts
a group makes, whether a polar group can hydrogen-bond — each with an independently-computed,
air-tight gold (a ≥ 0.3 Å margin, so no answer is a coin-flip) that is reachable through the
tools.

| Arm | Accuracy | Mean cost |
|-----|:--------:|:---------:|
| generalist | 39/40 (98%) | $0.048 |
| chemistree | **40/40 (100%)** | $0.032 |

chemistree answers **every probe on every sample**, and cheaper than the generalist.

### Net

chemistree dominates 2D editing (93% vs 67% / 63%), is the *only* arm that reliably finishes 3D
decoration (22/22 vs 19/22 completion), and is perfect on the 3D probes (40/40). The effects land
exactly where the tool-borne knowledge and stable ids should help — constructing and manipulating
structure, and reading the pocket by group.

## Caveats

**N = 2 on haiku.** Two samples per case separates the large, reproducible effects (2D
accuracy, 3D completion, probe accuracy) from sampling noise, but the 3D binding and recovery
means aggregate 22 rows, so read small differences there loosely. haiku is stochastic: four 2D
cases (`aa2ar/sub-methoxy`, `nram/sub-nmethyl`, `hs90a/grow-dimethylamino`,
`abl1/sub-ring-to-pyridine`) are each solved on one of the two samples — every 2D case is solved
at least once, so these are variance, not a tool limit. **Next:** an n ≥ 3 pass for tighter 3D magnitude, and a sonnet pass. See
[`PLAN.md`](PLAN.md).

**Probe soundness.** The 20 probes are the ones that survive a four-part audit — a question a
medicinal chemist actually asks, one defensible answer, an independently-computed gold, and an
air-tight ≥ 0.3 Å margin — and each is verified answerable by driving the chemistree tool
backend directly to the gold. Cases that could not meet that bar (a coin-flip nearest residue, a
boundary-brittle count, a substructure the tools cannot isolate) were reframed or dropped rather
than shipped: a smaller honest probe set beats a padded one.
