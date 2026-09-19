# chemistree benchmarks

Compares an agent **with** chemistree against the same model **without** it, on
chemical-structure tasks. This README covers the design (tracks, arms, scoring, fairness),
the setup and how to run the harness, and the full results (ten DUD-Z targets plus the abl1
development case).

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
ring-position feedback, and a `matches` check. The agent builds a ring by name and is told where
each substituent landed. This ring table is the effect under test — so **no arm's prompt (the
chemistree tool guidance included) hands a case-specific answer SMILES, a tautomer, or a ring
size**, and the tool docstrings carry no procedure keyed to a benchmark question. Every arm,
chemistree included, relies on the model's own chemistry knowledge to name the target ring; what
chemistree adds is the tool that builds it by name and verifies where each substituent sits. The
task prompt is byte-identical across arms; the difference is exactly this tool-borne structure,
which is the effect under test.

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

Runs are **resumable and idempotent**. Each `run.py` keeps the prior real results in its
`--out` file and redoes only the samples that hit an **API error** (`is_error`, e.g. a spend
or rate limit) or never ran, so a rerun fills the gaps rather than repeating everything. On
the first API error a run **stops gracefully** — it preserves what it has, drops the failed
sample, and touches `results/full/.api_error_stop`, which halts the sweep before the next
target. Just re-run `run_benchmark.sh` (after the limit clears) to resume where it stopped;
`--no-stop-on-api-error` runs every case regardless.

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

## Full DUD-Z suite (N = 3)

Eleven case sets — the ten DUD-Z suite targets (egfr, aa2ar, andr, hivpr, fa10, hdac8, parp1,
hs90a, ada, nram) plus **abl1, the target chemistree was developed against** — model **haiku**,
**N = 3 samples** per case (495 rows). Rows land uncompressed under
[`results/full/`](results/full/), regenerated by `bash benchmarks/run_benchmark.sh` (`REPEAT=3`).
The 2D and probe tracks are binary-scored (canonical-SMILES or answer match); decoration is
scored by an independent smina oracle. 19 timeouts, all in the generalist decoration arm
(re-docking by hand); the other arms and tracks have none.

This is the **de-biased** run: the chemistree prompt no longer hands any answer SMILES or
probe-answering recipe (see **Arms**), so the numbers reflect the representation, not coaching.
Every figure is recomputed from the raw rows by
[`notebooks/benchmark-statistics.ipynb`](../notebooks/benchmark-statistics.ipynb), which also
runs the paired sign tests.

### Track A — 2D editing (27 cases × 3 = 81 rows)

One computed gold per case, canonical-SMILES match.

| Arm | Accuracy | Mean footprint | Mean cost |
|-----|:--------:|:--------------:|:---------:|
| naked | 55/81 (68%) | 16k | $0.063 |
| generalist | 52/81 (64%) | 94k | $0.067 |
| chemistree | **72/81 (89%)** | 166k | $0.058 |

**chemistree wins cleanly and significantly** — case-level sign test vs generalist 13–1
(p = 0.0018), vs naked 10–1 (p = 0.012) — and is the *cheapest* arm per case. The edge is widest
where building structure is hard, by category (solved/total):

| Arm | substitution | growing | core-hopping |
|-----|:---:|:---:|:---:|
| naked | 24/30 | 17/24 | 14/27 |
| generalist | 25/30 | 15/24 | 12/27 |
| chemistree | **29/30** | **21/24** | **22/27** |

chemistree solves 26 of the 27 cases on at least one replicate. The exception is
**`hdac8/grow-oxetane` (0/3)** — one of the cases whose answer SMILES the old prompt leaked. With
the leak removed, haiku must derive `oxetan-3-yl` itself and misses it; that single honest failure
is the fairness fix showing up in the numbers.

### Track B1 — 3D decoration (11 targets × 3 = 33 rows per condition)

Each target's crystal ligand is stripped to its Murcko scaffold and re-elaborated. **Success** =
improved predicted binding (redock Δ < 0 vs the scaffold). **Completion** = the oracle could dock
the returned molecule at all (reported separately, not the success metric); the chemistree
`write_pose` SDF is also scored directly with `--score_only`.

| Arm | Guidance | Success (Δ<0) | Completed | Mean Δbind | Recovery | Pose | Mean cost |
|-----|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| chemistree | guided | **24/33** | **32/33** | −0.66 | 0.31 | −8.1 | $0.34 |
| chemistree | ablation | 22/33 | **33/33** | −0.44 | 0.36 | −7.5 | $0.20 |
| generalist | guided | 15/33 | 23/33 | −0.43 | 0.29 | — | $0.28 |
| generalist | ablation | 14/33 | 22/33 | −0.79 | 0.27 | — | $0.14 |

**The headline is reliability.** chemistree returns a dockable molecule on 32–33/33 and improves
binding on 24/33 guided; the generalist docks only 22–23/33 and improves binding on 14–15/33 — it
produces SMILES the oracle can't dock, and its 19 timeouts are all here. But among molecules that
*both* arms dock, the binding improvement is a **near-tie**: chemistree is better on 12/23
co-completed pairs (mean ΔΔ = +0.01), and the guided success sign test is 6–1 (p = 0.13, n = 11,
not significant at this sample size). The decoration win is *completing the task*, not a larger
per-molecule binding gain.

### Track B2 — 3D probes (20 cases × 3 = 60 rows)

Deterministic Q&A, answer matched to a computed truth. Every probe is a question a medicinal
chemist asks while designing — which residue anchors or packs against a group, how many contacts
a group makes, whether a polar group can hydrogen-bond — each with an independently-computed,
air-tight gold (a ≥ 0.3 Å margin, so no answer is a coin-flip).

| Arm | Accuracy | Mean cost |
|-----|:--------:|:---------:|
| generalist | 55/60 (92%) | $0.045 |
| chemistree | **57/60 (95%)** | $0.034 |

Both arms are strong and the gap is **not significant** (sign test 5–2, p = 0.45). With the
probe-answering recipe removed from the chemistree prompt, the near-perfect probe result is gone —
chemistree is a few points higher and ~25% cheaper, but this is no longer a decisive win.

### Net

chemistree clearly wins **2D editing** (89% vs 64% / 68%, p ≈ 0.002) and is the only arm that
reliably **completes 3D decoration** (32/33 vs 23/33). On **3D probes** and on **per-molecule
binding gain** it is comparable to the honest generalist, not dominant. The effects land where the
representation should help — constructing and manipulating structure, and returning a valid posed
molecule — and the removed coaching is exactly where the gaps closed.

## Caveats

**N = 3 on haiku.** Three samples per case separate the large, reproducible effects (2D accuracy,
3D completion) from sampling noise, but the 3D binding/recovery means aggregate ≤ 33 rows, so read
small differences there loosely. **19 timeouts**, all in the generalist decoration arm (re-docking
by hand), count as failures, not tool limits. **Next:** a sonnet pass.

**Probe soundness.** The 20 probes are the ones that survive a four-part audit — a question a
medicinal chemist actually asks, one defensible answer, an independently-computed gold, and an
air-tight ≥ 0.3 Å margin — and each is verified answerable by driving the chemistree tool
backend directly to the gold. The chemistree prompt is *not* given a procedure for answering them
(that coaching was removed); both arms answer from the same shared medchem guidance plus their own
tools.
