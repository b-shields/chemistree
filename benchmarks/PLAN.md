# chemistree benchmark plan

Status: **design, not yet built.** This document is the hand-off so the work can
continue on another machine. Read it top to bottom before writing code.

## 0. The claim we are testing

The chemistree thesis for the blog post: *a tree-based fragment representation with
stable atom/group ids and consistent 2D+3D geometry lets an LLM agent understand and
edit chemical structures better than a generalist agent with the same model.*

The benchmark must isolate **the representation** as the variable — not "tools help."
A generalist agent that can write RDKit and run a docking engine is the honest
competitor. If chemistree only beats a *no-tools* model, that proves nothing
interesting. The headline result we want is one of:

1. **chemistree(haiku) > generalist(haiku)** on structure editing/optimization, and/or
2. **chemistree(haiku) ≈ generalist(sonnet)** — the representation lets a cheap model
   match an expensive one.

## 1. Two tracks

### Track A — 2D editing (literature benchmark: ChemCoTBench)

- Source: https://github.com/IDEA-XL/ChemCoTBench (arXiv:2505.21318), HF dataset.
- Use the **Molecule Editing** subtask (add / delete / substitute a substructure).
  Best 1:1 fit to `grow` / `remove` / `swap` / `mutate`. SMILES in, SMILES out,
  canonical-match scoring.
- Optionally add the **physchem Optimization** subtask (improve QED / LogP / solubility)
  — cheap RDKit oracles, needs feedback in the loop.
- **Skip** Understanding, Reactions, and the DRD2/JNK3/GSK3β activity-oracle optimization
  tasks. The tree buys nothing on retrosynthesis/mechanism, and the activity oracles are
  QSAR models, not structure — including them dilutes the signal.
- Caveat to state in the writeup: this track tests only the *2D* half of the thesis.
  It has no receptor and cannot reward pose / clashes / `minimize`.

> **Verified (2026-09-01): 2D editing works.** Drove a receptor-free, conformer-free
> `DesignSession(smiles, three_d=False)` through swap / grow / mutate / remove / undo and
> chained edits with re-fragmentation on benzanilide — every edit produced the correct
> canonical SMILES, and describe/describe_group/positions/properties all render with no
> conformer. The 3D tools (clashes, minimize) self-gate with a clear "needs 3D
> coordinates" error rather than misbehaving. So Track A is unblocked; the only 2D concern
> is context hygiene, handled by the `--tools 2d` launch profile (see §3).

### Track B — 3D structure-based optimization (our own benchmark: DUD-Z)

This is the track chemistree is uniquely built for and no public agentic benchmark
covers. It is the differentiator.

- Targets: **DUD-Z / DUDE-Z** (Shoichet lab, property-matched decoys). Chosen because
  the receptors come prepared and each target ships a crystal/reference ligand pose we
  can seed from. Pick ~8–12 targets spanning easy/hard pockets.
- Task shape (**optimization**): seed the session with a known active (or the reference
  ligand) posed in the pocket; ask the agent to *improve predicted binding while keeping
  the molecule drug-like and physically valid*. The agent edits; we score the result
  with an **independent** docking oracle.
- Optional sub-track **B2 — 3D understanding probes**: deterministic Q&A with ground
  truth read straight from the prepared complex (e.g. "which residue is closest to the
  chlorine?", "list the groups that clash", "order these groups by distance to ASP381").
  Cheap to build, tests "understanding" directly, and only chemistree exposes the 3D
  view to answer from.

## 2. Arms (hold the model fixed; vary only tooling)

| Arm | Name | Tools | Role |
|-----|------|-------|------|
| **N** | naked | none (SMILES in the prompt) | floor; pure LLM structural reasoning |
| **G** | generalist | Bash with `rdkit` + (Track B) `smina` | the honest competitor: can code, edit SMILES, dock by hand |
| **C** | chemistree | chemistree MCP tools only, no Bash | the representation under test |

- Primary model: **haiku**. Secondary run: **sonnet**, all arms, to test claim (2).
- Optional 4th arm **C+G** (chemistree *and* Bash) to see whether they compose — keep
  optional, run only if time allows.
- Track A real contest: **G(rdkit) vs C**. Track B real contest: **G(rdkit+smina) vs C**
  (arm N is near-floor in 3D since it has no way to score a pose).

## 3. Architecture: how an agent gets chemistree (the `claude mcp add` question)

**Recommendation: build a standalone in-process MCP server. It is both the benchmark
harness's tool surface *and* the demo pitch ("`claude mcp add` and any agent understands
structure"). Prefer it over the SDK headless harness.**

### Why the current MCP server is not enough

`.mcp.json` runs `python -m chemistree.app.mcp`, and every tool there is a *thin HTTP
client* of the running FastAPI web app (`CHEMISTREE_URL`). It requires the browser app
up with a pre-configured, fixed session. Unusable for "add chemistree to an arbitrary
agent" and unusable for per-item molecule loading in a benchmark.

### What to build: `chemistree-mcp`, a self-contained server the agent binds into

A new module `src/chemistree/mcp_server.py` (public API — no `app.` prefix; it is a
first-class product surface, not part of the web app) that:

1. Starts **empty** — no molecule pre-loaded. It holds a single mutable session slot.
2. Exposes a **`bind` tool** the agent (or the harness's first prompt) calls to set the
   molecule:
   `bind(molecule: str, receptor: str | None = None) -> str` where `molecule` is a SMILES
   string or an SDF/MOL path and `receptor` is an optional PDB path. It constructs a
   `DesignSession` in-process, replacing any current one, and returns `describe()` so the
   agent immediately sees the group listing. Binding again resets — so one long-lived
   server handles many molecules (fast for benchmarking, natural for the demo).
3. All other tools call `commands.run_command(session, text)` **directly — no HTTP**.
   `run_command` is already the reusable core (the web route and the app MCP server both
   go through it), so the tool bodies are one-liners. They error clearly until `bind` is
   called.
4. Ships a poetry script: `chemistree-mcp = "chemistree.mcp_server:main"`.
5. Optional `CHEMISTREE_STATE_OUT` env path: on each edit (or on exit) writes the final
   canonical SMILES, so a benchmark harness reads the answer from a file instead of
   parsing agent free-text (fragile).
6. **Benchmark mode** — `--trace <path.jsonl>` / `CHEMISTREE_TRACE`: append one JSON
   record per tool call (tool, args, returned string + its char/approx-token size,
   latency, resulting SMILES, bind events). This is the tool-side instrumentation of §4's
   efficiency accounting — the per-tool context breakdown the model-side usage can't give.
   Off by default; zero overhead when unset.

Then, for any agent — no env plumbing, the agent binds what it's told to work on:
```
claude mcp add chemistree -- chemistree-mcp
# then: "bind O=C(Nc1ccccc1)c1ccccc1 and swap the left phenyl for a pyridine"
```

### The 2D/3D tool split (verified) and how to toggle it

Confirmed empirically (see "Verified" box in §1-notes below): the tools fall into three
tiers by what the bound molecule carries.

| Tier | `bind` gives | Tools available |
|------|--------------|-----------------|
| **2D** | SMILES, no conformer | describe, describe_group, smiles, swap, grow, mutate, remove, undo |
| **3D, no receptor** | + a conformer | + clashes, minimize (intra-only energy) |
| **3D + receptor** | + a PDB | + distance, contacts, affinity/inter scoring |

`bind` decides the tier per molecule: a receptor → 3D+receptor; otherwise 2D unless a
`pose=True`/`three_d` flag asks for a conformer. The 3D tools **already self-gate** — with
no conformer they raise `"...needs a ligand with 3D coordinates"` — so a wrong call is
safe, never silently wrong.

**But self-gating does not stop context pollution:** MCP clients enumerate a server's
tools once at connection, so 3D tools a 2D task never uses still consume prompt budget and
invite misuse. Dynamically hiding tools mid-session after `bind` is **not** reliably
supported (the client caches the tool list). So the toggle is a **launch-time tool
profile**, not in-session:
- `chemistree-mcp --tools 2d` — register only the 2D editing set. Use for Track A so the
  agent's context is not polluted with pocket/pose tools it can't use.
- `chemistree-mcp` (default) — register everything; the 3D tools self-gate until a
  conformer/receptor is bound.

(If a future MCP feature makes per-session tool lists dynamic, `bind` could set the tier
live; until then, launch profile is the honest mechanism.)

### Refactor to avoid duplicating tool docstrings

The tool docstrings **are** the agent's understanding of chemistree — they must have one
source of truth. Factor the tool set into `src/chemistree/mcp_tools.py` that registers
all `@mcp.tool` functions onto a passed-in `FastMCP`, given a backend:
- a `run(text: str, *, with_state: bool) -> str` callable, and
- a `state() -> dict` callable.

Two backends implement that interface:
- **in-process** (`mcp_server.py`): `run` = `run_command(session, ...)`, `state` reads
  the session directly. Used for `claude mcp add` and the benchmark.
- **HTTP** (`app/mcp.py`, unchanged behavior): `run`/`state` hit the web app. Used by the
  live demo so the viewers update.

Net: one `mcp_tools.py` with all docstrings; two ~20-line entrypoints. `app/mcp.py`
shrinks to the HTTP backend + `mcp_tools.register(mcp, http_backend)`.

### Compared to the SDK headless harness (the alternative I first proposed)

The headless/Agent-SDK harness (register in-process Python tools, run the agent loop in
one process) works, but it **re-implements** tool exposure, loses the "it's just an MCP
server anyone can add" story, and doesn't produce a reusable artifact. Keep it only as a
fallback if per-item process spawn proves too slow — it will not, for haiku on ~hundreds
of items, and spawns parallelize trivially.

### Per-item execution model (both tracks)

Fresh process per item = fresh session = no cross-contamination:
```
for item in dataset:
    (Track B) stage tmp receptor.pdb + define the docking box from the reference ligand
    spawn:  claude -p "<task prompt naming the SMILES/receptor to bind, then the task>" \
              --mcp-config <arm's mcp config> \
              --allowedTools <arm's tools> --disallowedTools <rest> \
              --model haiku --output-format stream-json
            with env CHEMISTREE_STATE_OUT=tmp_out.smi
    # the agent's first action is bind(<smiles>[, receptor]); then it edits
    wait; read final SMILES from CHEMISTREE_STATE_OUT (fallback: agent's stated answer)
    score (see §4); record tokens/turns/wall-time from the result line
```
The molecule reaches the agent **through the task prompt + `bind`**, not env vars — the
harness never has to reconfigure the server. Arm C uses `--tools 2d` for Track A.
Arm N: no `--mcp-config`, all tools disallowed. Arm G: no chemistree MCP, `Bash`
allowed, a staged workdir holding the receptor/ligand + smina on PATH. Reuse the
`build_command`-style flags from `app/chat.py` as the template.

(A single long-lived server also works since `bind` resets per item, but fresh-per-item
keeps isolation trivial and parallelizes; revisit only if spawn cost bites.)

## 4. Scoring & metrics

### Track A (2D editing)
- **Validity**: RDKit-parseable, sanitizable.
- **Correctness**: canonical-SMILES match to gold for deterministic edits
  (canonicalize both sides; consider tautomer-insensitive match). For substitution,
  also accept substructure-constraint satisfaction where the gold is a family.
- **Minimality**: Tanimoto / MCS to the intended product — did it change *only* what was
  asked (catches chemistree over-editing via re-fragmentation, and catches N/G mangling
  the SMILES).
- **Efficiency**: see the shared efficiency subsection below.

### Track B (3D optimization)
- **Oracle = independent docking**: re-dock the final molecule with **smina** into the
  same box and report affinity. Use smina's **Vina** default scoring, *not* Vinardo, so
  the oracle is not the same function chemistree optimizes against internally (avoid
  teaching-to-the-test). Report Δaffinity vs the seed.
- **Success**: fraction of items with Δaffinity beyond a threshold, at a fixed edit
  budget (`success@budget`), plus best-of.
- **Physical validity gate**: the pose must dock into the box (not fly out), pass a
  strain/clash check, and stay within a **property window** (QED / MW / cLogP guardrails)
  so an arm cannot "win" by bolting on greasy mass. Failing the gate = no credit.
- **Efficiency**: see the shared efficiency subsection below.

### Track B2 (understanding probes)
- Accuracy vs deterministic ground truth from the prepared complex.

### Efficiency & instrumentation (all tracks) — the context-efficiency win

Context efficiency is a first-class result: if arm C matches G on quality but uses far
fewer tokens/turns, that is a win worth reporting on its own. Two complementary logs, one
per side, merged per item by the harness:

**1. Model-side usage — from Claude Code's final `result` line** (works for every arm; it
is Claude Code's own accounting, tooling-agnostic). One `claude -p "<task>"` runs the
whole agentic loop and emits a single final `result` whose `usage` is cumulative and whose
`num_turns` counts the internal tool-call rounds. Capture, per item:
- `usage.input_tokens`, `usage.cache_creation_input_tokens`, `usage.cache_read_input_tokens`
  → **context footprint** = the sum of the three (all tokens the model had to process;
  cache only changes their price, not that they were context).
- `usage.output_tokens` (with `output_tokens_details.thinking_tokens`).
- `total_cost_usd` (list price — report alongside the raw footprint; they tell different
  stories because cache reads are discounted).
- `num_turns`, `duration_ms`.
Report the footprint **and** cost; do not collapse to one. This is where C should beat G
even at quality parity: G feeds raw rdkit/smina stdout back into context each turn, while
C's `describe`/`describe_group` output is compact and dense.

**2. Tool-side trace — the MCP server's benchmark mode** (arm C only; arm G would need an
analogous Bash/tool logger, out of scope — its tool cost shows up in G's model-side input
tokens anyway). The server sees the tool traffic the model-side usage cannot attribute, so
add a **benchmark mode** to `chemistree-mcp`: `--trace <path.jsonl>` (or env
`CHEMISTREE_TRACE`). Append one JSON record per tool call:
- timestamp, tool name, arguments;
- the returned string and its **size (chars + approx tokens)** — this is exactly the
  context that tool injected, so it attributes the footprint per tool;
- call latency; the resulting canonical SMILES (edit trajectory);
- `bind` events (molecule, receptor, tier).
This gives the per-tool breakdown ("describe returned N tokens, swap M, ...") and a
complete, replayable trajectory — useful for debugging *and* for the blog post's
"chemistree keeps the agent's context small" figure.

The harness joins the two by item id: model-side totals + tool-side breakdown → one row.

### Reporting
- Per-arm, per-track tables; paired per-item deltas (same seeds across arms).
- Headline plots: C vs G at fixed model; haiku-C vs sonnet-G; **context footprint (tokens)
  and cost per item, C vs G** — the efficiency story stands even where quality ties.

## 5. Data prep

### ChemCoTBench (Track A)
- Clone repo / pull HF dataset. Load the Molecule Editing split. Inspect the exact item
  schema in `baseline_and_eval/moledit_eval_demo.ipynb` (input SMILES, instruction, gold,
  scorer) and mirror their scorer so our numbers are comparable to their leaderboard.
- Sample a fixed, seeded subset (e.g. 200 items) for cost control; keep the seed.

### DUD-Z (Track B)
- Download chosen targets. Each gives a prepared receptor + a reference ligand pose.
- Per target: define the docking box from the reference ligand; stage `receptor.pdb`
  (+ pdbqt for smina), `seed_ligand.sdf`. Confirm smina redocks the reference to a sane
  affinity as a sanity check before using the target.
- Seed set: use the reference ligand (and/or a few actives) as starting points.

## 6. Fairness & validity threats (must address)

- **Oracle ≠ optimizer's scorer** — Track B uses Vina-scored smina; chemistree optimizes
  Vinardo. Non-negotiable.
- **Equal budgets** — same max edits/turns and comparable token budget across arms.
- **Anti-gaming guardrails** — property window + pose validity gate (see §4) so affinity
  can't be hacked by mass/lipophilicity.
- **Blind, answer-only scoring** — score the final molecule, never the transcript;
  scorer is agnostic to which arm produced it.
- **Identical staging for G** — arm G must get the *same* receptor/ligand files chemistree
  loads, plus smina on PATH, or the comparison is unfair.
- **Canonicalization** — normalize both sides for match scoring; decide tautomer policy up
  front and apply it uniformly.
- **Prompt parity** — the task prompt is identical across arms; only the tool list and the
  system note about available tools differ.

## 7. Proposed repo layout

```
benchmarks/
  PLAN.md                 <- this file
  README.md               <- how to run, once built
  data/
    chemcotbench/         <- pulled dataset (gitignored; a fetch script restores it)
    dudz/                 <- prepared targets (gitignored; a fetch/prep script restores)
  harness/
    run.py                <- per-item spawn loop, arm config, result capture
    metrics.py            <- parse Claude Code result usage; merge the MCP --trace jsonl
    arms.py               <- N / G / C tool + mcp-config definitions
    score_2d.py           <- Track A scoring (mirror ChemCoTBench scorer)
    score_3d.py           <- Track B smina redock + validity gate + property window
    oracle_smina.py       <- smina wrapper (box from reference ligand, Vina scoring)
  tasks/
    track_a.py            <- ChemCoTBench editing loader -> item prompts
    track_b.py            <- DUD-Z optimization loader -> item prompts + staging
    track_b2.py           <- understanding probes + ground-truth builders
  results/                <- per-run JSONL + summary tables (gitignored)
```
Plus, in the main package (not under `benchmarks/`):
```
src/chemistree/mcp_tools.py     <- shared tool registrations (one source of docstrings)
src/chemistree/mcp_server.py    <- standalone in-process MCP server (chemistree-mcp)
# app/mcp.py refactored to the HTTP backend using mcp_tools.register(...)
```

## 8. Build order (suggested)

1. **Standalone MCP server** (`mcp_tools.py` refactor + `mcp_server.py` + `bind` tool +
   `--tools {2d,all}` launch profile + `--trace` benchmark mode + poetry script). Verify
   by hand: `claude mcp add`
   it, ask the agent to `bind` a SMILES (and separately an SDF + receptor from
   `tests/data/abl1`), describe, and swap a ring; confirm the session mutates and
   `CHEMISTREE_STATE_OUT` holds the result. Add tests locking the verified 2D edit path
   (swap/grow/mutate/remove/undo on a receptor-free session) — it is currently untested.
   Keep `app/mcp.py` behavior identical (the live demo must not regress). Run the suite.
2. **Harness skeleton** (`harness/run.py` + `arms.py`) with arm C only, on a couple of
   hand-made 2D editing items. Get the spawn/capture/score loop working end to end.
3. **Track A**: ChemCoTBench loader + scorer, then arms N and G. Run the seeded subset.
4. **Track B**: DUD-Z prep + smina oracle + validity gate, arm C, then G and N.
5. **Track B2** understanding probes (optional, cheap).
6. **Sonnet** secondary run across arms for the "cheap model catches up" result.
7. Summaries + plots for the blog post.

## 9. Open questions for the next session

- ChemCoTBench exact item schema and official scorer — read the eval notebook and match
  it so results are leaderboard-comparable.
- DUD-Z target shortlist and whether to seed from the reference ligand only or also from
  a few actives.
- Property-window thresholds for the Track B validity gate (start: QED ≥ seed − 0.1,
  MW ≤ seed + 100, cLogP ≤ 5; tune on a dry run).
- Edit/turn budget per item (start: 15 edits, tune).
- Whether to include the optional C+G composed arm.

## 10. Environment / reproduction

- chemistree: conda env `chemistree` (Python 3.14), `poetry install --with app`. Run
  tools via `conda run -n chemistree poetry run ...`. See `CLAUDE.md` for conventions
  (TDD, Google docstrings, black 88). pre-commit runs black/ruff/mypy — run pytest
  yourself.
- Extra benchmark deps (add under a `benchmark` poetry group when building): the
  ChemCoTBench dataset loader (HF `datasets`), and `smina` for Track B (conda-installable
  from bioconda; must be on PATH for arm G and the oracle).
- `claude` CLI must be on PATH (the harness spawns it, as `app/chat.py` already does).
- Keep `data/` and `results/` gitignored; commit fetch/prep scripts, not the payloads.
