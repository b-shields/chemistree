# chemistree benchmarks

Compares an agent **with** chemistree against the same model **without** it, on
chemical-structure tasks. See [`PLAN.md`](PLAN.md) for the full design (tracks, arms,
scoring, fairness). This README covers how to run the harness and the current results.

## Arms

Same model, same case; only the tool surface differs (see `arms.py`):

- **chemistree** — the chemistree MCP tools only (`bind`, `describe`, `swap`, …).
- **generalist** — `Bash` (rdkit, and smina for 3D) to do it by hand; the honest competitor.
- **naked** — no tools; reasons over the SMILES in the prompt. 2D only — with no 3D access it
  is dropped from both 3D tracks.

## Running

Each case runs one isolated `claude -p` (its own MCP-server child for the chemistree
arm), so cases never share context and every token is attributed to that case. The
**task prompt is byte-identical across arms** — only the appended system prompt differs
(how the arm reaches the molecule, plus optional guidance). Each result row records both
`prompt` and `system_prompt`, so parity is auditable. A 2D/decoration agent ends with a
`FINAL_SMILES:` line; a probe agent ends with `FINAL_ANSWER:`. The runner parses it and
records token usage, cost, turns, and — for the chemistree arm — a per-tool trace.

```bash
conda activate chemistree
python -m benchmarks.run --items benchmarks/cases/abl1_2d.jsonl --arm chemistree
python -m benchmarks.run --items benchmarks/cases/abl1_2d.jsonl --arm naked --dry-run   # print commands only
```

Flags: `--arm {chemistree,generalist,naked}`, `--model` (default `haiku`), `--out`,
`--limit`, `--timeout`, `--no-guidance` (drop the strategy layer — the base-prompt
ablation). The whole abl1 mock-up (all arms, 2D + 3D) runs from
[`test_abl1.sh`](test_abl1.sh).

### smina (3D only)

The 3D track scores with **smina**, which cannot go in the `chemistree` env (its openbabel
dependency conflicts with Python 3.14), so install it in **its own env** and point the
oracle at the binary:

```bash
conda create -n smina -c conda-forge smina -y
export SMINA_BIN="$(conda run -n smina which smina)"   # the runner and generalist arm read this
```

`claude` must also be on `PATH` (the runner spawns it).

## The abl1 worked example

One DUD-Z target, **abl1** (`tests/data/abl1/`), taken all the way through both tracks on
model **haiku**. The crystal ligand is
`Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)ccc1F`. All rows are kept uncompressed under
[`results/test_2d/`](results/test_2d/) and [`results/test_3d/`](results/test_3d/) as the
canonical example. **A timeout drops the row — read a missing row as a failure.**

### Prompt parity

Every arm gets the **same task prompt**; only the system prompt differs. For the
`grow-halogen` case:

```
Add a chlorine to the aniline ring at the open ring carbon that is ortho to the fluorine.
The molecule as SMILES: Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)ccc1F
End your reply with a line exactly of the form:
FINAL_SMILES: <smiles>
```

System prompt per arm (mechanics only, for a 2D case): **naked** — none; **generalist** —
"You have a Bash tool with rdkit (python) available."; **chemistree** — "Edit the molecule
with the chemistree tools. Load it: call bind with molecule "…". When done, call the smiles
tool and report its output."

## Track A — 2D editing

Five NL→SMILES edits over the full crystal ligand (`cases/abl1_2d.jsonl`), one computed
gold each, canonical-SMILES match. Categories: **substitution** (×3), **growing**, **core
hopping**.

Per-arm summary (n = 5):

| Arm | Accuracy | Mean footprint (tok) | Mean output (tok) | Mean turns | Total cost |
|-----|:--------:|---------------------:|------------------:|-----------:|-----------:|
| naked      | 2/5 | 12,186  | 13,794 | 1.0 | $0.419 |
| generalist | 3/5 | 79,606  | 5,393  | 4.0 | $0.315 |
| chemistree | 3/5 | 102,858 | 5,167  | 6.2 | $0.323 |

Per-case correctness:

| Case | Category | naked | generalist | chemistree |
|------|----------|:-----:|:----------:|:----------:|
| sub-halogen (F → Cl)                              | substitution | PASS | PASS | PASS |
| sub-ring-to-pyridine (fluoro-phenyl → pyridine)   | substitution | FAIL | PASS | PASS |
| swap-to-oxazole (fluoro-phenyl → oxazole, keep both subs) | substitution | FAIL | FAIL | FAIL |
| grow-halogen (add ortho Cl)                       | growing      | PASS | PASS | PASS |
| core-hop-quinazoline (pyrimidine core → quinazoline) | core_hopping | FAIL | FAIL | FAIL |

### Reading this

On real drug-like edits (not toys) the tool arms tie at **3/5** and beat **naked at 2/5**.
Two cases defeat every arm — the oxazole swap (a heteroatom shuffle that must keep both
substituents in the right ring positions) and the quinazoline core-hop — hard regiochemistry
that haiku gets wrong regardless of tool surface. The separation that does appear: **naked
is the most expensive arm** ($0.419) despite the smallest context footprint, because it
pours tokens into long free-form reasoning (mean output 13.8k vs ~5k for the tool arms) and
still lands joint-lowest on accuracy. The tool arms spend context reading structure but keep
their output terse.

## Track B1 — 3D scaffold recovery

The crystal ligand is stripped to its **Murcko scaffold** (coords kept, so it stays posed —
`tasks/strip_murcko.py`) and each arm is asked to *re-elaborate it into a potent binder*
(`cases/abl1_3d_decoration.jsonl`). Scored two ways by smina Vinardo — **redock** the final
SMILES (all arms, fair) Δ vs the redocked scaffold baseline (**−11.1**; crystal −12.1), and
**score-only the chemistree `write_pose` SDF** (chemistree only) vs the scored scaffold pose
(**−9.9**) — plus **recovery**, ECFP4 Tanimoto to the crystal (floor **0.328**). Each arm is
run **with and without** the strategy guidance (the base-prompt ablation). Naked is dropped:
it has no 3D access, so its output would be noise.

| Arm | Guidance | Δbinding (redock) | Pose-score (scaffold→final) | Recovery (→crystal) | Turns | Footprint | Cost |
|-----|:--------:|:-----------------:|:---------------------------:|:-------------------:|:-----:|----------:|-----:|
| chemistree | yes | **−1.5** | −9.9 → **−11.8** | 0.328 → 0.333 | 61 | 1.93M | $0.326 |
| chemistree | no  | −0.8 | −9.9 → −10.5 | 0.328 → **0.408** | 66 | 2.13M | $0.365 |
| generalist | yes | — (**timeout → fail**) | — | — | — | — | — |
| generalist | no  | +0.5 (worse) | — | 0.328 → 0.308 (worse) | 12 | 0.33M | $0.191 |

### Reading this (n = 1 — illustrative, not conclusive)

- **Only chemistree finished a valid, improved, well-posed binder** — in both the guided and
  the ablation run. Guided, it improved redock binding by **−1.5**, and the pose it actually
  wrote scored **−11.8** vs the −9.9 scaffold pose: the `write_pose` score-only column catches
  a minimum the blind redock does not always find.
- **The generalist timed out under guidance** (300 s, no `FINAL_SMILES` → failure) and, in
  the ablation, produced a molecule that docked *worse* than the bare scaffold (Δ = +0.5) and
  drifted *away* from the crystal (recovery 0.328 → 0.308). Editing SMILES as text with no
  structural feedback is fragile — exactly what a structure-aware representation is meant to
  prevent.
- **Recovery vs binding pull apart.** chemistree improved binding but explored a *different*
  chemotype, so guided recovery barely moved (0.333); interestingly the ablation recovered
  *more* (0.408). Recovery-to-a-single-crystal is harsh — it penalizes finding a different
  good binder — so read the two columns together.
- **Cost:** chemistree **badly over-explores** (61–66 turns, ~2M tokens, ~$0.35) — the clear
  argument for an edit/turn budget.

## Track B2 — 3D understanding probes

Deterministic Q&A over the **full posed** crystal ligand (`cases/abl1_3d_probes.jsonl`),
answer matched to a computed ground truth. chemistree vs generalist.

| Probe | Expected | chemistree | generalist |
|-------|:--------:|:----------:|:----------:|
| nearest-residue-cl (closest residue to the chlorines) | ASP149 | ASP149 ✓ | ASP149 ✓ |
| cl-contact-count (residues within 4.5 Å of the Cls)   | 9      | 6 ✗       | 9 ✓       |

Both arms read the pocket well; chemistree nailed the nearest residue but **miscounted**
chlorine contacts (6 vs 9), where the generalist's rdkit distance loop got the exact count.

## Caveats

**n = 1** target for the 3D tracks, **haiku**, and a **weak-signal** case (abl1's stripped
groups are small halogens worth ~1 kcal). This one worked example validates the harness and
the metrics end to end; it does not yet support or refute the thesis. Next: more DUD-Z
targets (bigger R-groups), an edit/turn budget to rein in cost, and a sonnet pass. See
`PLAN.md`.
