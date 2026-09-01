# chemistree

**A tree-based fragment representation that lets an LLM agent understand and edit a
molecule in 2D and 3D.**

chemistree fragments a molecule into a tree of groups. Every atom and group carries a
stable id, and every fragment keeps its own 3D pose in one shared frame. An agent edits
by those ids — swap a ring, grow a substituent, mutate an atom, hop a scaffold — and the
tree keeps the 2D graph and the 3D geometry consistent after each edit (structural edits
go through a re-fragmentation step so the tree always matches building fresh from the
edited molecule). With a receptor loaded, each edit is scored with a docking scoring function,
so the agent can optimize binding while watching drug-likeness, internal strain, and
steric clashes.

The project includes a local web app as a demonstration. The app puts a 2D + 3D viewer
next to a chat box, so you can watch an agent design against a pocket in real time.

## Setup

```bash
conda env create -f conda.yml   # the `chemistree` env (Python 3.14 + poetry)
conda activate chemistree
poetry install --with app       # include the web-app dependencies
pre-commit install              # black / ruff / mypy on commit (once, after cloning)
```

## App

Point the app at a ligand SDF, optionally with a receptor PDB for proximity context:

```bash
conda activate chemistree
chemistree tests/data/abl1/reference.sdf --receptor tests/data/abl1/receptor.pdb        # serves http://127.0.0.1:8000
```

Open the page to see the ligand in 2D and 3D inside the receptor pocket. With a receptor
loaded, the 3D view shows binding-site residues within 6 Å of the ligand as labeled lines
over the ribbon.

Flags:

- `--receptor PDB` — a receptor for proximity context, 3D display, and Vinardo scoring.
- `--model {haiku,sonnet,opus}` — the Claude model the chat agent runs (default `haiku`).
- `--chat-mode {primed,explore}` — `primed` (default) seeds the agent with the group
  listing for snappier edits; `explore` lets it look groups up (its reasoning is visible).
- `--skip-narration` — turn off the agent's one-line rationale before each edit.
- `--host` / `--port` — bind address (default `127.0.0.1:8000`).

## Chat with the molecule

The chat box drives **headless Claude Code**, and the app doubles as an MCP server, so
the agent edits the same session the viewers show. Run the app from the repo root (so
Claude Code finds `.mcp.json`), then talk to it:

> *"remove the chlorine nearest to ASP381"* · *"swap the oxazole ring for a pyridine"* ·
> *"grow a methyl ortho to the amine"* · *"optimize the binding affinity"* · *"undo that"*

The MCP tools run against the live session, so every edit updates the 2D and 3D views in
real time:

- **Inspect** — `describe` (the group listing), `describe_group` (one group's atom
  positions and ring neighbourhood), `smiles`.
- **Edit** — `swap` (replace a group), `grow` (add a group at a hydrogen), `mutate`
  (change one atom's element), `remove` (delete a leaf), `undo`.
- **3D / pocket** — `distance` (each group's distance to a residue), `contacts` (map the
  binding site), `clashes` (report steric overlaps), `minimize` (settle a group about its
  attachment bond to its best-scoring rotamer by the full Vinardo energy).

The group listing ends with a physicochemical profile (MW, cLogP, TPSA, H-bond
donors/acceptors, rotatable bonds, aromatic rings, Fsp3, charge), a structure-alert line
(PAINS/BRENK/NIH), and — with a receptor — the predicted affinity and the ligand's
internal Vinardo energy, all refreshed after every edit. Group arguments accept a common
name (`isopropyl`, `trifluoromethyl`) or a SMILES with one dummy `[*]` per attachment
point (`[*]C(F)(F)F`).

The chat panel has a reasoning toggle (show/hide the agent's spoken rationale) and a
button to copy the whole conversation.

## How it works

- `session.py` — `DesignSession`: the editable molecule. Edits address atoms and groups
  by stable id; undo restores a whole-tree snapshot, so every edit is atomic.
- `tree.py` / `fragment.py` — the fragment tree, port-paired edges, and reconstruction
  of the whole molecule from the per-fragment 3D poses.
- `edits.py` — the geometry of an edit: place a new group onto the retained frame, and
  for a multiport swap or core hop, move each branch to follow the new group.
- `scoring.py` — the Vinardo function: intermolecular (binding) and intramolecular
  (strain) energy, vendored from cmxflow.
- `describe.py` / `annotations.py` — the agent-facing group listing and the
  chemist's-terms atom positions (ortho/meta/para, greek by bond count).
- `app/` — the FastAPI server, the single-page viewer, and the MCP tool server.

## Development

```bash
poetry run pytest        # the test suite (fast, deterministic)
pre-commit run --all-files
```

Tests live in `tests/`, mirroring `src/chemistree/`. Development is test-first; see
`CLAUDE.md` for the coding and documentation conventions.

## Voice (optional)

Install the extra with `poetry install --with voice` (adds faster-whisper, sounddevice,
webrtcvad). Click the 🎤 in the page to listen hands-free: the server captures the mic,
detects when you start and stop speaking, and transcribes each utterance into the chat
box. The **auto-send** checkbox sends each transcript after a short delay you can cancel
by typing; unchecked, it just fills the box for you to send. The first run downloads a
small Whisper model.

For better recognition of chemistry terms, use a larger model:
`CHEMISTREE_WHISPER_MODEL=distil-large-v3` (or `small.en`, `medium.en`, `large-v3`)
before launching — bigger is more accurate but slower on CPU.
