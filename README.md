# chemistree

Natural-language editing of molecular structures via tree-based fragment representation

## Setup

```bash
conda env create -f conda.yml   # creates the `chemistree` env (Python 3.14 + poetry)
conda activate chemistree
poetry install --with app       # include the web-app dependencies
```

## App

A local web app with 2D + 3D viewers and a command box, over one design session.
Point it at a ligand SDF, optionally with a receptor PDB for proximity context:

```bash
conda activate chemistree
python -m chemistree.app tests/data/abl1/reference.sdf \
    --receptor tests/data/abl1/receptor.pdb        # serves http://127.0.0.1:8000
```

(Equivalently, the `chemistree` console script after `poetry install --with app`.)
Open the page to see the ligand in 2D (RDKit) and 3D (3Dmol.js) inside the receptor
pocket, with its fragment breakdown. Drive edits from the command box:

- `find chloro` — list node ids matching a fragment name
- `nearest chloro ASP` — the chlorine closest to an ASP residue (needs a receptor)
- `swap 7 fluoro` — replace the fragment at node 7 with a fluorine
- `add 2 methyl ortho chloro` — grow a methyl ortho to a chloro
- `undo 7` — revert node 7's last edit

Group arguments accept a common name (`isopropyl`, `trifluoromethyl`) or a SMILES
(`[*]C(F)(F)F`). With a receptor loaded, the 3D view shows binding-site side chains
within 5 Å of the ligand as labeled lines.

## Chat with Claude Code

The app doubles as an MCP server, so Claude Code can edit the same molecule and you
watch the viewers update live. With the app running (above):

```bash
conda activate chemistree
claude            # in the repo root; approve the "chemistree" MCP server when asked
```

Then talk to it about the molecule — e.g. *"which chlorine is nearest the ASP?"*,
*"swap that one for a fluorine"*, *"grow a methyl ortho to it"*. The tools
(`describe`, `find`, `nearest`, `swap`, `add`, `undo`) run against the same session
the browser shows, so every edit updates the 2D and 3D views in real time.
