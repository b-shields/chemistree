# chemistree

Natural-language editing of molecular structures via tree-based fragment representation

## Setup

```bash
conda env create -f conda.yml   # creates the `chemistree` env (Python 3.14 + poetry)
conda activate chemistree
poetry install --with app       # include the web-app dependencies
```

## App

A local web app with 2D + 3D viewers and a chat box, over one design session.
Point it at a ligand SDF, optionally with a receptor PDB for proximity context:

```bash
conda activate chemistree
python -m chemistree.app tests/data/abl1/reference.sdf \
    --receptor tests/data/abl1/receptor.pdb        # serves http://127.0.0.1:8000
```

(Equivalently, the `chemistree` console script after `poetry install --with app`.)
Open the page to see the ligand in 2D (RDKit) and 3D (3Dmol.js) inside the receptor
pocket. With a receptor loaded, the 3D view shows binding-site residues within 6 Å
of the ligand as labeled lines over the ribbon.

## Chat with the molecule

The chat box drives **headless Claude Code** (Haiku by default), and the app
doubles as an MCP server, so the agent edits the same session the viewers show.
Run the app from the repo root (so Claude Code finds `.mcp.json`), then talk to it:

> *"which chlorine is nearest the ASP?"* · *"swap that one for a fluorine"* ·
> *"grow a methyl ortho to it"* · *"undo that"*

The tools (`describe`, `find`, `nearest`, `swap`, `add`, `mutate`, `remove`,
`undo`) run against the live session, so every edit updates the 2D and 3D views
in real time. Group arguments accept a common name (`isopropyl`,
`trifluoromethyl`) or a SMILES (`[*]C(F)(F)F`).
