# chemistree

Natural-language editing of molecular structures via tree-based fragment representation

## Setup

```bash
conda env create -f conda.yml   # creates the `chemistree` env (Python 3.14 + poetry)
conda activate chemistree
poetry install --with demo      # include the web-demo dependencies
```

## Demo

A local web app with 2D + 3D viewers and a command box, over one shared design
session:

```bash
conda activate chemistree
python -m chemistree.demo        # serves http://127.0.0.1:8000
```

Open the page to see the molecule (aspirin to start) in 2D (RDKit) and 3D
(3Dmol.js) with its fragment breakdown. Drive edits from the command box:

- `swap 3 trifluoromethyl` — replace the fragment at node 3 with a CF3 group
- `add 0 isopropyl ortho carboxyl` — grow an isopropyl ortho to the carboxyl
- `undo 3` — revert node 3's last edit
- `find carboxyl` — list node ids matching a fragment name

Group arguments accept a common name (`isopropyl`, `trifluoromethyl`) or a SMILES
(`[*]C(F)(F)F`). A mirrored Claude Code terminal driving the same session is
planned next.
