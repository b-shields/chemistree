"""Command-line entry point for the chemistree app.

Serve a ligand (and optional receptor) in the browser::

    chemistree ligand.sdf --receptor receptor.pdb
"""

from __future__ import annotations

import argparse

import uvicorn
from rdkit import Chem

from chemistree.app import state


def main() -> None:
    """Parse arguments, load the molecule, and serve the app."""
    parser = argparse.ArgumentParser(description="Edit a molecule in the browser.")
    parser.add_argument("molecule", help="Ligand SDF/MOL file.")
    parser.add_argument("--receptor", help="Receptor PDB file for proximity context.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    ligand = Chem.MolFromMolFile(args.molecule, removeHs=False)
    if ligand is None:
        parser.error(f"could not read molecule: {args.molecule}")
    receptor = None
    if args.receptor:
        receptor = Chem.MolFromPDBFile(args.receptor, removeHs=False, sanitize=False)
        if receptor is None:
            parser.error(f"could not read receptor: {args.receptor}")

    state.configure(ligand, receptor)

    from chemistree.app.server import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
