"""Command-line entry point for the chemistree app.

Serve a ligand (and optional receptor) in the browser::

    chemistree ligand.sdf --receptor receptor.pdb
    chemistree "O=c1[nH]c2nc(Nc3ccccc3)ncc2cc1-c1ccccc1"   # or a SMILES string
"""

from __future__ import annotations

import argparse
import os

import uvicorn
from rdkit import Chem

from chemistree.app import state
from chemistree.app.chat import DEFAULT_MODE, DEFAULT_MODEL, MODES

# Model aliases the ``--model`` flag offers, cheapest and fastest first.
MODEL_CHOICES = ("haiku", "sonnet", "opus")


def read_ligand(spec: str) -> Chem.Mol | None:
    """Read a ligand from an SDF/MOL file path, or parse it as a SMILES string.

    A SMILES has no 3D pose; the session embeds a conformer for the viewer. Use a
    SMILES to poke at the 2D benchmark molecules without a prepared SDF.

    Args:
        spec: A path to an SDF/MOL file, or a SMILES string.

    Returns:
        The molecule (explicit hydrogens kept when read from a file), or None if it
        could not be read or parsed.
    """
    if os.path.exists(spec):
        return Chem.MolFromMolFile(spec, removeHs=False)
    return Chem.MolFromSmiles(spec)


def main() -> None:
    """Parse arguments, load the molecule, and serve the app."""
    parser = argparse.ArgumentParser(description="Edit a molecule in the browser.")
    parser.add_argument(
        "molecule", help="Ligand as an SDF/MOL file path, or a SMILES string."
    )
    parser.add_argument("--receptor", help="Receptor PDB file for proximity context.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--chat-mode",
        choices=sorted(MODES),
        default=DEFAULT_MODE.name,
        help="'primed' seeds the agent with fragments (snappier); 'explore' lets "
        "it look them up (its reasoning is visible).",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
        help="Claude model the chat agent runs: 'haiku' (default, fastest), "
        "'sonnet' (balanced), or 'opus' (most capable).",
    )
    parser.add_argument(
        "--skip-narration",
        action="store_true",
        help="Turn off the agent's one-line rationale before each edit (quieter, "
        "cheaper). Narration is on by default.",
    )
    args = parser.parse_args()

    ligand = read_ligand(args.molecule)
    if ligand is None:
        parser.error(
            f"could not read molecule (as an SDF/MOL file or SMILES): {args.molecule}"
        )
    receptor = None
    if args.receptor:
        receptor = Chem.MolFromPDBFile(args.receptor, removeHs=False, sanitize=False)
        if receptor is None:
            parser.error(f"could not read receptor: {args.receptor}")

    state.configure(
        ligand,
        receptor,
        mode=MODES[args.chat_mode],
        model=args.model,
        narrate=not args.skip_narration,
    )

    from chemistree.app.server import app

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
