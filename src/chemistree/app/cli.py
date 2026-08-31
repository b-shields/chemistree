"""Command-line entry point for the chemistree app.

Serve a ligand (and optional receptor) in the browser::

    chemistree ligand.sdf --receptor receptor.pdb
"""

from __future__ import annotations

import argparse

import uvicorn
from rdkit import Chem

from chemistree.app import state
from chemistree.app.chat import DEFAULT_MODE, DEFAULT_MODEL, MODES

# Model aliases the ``--model`` flag offers, cheapest and fastest first.
MODEL_CHOICES = ("haiku", "sonnet", "opus")


def main() -> None:
    """Parse arguments, load the molecule, and serve the app."""
    parser = argparse.ArgumentParser(description="Edit a molecule in the browser.")
    parser.add_argument("molecule", help="Ligand SDF/MOL file.")
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

    ligand = Chem.MolFromMolFile(args.molecule, removeHs=False)
    if ligand is None:
        parser.error(f"could not read molecule: {args.molecule}")
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
