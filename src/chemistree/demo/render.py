"""Render a design session into viewer artifacts (2D SVG + 3D molblock)."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from chemistree.session import DesignSession


def render_state(session: DesignSession) -> dict:
    """Viewer artifacts for the session's current molecule.

    Args:
        session: The session to render.

    Returns:
        A dict with the canonical SMILES, a 2D SVG, a 3D molblock, and the markdown
        fragment summary.
    """
    mol = session.molecule()
    molblock = Chem.MolToMolBlock(mol)

    flat = Chem.RemoveHs(Chem.Mol(mol))
    AllChem.Compute2DCoords(flat)
    drawer = rdMolDraw2D.MolDraw2DSVG(440, 360)
    drawer.DrawMolecule(flat)
    drawer.FinishDrawing()

    return {
        "smiles": session.smiles(),
        "svg": drawer.GetDrawingText(),
        "molblock": molblock,
        "describe": session.describe(),
    }
