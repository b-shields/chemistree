"""Render a design session into viewer artifacts (2D SVG + 3D molblock)."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from chemistree.session import DesignSession

# Square pixel sizes for the main depiction and the trace thumbnails.
_MAIN_SIZE = 380
_THUMB_SIZE = 150


def render_state(session: DesignSession) -> dict:
    """Viewer artifacts for the session's current molecule.

    Args:
        session: The session to render.

    Returns:
        A dict with the current molecule's SMILES, 2D SVG, and 3D molblock, the
        markdown fragment summary, and a ``trace`` of one thumbnail per edit step.
    """
    state = render_molecule(session.molecule())
    state["describe"] = session.describe()
    state["trace"] = _trace(session)
    return state


def render_molecule(mol: Chem.Mol) -> dict:
    """Viewer artifacts for a single molecule (the current one, or a trace step).

    Args:
        mol: A molecule with a 3D conformer.

    Returns:
        A dict with the molecule's canonical SMILES, a 2D SVG, and a 3D molblock.
    """
    return {
        "smiles": _smiles(mol),
        "svg": _depiction_svg(mol, _MAIN_SIZE),
        "molblock": Chem.MolToMolBlock(mol),
    }


def _trace(session: DesignSession) -> list[dict]:
    """A thumbnail per distinct molecule visited, oldest first.

    Args:
        session: The session whose history to depict.

    Returns:
        One ``{"smiles", "svg", "affinity"}`` entry per molecule in the session
        history; the last entry is the current molecule. ``affinity`` is the
        Vinardo score, or None when there is no posed receptor.
    """
    return [
        {
            "smiles": _smiles(mol),
            "svg": _depiction_svg(mol, _THUMB_SIZE),
            "affinity": session.affinity(mol),
        }
        for mol in session.history()
    ]


def _smiles(mol: Chem.Mol) -> str:
    """Canonical SMILES of a molecule, without explicit hydrogens."""
    return str(Chem.MolToSmiles(Chem.RemoveHs(mol)))


def _depiction_svg(mol: Chem.Mol, size: int) -> str:
    """Draw a molecule to a dark-mode 2D SVG of a square size.

    A clean 2D layout is generated (any input conformer is ignored), so the
    depiction is textbook-style rather than a projection of the 3D pose.

    Args:
        mol: Molecule to draw.
        size: Width and height of the square SVG, in pixels.

    Returns:
        The SVG document as a string.
    """
    flat = Chem.RemoveHs(Chem.Mol(mol))
    rdCoordGen.AddCoords(flat)
    rdDepictor.StraightenDepiction(flat)
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    rdMolDraw2D.SetDarkMode(drawer)
    drawer.DrawMolecule(flat)
    drawer.FinishDrawing()
    return str(drawer.GetDrawingText())
