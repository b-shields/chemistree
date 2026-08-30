"""Physicochemical profile and structure-alert screening for a molecule.

Cheap medicinal-chemistry context read straight off the 2D graph: a
Lipinski/Veber property profile and a scan against RDKit's liability catalogs
(PAINS, BRENK, NIH). Both are pure functions of the molecule, so they hold for a
2D-only session and a 3D one alike, and both ride along in the session's
``describe`` next to the predicted affinity.

The liability catalog is built once at import (its construction is the slow
part) and reused for every scan.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


@dataclass(frozen=True)
class MolProperties:
    """A molecule's medicinal-chemistry property profile.

    Attributes:
        molecular_weight: Molecular weight (g/mol), including implicit hydrogens.
        clogp: Crippen calculated logP, a lipophilicity estimate.
        tpsa: Topological polar surface area (angstrom squared).
        h_bond_donors: Number of hydrogen-bond donors.
        h_bond_acceptors: Number of hydrogen-bond acceptors.
        rotatable_bonds: Number of rotatable bonds, a flexibility measure.
        aromatic_rings: Number of aromatic rings.
        fraction_csp3: Fraction of sp3 carbons, a measure of three-dimensionality.
        formal_charge: Net formal charge of the molecule.
    """

    molecular_weight: float
    clogp: float
    tpsa: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    aromatic_rings: int
    fraction_csp3: float
    formal_charge: int


def compute_properties(mol: Chem.Mol) -> MolProperties:
    """Compute the property profile of a molecule.

    Args:
        mol: The molecule to profile. Explicit hydrogens are ignored, so a posed
            ligand and its plain SMILES give the same profile.

    Returns:
        The molecule's :class:`MolProperties`.
    """
    flat = Chem.RemoveHs(Chem.Mol(mol))
    return MolProperties(
        molecular_weight=Descriptors.MolWt(flat),
        clogp=Crippen.MolLogP(flat),
        tpsa=rdMolDescriptors.CalcTPSA(flat),
        h_bond_donors=rdMolDescriptors.CalcNumHBD(flat),
        h_bond_acceptors=rdMolDescriptors.CalcNumHBA(flat),
        rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(flat),
        aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(flat),
        fraction_csp3=rdMolDescriptors.CalcFractionCSP3(flat),
        formal_charge=Chem.GetFormalCharge(flat),
    )


# Liability catalogs screened for structure alerts, combined once at import.
_ALERT_CATALOGS = (
    FilterCatalogParams.FilterCatalogs.PAINS,
    FilterCatalogParams.FilterCatalogs.BRENK,
    FilterCatalogParams.FilterCatalogs.NIH,
)


def _build_catalog() -> FilterCatalog:
    """Build the combined liability catalog once (construction is the slow part)."""
    params = FilterCatalogParams()
    for catalog in _ALERT_CATALOGS:
        params.AddCatalog(catalog)
    return FilterCatalog(params)


_CATALOG = _build_catalog()


def structure_alerts(mol: Chem.Mol) -> list[str]:
    """Names of the liability alerts a molecule triggers.

    Args:
        mol: The molecule to screen. Explicit hydrogens are ignored.

    Returns:
        The description of each catalog entry the molecule matches (PAINS, BRENK,
        or NIH), in catalog order; empty when the molecule is clean.
    """
    flat = Chem.RemoveHs(Chem.Mol(mol))
    return [match.GetDescription() for match in _CATALOG.GetMatches(flat)]


def profile_markdown(mol: Chem.Mol) -> str:
    """Render a molecule's property profile and alerts as two markdown lines.

    Args:
        mol: The molecule to profile.

    Returns:
        A ``**Properties:**`` line and a ``**Structure alerts:**`` line (the
        matched alert names, or ``none``), for the session's ``describe``.
    """
    props = compute_properties(mol)
    alerts = structure_alerts(mol)
    return (
        f"**Properties:** MW {props.molecular_weight:.0f}, "
        f"cLogP {props.clogp:.1f}, TPSA {props.tpsa:.0f}, "
        f"HBD {props.h_bond_donors}, HBA {props.h_bond_acceptors}, "
        f"RotB {props.rotatable_bonds}, aromatic rings {props.aromatic_rings}, "
        f"Fsp3 {props.fraction_csp3:.2f}, formal charge {props.formal_charge}\n"
        f"**Structure alerts:** {', '.join(alerts) if alerts else 'none'}"
    )
