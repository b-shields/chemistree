"""Vendored parent heteroaromatic rings: name and canonical SMILES.

The single source of truth for the named rings the agent may ask about. Backs the
``matches`` tool, so a ring can be checked by name and the ring's name is echoed
back. The same list is shown to the agent in the guidance prompts; a test keeps the
two in sync. Extend freely.
"""

from __future__ import annotations

from rdkit import Chem

# name -> parent-ring SMILES (rdkit canonical). Names are lower-case for lookup.
HETEROCYCLES: dict[str, str] = {
    # 5-membered, one heteroatom
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "pyrrole": "c1cc[nH]c1",
    # 5-membered, two heteroatoms
    "pyrazole": "c1cn[nH]c1",
    "imidazole": "c1c[nH]cn1",
    "isoxazole": "c1cnoc1",
    "oxazole": "c1cocn1",
    "isothiazole": "c1cnsc1",
    "thiazole": "c1cscn1",
    # 5-membered, three or more heteroatoms
    "1,2,3-triazole": "c1cn[nH]n1",
    "1,2,4-triazole": "c1nc[nH]n1",
    "tetrazole": "c1nn[nH]n1",
    "1,2,4-oxadiazole": "c1ncon1",
    "1,3,4-oxadiazole": "c1nnco1",
    "1,3,4-thiadiazole": "c1nncs1",
    # 6-membered
    "pyridine": "c1ccncc1",
    "pyridazine": "c1ccnnc1",
    "pyrimidine": "c1cncnc1",
    "pyrazine": "c1cnccn1",
    "1,3,5-triazine": "c1ncncn1",
    # fused 5-6
    "indole": "c1ccc2[nH]ccc2c1",
    "indazole": "c1ccc2[nH]ncc2c1",
    "benzimidazole": "c1ccc2[nH]cnc2c1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",
    "benzoxazole": "c1ccc2ocnc2c1",
    "benzothiazole": "c1ccc2scnc2c1",
    "7-azaindole": "c1cnc2[nH]ccc2c1",
    "purine": "c1ncc2[nH]cnc2n1",
    "pyrrolotriazine": "c1cc2cncnn2c1",  # pyrrolo[2,1-f][1,2,4]triazine
    # fused 6-6
    "quinoline": "c1ccc2ncccc2c1",
    "isoquinoline": "c1ccc2cnccc2c1",
    "quinazoline": "c1ccc2ncncc2c1",
    "quinoxaline": "c1ccc2nccnc2c1",
    "1,8-naphthyridine": "c1cnc2ncccc2c1",
}

# canonical SMILES -> name, built once for layout-independent reverse lookup.
_BY_CANONICAL: dict[str, str] = {}
for _name, _smiles in HETEROCYCLES.items():
    _mol = Chem.MolFromSmiles(_smiles)
    if _mol is not None:
        _BY_CANONICAL[Chem.MolToSmiles(_mol)] = _name


def heterocycle_smiles(name: str) -> str | None:
    """The parent-ring SMILES for a heterocycle name, or None if unknown.

    Args:
        name: A heterocycle name (case-insensitive), e.g. ``"quinazoline"``.

    Returns:
        The ring SMILES, or None when the name is not in the table.
    """
    return HETEROCYCLES.get(name.strip().lower())


def heterocycle_name(smiles: str) -> str | None:
    """The heterocycle name whose parent ring equals a SMILES, or None.

    Args:
        smiles: A SMILES string; canonicalized before the lookup.

    Returns:
        The matching heterocycle name, or None if it is not a known parent ring.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _BY_CANONICAL.get(Chem.MolToSmiles(mol))
