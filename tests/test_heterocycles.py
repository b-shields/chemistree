"""Vendored heterocycle table: name/SMILES lookups and prompt-sync."""

import pathlib
import re

from rdkit import Chem

from chemistree.heterocycles import (
    HETEROCYCLES,
    RING_POSITIONS,
    heterocycle_name,
    heterocycle_smiles,
    number_ring_system,
)

_MEDCHEM = pathlib.Path("src/chemistree/mcp/prompts/medchem.md")


def _canon(smiles: str) -> str:
    return str(Chem.CanonSmiles(smiles))


def test_forward_lookup_is_case_insensitive():
    assert _canon(heterocycle_smiles("quinazoline")) == _canon("c1ccc2ncncc2c1")
    assert heterocycle_smiles("QUINAZOLINE") == heterocycle_smiles("quinazoline")


def test_reverse_lookup_names_a_ring_from_any_layout():
    # A quinoxaline written a different way still resolves to "quinoxaline".
    assert heterocycle_name("c1cnc2ccccc2n1") == "quinoxaline"
    assert heterocycle_name(HETEROCYCLES["quinazoline"]) == "quinazoline"


def test_unknown_lookups_return_none():
    assert heterocycle_smiles("not-a-ring") is None
    assert heterocycle_name("CCO") is None  # ethanol is no parent heteroaromatic


def test_dict_and_prompt_table_stay_in_sync():
    # The vendored dict is the source of truth; the guidance table must mirror it,
    # so the agent reads exactly the SMILES the matches tool resolves.
    text = _MEDCHEM.read_text()
    prompt_smiles = {_canon(m) for m in re.findall(r"^- .+: `([^`]+)`$", text, re.M)}
    dict_smiles = {_canon(s) for s in HETEROCYCLES.values()}
    assert dict_smiles == prompt_smiles


def _element_at(name, position):
    """The element symbol of the atom bearing a given IUPAC position."""
    m = Chem.MolFromSmiles(HETEROCYCLES[name])
    pos = RING_POSITIONS[name]
    idx = next(a.GetIdx() for a in m.GetAtoms() if pos[a.GetIdx()] == position)
    return m.GetAtomWithIdx(idx).GetSymbol()


def test_ring_positions_align_with_the_smiles_atom_order():
    # Length matches the atom count, and known positions land on the right element --
    # so a miscounted tuple or a rewritten SMILES cannot ship silently.
    for name, pos in RING_POSITIONS.items():
        assert len(pos) == Chem.MolFromSmiles(HETEROCYCLES[name]).GetNumAtoms()
    assert (_element_at("oxazole", 1), _element_at("oxazole", 3)) == ("O", "N")
    assert _element_at("thiazole", 1) == "S"
    assert _element_at("pyridine", 1) == "N"
    quinazoline = (_element_at("quinazoline", i) for i in (1, 2, 3))
    assert tuple(quinazoline) == ("N", "C", "N")  # C2 between the two ring nitrogens


def test_number_ring_system_quinazoline_c2_sits_between_the_two_nitrogens():
    m = Chem.MolFromSmiles("c1ccc2ncncc2c1")
    name, pos = number_ring_system(m)
    assert name == "quinazoline"
    c2 = next(i for i, p in pos.items() if p == 2)
    ring_n = sum(n.GetSymbol() == "N" for n in m.GetAtomWithIdx(c2).GetNeighbors())
    assert ring_n == 2


def test_number_ring_system_distinguishes_a_swapped_substituent():
    gold = Chem.MolFromSmiles("Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)ccc3n2)ccc1F")
    haiku = Chem.MolFromSmiles("Cc1cc(Nc2ccc3cnc(-c4c(Cl)cccc4Cl)nc3c2)ccc1F")

    def dichlorophenyl_position(m):
        _, pos = number_ring_system(m)
        carbon = m.GetSubstructMatch(Chem.MolFromSmarts("c(-c1c(Cl)cccc1Cl)"))[0]
        return pos[carbon]

    assert dichlorophenyl_position(gold) == 6
    assert dichlorophenyl_position(haiku) == 2  # the inversion is visible


def test_number_ring_system_prefers_the_largest_ring():
    # a quinoline group numbers as quinoline, not its pyridine sub-ring
    name, _ = number_ring_system(Chem.MolFromSmiles("c1ccc2ncccc2c1"))
    assert name == "quinoline"


def test_number_ring_system_gives_a_lone_substituent_the_lowest_locant():
    def methyl_position(smiles):
        m = Chem.MolFromSmiles(smiles)
        _, pos = number_ring_system(m)
        carbon = next(
            a.GetIdx()
            for a in m.GetAtoms()
            if a.GetIsAromatic()
            and any(
                n.GetSymbol() == "C" and not n.GetIsAromatic() for n in a.GetNeighbors()
            )
        )
        return pos[carbon]

    assert methyl_position("Cc1ncccn1") == 2  # 2-methylpyrimidine
    assert methyl_position("Cc1ccncn1") == 4  # 4/6 symmetric -> lowest locant 4
    assert methyl_position("Cc1cncnc1") == 5  # 5-methylpyrimidine


def test_number_ring_system_none_for_a_carbocycle():
    assert number_ring_system(Chem.MolFromSmiles("c1ccccc1")) is None


def test_number_ring_system_breaks_symmetric_ties_deterministically():
    # The same molecule written two ways numbers the same: on a symmetric ring the
    # lower-atomic-number substituent takes the lower locant, whatever the SMILES order.
    def halogen_positions(smiles):
        m = Chem.MolFromSmiles(smiles)
        _, pos = number_ring_system(m)
        out = {}
        for atom in m.GetAtoms():
            p = pos.get(atom.GetIdx())
            if not (atom.GetIsAromatic() and p):
                continue
            for nb in atom.GetNeighbors():
                if nb.GetSymbol() in ("F", "Cl"):
                    out[nb.GetSymbol()] = p
        return out

    assert halogen_positions("Fc1cccc(Cl)n1") == halogen_positions("Clc1cccc(F)n1")
    assert halogen_positions("Fc1cccc(Cl)n1") == {"F": 2, "Cl": 6}
