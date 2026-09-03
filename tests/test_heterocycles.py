"""Vendored heterocycle table: name/SMILES lookups and prompt-sync."""

import pathlib
import re

import pytest
from rdkit import Chem

from chemistree.heterocycles import (
    HETEROCYCLES,
    RING_POSITIONS,
    build_ported_ring,
    heterocycle_name,
    heterocycle_smiles,
    named_ring_locants,
    number_ring_system,
    ring_locant_summary,
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


# Ground-truth: the exact gold vs agent SMILES from the two abl1 2D failures the
# substituent-locant report was built to catch (a right ring, a wrong substituent).
_OXAZOLE_GOLD = "Cc1oc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)nc1F"
_OXAZOLE_AGENT = "Cc1nc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)c(=O)n(C)c3n2)oc1F"
_QUINAZOLINE_GOLD = "Cc1cc(Nc2ncc3cc(-c4c(Cl)cccc4Cl)ccc3n2)ccc1F"
_QUINAZOLINE_AGENT = "Cc1cc(Nc2nc(-c3c(Cl)cccc3Cl)c3ccccc3n2)ccc1F"


def test_named_ring_locants_places_oxazole_substituents():
    gold = named_ring_locants(Chem.MolFromSmiles(_OXAZOLE_GOLD), "oxazole")
    agent = named_ring_locants(Chem.MolFromSmiles(_OXAZOLE_AGENT), "oxazole")
    # gold: methyl (C) at 5, F at 4; agent flips them — both attach the aniline N at 2
    assert "N at C2" in gold and "F at C4" in gold and "C at C5" in gold
    assert "C at C4" in agent and "F at C5" in agent
    assert gold != agent


def test_named_ring_locants_places_quinazoline_on_the_right_ring():
    gold = named_ring_locants(Chem.MolFromSmiles(_QUINAZOLINE_GOLD), "quinazoline")
    agent = named_ring_locants(Chem.MolFromSmiles(_QUINAZOLINE_AGENT), "quinazoline")
    # the dichlorophenyl belongs on the benzo ring at C6, not the pyrimidine ring at C4
    assert "C at C6" in gold
    assert "C at C4" in agent and "C at C6" not in agent


def test_named_ring_locants_none_when_absent_or_unvendored():
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert named_ring_locants(benzene, "oxazole") is None  # ring not present
    assert named_ring_locants(benzene, "benzene") is None  # name not vendored


def test_ring_locant_summary_reports_a_port_and_a_baked_substituent():
    # a 2-oxazolyl fragment: the attachment is a port, methyl and F are kept atoms
    summary = ring_locant_summary(Chem.MolFromSmiles("Cc1oc([3*])nc1F"))
    assert summary is not None
    assert "[3*] at C2" in summary  # the port
    assert "F at C4" in summary and "C at C5" in summary  # the baked substituents


def test_ring_locant_summary_none_for_a_carbocycle():
    assert ring_locant_summary(Chem.MolFromSmiles("c1ccccc1")) is None


def _ports_by_locant(mol):
    """Map each dummy port's isotope label to the IUPAC locant it sits on."""
    _, pos = number_ring_system(mol)
    return {
        a.GetIsotope(): pos[a.GetNeighbors()[0].GetIdx()]
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 0
    }


def test_build_ported_ring_places_each_port_at_its_locant():
    ring = build_ported_ring("quinazoline", [(3, 2), (4, 6)])
    assert number_ring_system(ring)[0] == "quinazoline"
    assert _ports_by_locant(ring) == {3: 2, 4: 6}  # [3*] at C2, [4*] at C6


def test_build_ported_ring_bare_port_for_grow():
    ring = build_ported_ring("pyridine", [(0, 3)])
    dummies = [a for a in ring.GetAtoms() if a.GetAtomicNum() == 0]
    assert len(dummies) == 1 and dummies[0].GetIsotope() == 0  # a single bare [*]
    assert _ports_by_locant(ring) == {0: 3}


def test_build_ported_ring_rejects_a_ring_nitrogen_locant():
    # quinazoline position 1 is a ring nitrogen with no free valence for a port
    with pytest.raises(ValueError, match="no free valence"):
        build_ported_ring("quinazoline", [(3, 1)])


def test_build_ported_ring_rejects_an_unnumbered_or_unknown_name():
    with pytest.raises(ValueError, match="not a named ring"):
        build_ported_ring("benzene", [(1, 1)])  # not a vendored ring
    with pytest.raises(ValueError, match="not a named ring"):
        build_ported_ring("purine", [(1, 2)])  # vendored but numbering deferred
