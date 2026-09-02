"""Vendored heterocycle table: name/SMILES lookups and prompt-sync."""

import pathlib
import re

from rdkit import Chem

from chemistree.heterocycles import (
    HETEROCYCLES,
    heterocycle_name,
    heterocycle_smiles,
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
