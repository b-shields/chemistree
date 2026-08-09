"""Agent-facing tree annotations and their serializations."""

import json

from chemistree import annotate, fragment, prepare_molecule, swap


def _tree(smiles: str, *, three_d: bool = True):
    return fragment(prepare_molecule(smiles, three_d=three_d))


def _by_name(annotation, name: str):
    return next(n for n in annotation.nodes if n.name == name)


def test_toluene_nodes_named_and_classified():
    ann = annotate(_tree("Cc1ccccc1", three_d=False))
    assert len(ann.nodes) == 2

    methyl = _by_name(ann, "methyl")
    assert methyl.role == "leaf"
    assert methyl.classification == "alkyl"
    assert methyl.is_ring is False
    assert methyl.formula == "CH3"

    phenyl = _by_name(ann, "phenyl")
    assert phenyl.classification == "aromatic"
    assert phenyl.is_ring is True


def test_neighbors_link_by_port():
    ann = annotate(_tree("Cc1ccccc1", three_d=False))
    methyl = _by_name(ann, "methyl")
    phenyl = _by_name(ann, "phenyl")
    (neighbor,) = methyl.neighbors
    assert neighbor.node_id == phenyl.id
    assert neighbor.name == "phenyl"
    assert neighbor.port in phenyl.ports


def test_internal_ring_is_a_scaffold():
    ann = annotate(_tree("Cc1cc(C)cc(C)c1", three_d=False))  # 1,3,5-trisubstituted
    ring = next(n for n in ann.nodes if n.is_ring)
    assert ring.role == "scaffold"
    assert len(ring.ports) == 3
    assert sum(1 for n in ann.nodes if n.role == "leaf") == 3


def test_atom_detail_is_opt_in():
    plain = annotate(_tree("CCc1ccccc1", three_d=False))
    assert all(n.atoms is None for n in plain.nodes)

    detailed = annotate(_tree("CCc1ccccc1", three_d=False), atoms=True)
    ethyl = next(n for n in detailed.nodes if n.name == "ethyl")
    terminal = max(ethyl.atoms, key=lambda a: a.num_h)
    assert terminal.element == "C"
    assert terminal.num_h == 3  # a CH3 growth site


def test_to_dict_is_json_serializable():
    ann = annotate(_tree("Cc1ccccc1", three_d=False))
    restored = json.loads(json.dumps(ann.to_dict()))
    assert {n["name"] for n in restored["nodes"]} == {"methyl", "phenyl"}


def test_to_markdown_mentions_name_and_smiles():
    md = annotate(_tree("Cc1ccccc1", three_d=False)).to_markdown()
    assert "methyl" in md
    assert "[1*]C" in md or "[*]C" in md


def test_to_markdown_falls_back_to_classification_never_none():
    # An unnamed fragment (the boronic acid) shows its class, never a bare "?".
    md = annotate(_tree("OB(O)c1ccccc1", three_d=False)).to_markdown()
    assert "?" not in md
    assert "other" in md


def test_node_id_stable_across_swap():
    tree = _tree("Cc1ccccc1")
    before = annotate(tree)
    methyl = _by_name(before, "methyl")

    swap(tree.node(methyl.id), "[*]C(F)(F)F")
    after = annotate(tree)

    same = next(n for n in after.nodes if n.id == methyl.id)
    assert same.name == "trifluoromethyl"
