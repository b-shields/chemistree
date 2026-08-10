"""Natural-language editing of molecules via tree-based fragment representation."""

from chemistree.annotations import (
    AtomAnnotation,
    NeighborRef,
    NodeAnnotation,
    TreeAnnotation,
    annotate,
)
from chemistree.chem import prepare_molecule
from chemistree.edits import add_substituent, swap
from chemistree.fragment import Fragment, Port
from chemistree.fragmenter import fragment, should_break
from chemistree.naming import classify_fragment, name_fragment
from chemistree.session import DesignSession
from chemistree.tree import Edge, FragmentNode, FragmentTree

__version__ = "0.1.0"

__all__ = [
    "Fragment",
    "Port",
    "DesignSession",
    "Edge",
    "FragmentNode",
    "FragmentTree",
    "TreeAnnotation",
    "NodeAnnotation",
    "AtomAnnotation",
    "NeighborRef",
    "add_substituent",
    "annotate",
    "classify_fragment",
    "fragment",
    "name_fragment",
    "prepare_molecule",
    "should_break",
    "swap",
]
