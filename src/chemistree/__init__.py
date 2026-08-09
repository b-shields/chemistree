"""Natural-language editing of molecules via tree-based fragment representation."""

from chemistree.fragment import Fragment, Port
from chemistree.fragmenter import fragment, should_break
from chemistree.tree import Edge, FragmentNode, FragmentTree

__version__ = "0.1.0"

__all__ = [
    "Fragment",
    "Port",
    "Edge",
    "FragmentNode",
    "FragmentTree",
    "fragment",
    "should_break",
]
