"""Fragment and attachment-point schema."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class Port:
    """An attachment point where a bond was broken.

    Attributes:
        label: Shared integer pairing this port with its partner across the tree.
        dummy_idx: Index of the dummy (`*`) atom marking the attachment.
        anchor_idx: Index of the heavy atom the original bond connected to.
        bond_type: Bond order to restore when the port is fused.
    """

    label: int
    dummy_idx: int
    anchor_idx: int
    bond_type: Chem.BondType


class Fragment:
    """A single fragment snapshot: an RDKit molecule with dummy-capped ports.

    Ports are derived from dummy atoms whose isotope carries the pairing label.
    """

    def __init__(self, mol: Chem.Mol):
        self.mol = mol

    @property
    def ports(self) -> list[Port]:
        """Attachment points, one per dummy atom."""
        ports = []
        for atom in self.mol.GetAtoms():
            if atom.GetAtomicNum() != 0:
                continue
            neighbors = atom.GetNeighbors()
            anchor = neighbors[0]
            bond = self.mol.GetBondBetweenAtoms(atom.GetIdx(), anchor.GetIdx())
            ports.append(
                Port(
                    label=atom.GetIsotope(),
                    dummy_idx=atom.GetIdx(),
                    anchor_idx=anchor.GetIdx(),
                    bond_type=bond.GetBondType(),
                )
            )
        return ports

    @property
    def smiles(self) -> str:
        """Canonical SMILES, including dummy atoms."""
        return str(Chem.MolToSmiles(self.mol))

    def __repr__(self) -> str:
        return f"Fragment({self.smiles!r})"
