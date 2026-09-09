"""Generate the DUD-Z benchmark case files: 2D edits, 3D decoration, 3D probes.

Golds and probe answers are **computed and self-checked with RDKit / numpy, independent
of the chemistree tools under test** — a bug in those tools cannot hide in the answer
key. The Murcko scaffold decoration inputs are produced with the same
``benchmarks.tasks.strip_murcko`` the runner uses at run time, so the materialized seed
matches what the agent is given.

Run from the repo root: ``python -m benchmarks.gen_cases``. Reads
``benchmarks/data/manifest.jsonl`` and writes ``benchmarks/cases/<target>_2d.jsonl``,
``<target>_3d_decoration.jsonl``, ``<target>_3d_probes.jsonl`` and
``benchmarks/data/<target>/scaffold.sdf``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from benchmarks.tasks import strip_murcko

DATA = Path("benchmarks/data")
CASES = Path("benchmarks/cases")
MANIFEST = {
    j["target"].lower(): j
    for j in (
        json.loads(line) for line in (DATA / "manifest.jsonl").read_text().splitlines()
    )
}
TARGETS = [
    "egfr",
    "aa2ar",
    "andr",
    "hivpr",
    "fa10",
    "hdac8",
    "parp1",
    "hs90a",
    "ada",
    "nram",
]

_AA = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HID",
    "HIE",
    "HIP",
    "CYX",
    "SEC",
    "MSE",
}


# --- RDKit edit primitives (compute a canonical gold from a flat start) --------------
def canon(smiles: str) -> str:
    """Canonical SMILES."""
    return str(Chem.CanonSmiles(smiles))


def flat(smiles: str) -> str:
    """The molecule's canonical connectivity SMILES, stereochemistry removed."""
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    return str(Chem.MolToSmiles(mol))


def formula(smiles: str) -> str:
    """Molecular formula, for the self-check report."""
    return str(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)))


def _out(rw: Chem.RWMol) -> str:
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return str(Chem.MolToSmiles(mol))


def only(mol: Chem.Mol, smarts: str) -> int:
    """The first atom of the sole match of ``smarts`` (asserts exactly one match)."""
    ms = mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))
    assert len(ms) == 1, f"{smarts!r}: {len(ms)} matches, need 1"
    return int(ms[0][0])


def mutate(smiles: str, idx: int, num: int, *, arom_no_h: bool = False) -> str:
    """Change the element of one atom (halogen swap, aromatic C->N, C->O, O->F)."""
    rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
    atom = rw.GetAtomWithIdx(idx)
    atom.SetAtomicNum(num)
    if arom_no_h:
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return _out(rw)


def aza(smiles: str, idx: int) -> str:
    """Turn the aromatic C-H at idx into a pyridine-type nitrogen."""
    return mutate(smiles, idx, 7, arom_no_h=True)


def attach(smiles: str, idx: int, group: str) -> str:
    """Attach ``group`` (one ``[*]`` port) at heavy-atom idx, which must bear an H."""
    rw = Chem.RWMol(Chem.MolFromSmiles(smiles))
    atom = rw.GetAtomWithIdx(idx)
    hs = atom.GetTotalNumHs()
    assert hs > 0, f"atom {idx} has no H"
    frag = Chem.MolFromSmiles(group)
    star = next(a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0)
    join = frag.GetAtomWithIdx(star).GetNeighbors()[0].GetIdx()
    off = rw.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(rw, frag))
    combo.AddBond(idx, off + join, Chem.BondType.SINGLE)
    combo.RemoveAtom(off + star)
    grown = combo.GetAtomWithIdx(idx)
    grown.SetNumExplicitHs(hs - 1)
    grown.SetNoImplicit(True)
    return _out(combo)


def ring_ch(
    mol: Chem.Mol, anchor: int, dist: int, *, aromatic: bool = True
) -> list[int]:
    """C-H atoms sharing a ring with ``anchor``, at graph distance ``dist`` from it."""
    dmat = Chem.GetDistanceMatrix(mol)
    anchor_rings = [set(r) for r in mol.GetRingInfo().AtomRings() if anchor in r]
    out = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() < 1:
            continue
        if aromatic and not atom.GetIsAromatic():
            continue
        if int(dmat[anchor][i]) == dist and any(i in r for r in anchor_rings):
            out.append(i)
    return out


def attach_ring(smiles: str, cands: list[int], group: str) -> str:
    """Attach ``group`` at ``cands``; require all candidates give the same gold."""
    assert cands, "no candidate position"
    golds = {canon(attach(smiles, i, group)) for i in cands}
    assert len(golds) == 1, f"ambiguous position: {len(golds)} distinct golds"
    return attach(smiles, cands[0], group)


def react(smiles: str, smarts: str) -> str:
    """Apply a single-product reaction SMARTS (e.g. carboxyl -> tetrazole)."""
    from rdkit.Chem import AllChem

    rxn = AllChem.ReactionFromSmarts(smarts)
    outs = set()
    for (prod,) in rxn.RunReactants((Chem.MolFromSmiles(smiles),)):
        try:
            Chem.SanitizeMol(prod)
            outs.add(Chem.MolToSmiles(prod))
        except ValueError:
            pass
    assert len(outs) == 1, f"reaction gave {len(outs)} products"
    return str(outs.pop())


def _c(cid: str, category: str, instruction: str, gold: str) -> dict:
    return {
        "id": cid,
        "category": category,
        "instruction": instruction,
        "gold": canon(gold),
    }


def edits_2d(target: str) -> list[dict]:
    """The authored 2D edits for a target; each gold is computed from the flat start."""
    s = flat(MANIFEST[target]["smiles"])
    m = Chem.MolFromSmiles(s)
    if target == "egfr":
        fc = only(m, "[c;H0](F)")
        pip_n = only(m, "[NX3;H0]1[CH2][CH2][CH2][CH2][CH2]1")
        return [
            _c(
                "grow-cyclopropyl",
                "growing",
                "Add a cyclopropyl on the benzyl ring, para to the fluorine.",
                attach_ring(s, ring_ch(m, fc, 3), "[*]C1CC1"),
            ),
            _c(
                "core-piperazine",
                "core_hopping",
                "Change the terminal piperidine to a piperazine (replace the ring CH2 "
                "para to the ring nitrogen with an NH).",
                mutate(s, ring_ch(m, pip_n, 3, aromatic=False)[0], 7),
            ),
            _c(
                "core-triazine",
                "core_hopping",
                "Aza-substitute the aminopyrimidine CH between the two ring nitrogens, "
                "making a 1,2,3-triazine.",
                aza(s, only(m, "[cH1](:n):n")),
            ),
        ]
    if target == "aa2ar":
        oh = only(m, "[OX2H]")
        phenol_c = m.GetAtomWithIdx(oh).GetNeighbors()[0].GetIdx()
        return [
            _c(
                "grow-nitrile",
                "growing",
                "Add a nitrile to the phenol ring, ortho to the hydroxyl.",
                attach_ring(s, ring_ch(m, phenol_c, 1), "[*]C#N"),
            ),
            _c(
                "core-benzofuran",
                "core_hopping",
                "Fuse a benzene onto the furan to make a benzofuran, keeping the "
                "attachment at the 2-position.",
                react(s, "[c:1]1[cH][cH][cH][o:2]1>>[c:1]1cc2ccccc2[o:2]1"),
            ),
            _c(
                "sub-methoxy",
                "substitution",
                "O-methylate the phenol to a methyl ether (methoxy).",
                attach(s, oh, "[*]C"),
            ),
        ]
    if target == "andr":
        return [
            _c(
                "grow-methyl",
                "growing",
                "Add a methyl at C4, the vinyl carbon of the A-ring enone.",
                attach(s, only(m, "[CH1]=[C]"), "[*]C"),
            ),
            _c(
                "sub-amine",
                "substitution",
                "Replace the 17-hydroxyl with a primary amine.",
                mutate(s, only(m, "[OX2H]"), 7),
            ),
        ]
    if target == "hivpr":
        s_atom = only(m, "[SX4](=O)(=O)")
        sulfonyl_ph = [
            n.GetIdx()
            for n in m.GetAtomWithIdx(s_atom).GetNeighbors()
            if n.GetIsAromatic()
        ][0]
        return [
            _c(
                "grow-methoxy",
                "growing",
                "Add a methoxy para on the benzenesulfonyl ring.",
                attach_ring(s, ring_ch(m, sulfonyl_ph, 3), "[*]OC"),
            ),
            _c(
                "core-naphthalene",
                "core_hopping",
                "Expand the N-benzyl phenyl to a 2-naphthyl (fuse a second benzene "
                "ring), keeping the methylene attachment.",
                react(
                    s,
                    "[CH2:1][c:2]1[cH][cH][cH][cH][cH]1>>[CH2:1][c:2]1ccc2ccccc2c1",
                ),
            ),
        ]
    if target == "fa10":
        return [
            _c(
                "sub-bromo",
                "substitution",
                "Replace the chlorine on the naphthalene with a bromine.",
                mutate(s, only(m, "[Cl]"), 35),
            ),
            _c(
                "sub-nmethyl",
                "substitution",
                "N-methylate the lactam N-H of the diazepanone ring.",
                attach(s, only(m, "[NX3;H1][CX3]=O"), "[*]C"),
            ),
        ]
    if target == "hdac8":
        benzyl_c = only(m, "[CH2]c1ccccc1")
        phenyl = [
            n.GetIdx()
            for n in m.GetAtomWithIdx(benzyl_c).GetNeighbors()
            if n.GetIsAromatic()
        ][0]
        return [
            _c(
                "grow-oxetane",
                "growing",
                "Add an oxetan-3-yl para on the phenyl of the phenylacetyl group.",
                attach_ring(s, ring_ch(m, phenyl, 3), "[*]C1COC1"),
            ),
            _c(
                "core-thiophene",
                "core_hopping",
                "Replace the phenacyl phenyl with a 2-thienyl (contract the benzene to "
                "a thiophene), keeping the methylene attachment.",
                react(
                    s,
                    "[CH2:1][c:2]1[cH][cH][cH][cH][cH]1>>[CH2:1][c:2]1[cH][cH][cH]s1",
                ),
            ),
        ]
    if target == "parp1":
        fc = only(m, "[c;H0](F)")
        pip_n = only(m, "[NX3;H1]1[CH2][CH2][CH2][CH2][CH1]1")
        return [
            _c(
                "grow-fluoro",
                "growing",
                "Add a second fluorine on the fluorophenyl ring, para to the fluorine.",
                attach_ring(s, ring_ch(m, fc, 3), "[*]F"),
            ),
            _c(
                "core-piperazine",
                "core_hopping",
                "Change the piperidine to a piperazine (replace the ring CH2 para to "
                "the nitrogen with an NH).",
                mutate(s, ring_ch(m, pip_n, 3, aromatic=False)[0], 7),
            ),
        ]
    if target == "hs90a":
        benzyl_c = only(m, "[cH0](Cc)")
        return [
            _c(
                "sub-nitrile",
                "substitution",
                "Replace the ring fluorine with a nitrile.",
                react(s, "[c:1]F>>[c:1]C#N"),
            ),
            _c(
                "grow-dimethylamino",
                "growing",
                "Add a dimethylamino on the dimethoxybenzene, ortho to the methylene "
                "link.",
                attach_ring(s, ring_ch(m, benzyl_c, 1), "[*]N(C)C"),
            ),
        ]
    if target == "ada":
        return [
            _c(
                "core-triazole",
                "core_hopping",
                "Aza-substitute the imidazole C2 (between the two ring nitrogens) to a "
                "nitrogen, making a 1,2,3-triazole.",
                aza(s, only(m, "[cH1](:n):n")),
            ),
            _c(
                "sub-nitrile",
                "substitution",
                "Dehydrate the imidazole carboxamide to a nitrile.",
                react(s, "[c:1][CX3](=O)[NX3H2]>>[c:1]C#N"),
            ),
        ]
    if target == "nram":
        return [
            _c(
                "core-tetrazole",
                "core_hopping",
                "Replace the benzoic-acid carboxyl with a tetrazole bioisostere.",
                react(s, "[c:1][CX3](=O)[OX2H1]>>[c:1]c1nnn[nH]1"),
            ),
            _c(
                "sub-nmethyl",
                "substitution",
                "N-methylate the secondary aniline nitrogen.",
                attach(s, only(m, "[NX3;H1][c]"), "[*]C"),
            ),
        ]
    return []


# --- probe answers (independent of chemistree: raw RDKit + numpy) --------------------
def _receptor_residues(pdb: str) -> dict[str, np.ndarray]:
    """Group receptor atoms into standard-AA residues, label -> (K,3) coords.

    The label is ``NAME + number + "/" + chain`` so a homodimer's two copies of a
    residue (e.g. PRO81/A and PRO81/C) stay distinct. Keying only by name+number
    would collapse them (last write wins), corrupting both the residue's coordinates
    and any per-residue count.
    """
    rec = Chem.MolFromPDBFile(pdb, removeHs=False, sanitize=False)
    conf = rec.GetConformer()
    groups: dict[tuple, list] = defaultdict(list)
    for atom in rec.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None or info.GetResidueName().strip() not in _AA:
            continue
        key = (
            info.GetChainId(),
            info.GetResidueNumber(),
            info.GetResidueName().strip(),
        )
        pos = conf.GetAtomPosition(atom.GetIdx())
        groups[key].append((pos.x, pos.y, pos.z))
    return {f"{k[2]}{k[1]}/{k[0]}": np.array(v) for k, v in groups.items()}


def _group_coords(sdf: str, smarts: str) -> np.ndarray:
    """Heavy-atom coordinates of the salient group in the posed ligand."""
    lig = Chem.MolFromMolFile(sdf, removeHs=False)
    conf = lig.GetConformer()
    matched: set[int] = set()
    for match in lig.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
        matched.update(match)
    heavy = [i for i in matched if lig.GetAtomWithIdx(i).GetAtomicNum() > 1]
    assert heavy, f"{smarts!r}: no heavy atoms in {sdf}"
    return np.array([list(conf.GetAtomPosition(i)) for i in heavy])


def _mindist(a: np.ndarray, b: np.ndarray) -> float:
    diff = a[:, None, :] - b[None, :, :]
    return float(np.sqrt((diff * diff).sum(-1)).min())


def _distances(target: str, smarts: str) -> list[tuple[float, str]]:
    j = MANIFEST[target]
    coords = _group_coords(j["reference"], smarts)
    res = _receptor_residues(j["receptor"])
    return sorted((_mindist(c, coords), label) for label, c in res.items())


def _name_num(label: str) -> str:
    """The ``NAME+number`` part of a residue label, dropping the ``/chain`` suffix."""
    return label.split("/")[0]


def nearest_answer(target: str, smarts: str, *, min_margin: float = 0.30) -> str:
    """The nearest residue's name+number; assert the runner-up is comfortably farther.

    Args:
        target: Benchmark target key.
        smarts: SMARTS selecting the ligand group to measure from.
        min_margin: Minimum gap (Angstrom) to the nearest residue of a *different*
            name+number. A same-name copy on another chain is not a competing
            answer, so it does not shrink the margin.

    Returns:
        The winning residue as ``NAME+number`` (e.g. ``ILE50``), matching the
        question's requested format.

    Raises:
        AssertionError: If the margin to the next distinct residue is below
            ``min_margin`` (a coin-flip nearest is refused, not shipped).
    """
    ds = _distances(target, smarts)
    winner = _name_num(ds[0][1])
    runner = next(d for d, lab in ds[1:] if _name_num(lab) != winner)
    margin = runner - ds[0][0]
    assert margin >= min_margin, f"{target}: nearest margin {margin:.2f} too small"
    return winner


def count_answer(
    target: str,
    smarts: str,
    *,
    cutoff: float = 4.0,
    prefixes: tuple[str, ...] | None = None,
) -> str:
    """Count residues (optionally of given name prefixes) within cutoff of the group."""
    ds = _distances(target, smarts)
    n = sum(
        1
        for d, lab in ds
        if d <= cutoff and (prefixes is None or lab.startswith(prefixes))
    )
    return str(n)


# (kind, smarts, description, question). kind: nearest | count | count_name(prefixes).
_PROBES: dict[str, list[dict]] = {
    "egfr": [
        {"kind": "nearest", "smarts": "[NX3H2]", "group": "aminopyrimidine NH2"},
        {"kind": "count", "smarts": "[F]", "group": "benzyl fluorine"},
    ],
    "aa2ar": [
        {"kind": "count", "smarts": "[NX3H2]", "group": "exocyclic amino (NH2)"},
        {"kind": "count", "smarts": "o1cccc1", "group": "furan ring"},
    ],
    "andr": [
        {"kind": "count", "smarts": "[#6]=O", "group": "A-ring ketone oxygen"},
        {"kind": "count", "smarts": "[OX2H]", "group": "17-hydroxyl"},
    ],
    "hivpr": [
        {"kind": "nearest", "smarts": "[SX4](=O)(=O)", "group": "sulfonyl group"},
        # Nearest, not count: the sulfonyl count was boundary-brittle (ILE50 at
        # 4.28 A, margin 0.28) and "the sulfonyl group" (S+2O) is not a resolvable
        # target -- the group decomposition lumps it into the whole
        # sulfonamide+isobutyl. The pyrrolidine ring is a clean group, and its nearest
        # residue is the catalytic ASP25 (gap 0.75): the canonical HIV-protease core
        # question, pairing with the flap contact in probe-1.
        {
            "kind": "nearest",
            "smarts": "[NX3;R][CX4;R][CX4;R][CX4;R][CX4;R]",
            "group": "pyrrolidine ring",
        },
    ],
    "fa10": [
        {"kind": "nearest", "smarts": "[OX2H]", "group": "secondary alcohol hydroxyl"},
        {"kind": "count", "smarts": "[Cl]", "group": "naphthalene chlorine"},
    ],
    "hdac8": [
        {"kind": "count", "smarts": "[CX3](=O)[NX3][OX2H]", "group": "hydroxamic acid"},
        {
            "kind": "count_name",
            "smarts": "[NX3][OX2H]",
            "group": "hydroxamic acid",
            "prefixes": ("HID", "HIE", "HIP", "HIS"),
            "resname": "histidine",
        },
    ],
    "parp1": [
        {"kind": "nearest", "smarts": "[NX3H2][CX3]=O", "group": "primary carboxamide"},
        {"kind": "count", "smarts": "[NX3H2][CX3]=O", "group": "primary carboxamide"},
    ],
    "hs90a": [
        {"kind": "nearest", "smarts": "[NX3H2]", "group": "aminopyrimidine NH2"},
        # Nearest, not count: the within-4.0-A count (=5) was boundary-brittle --
        # MET98 sits at 3.88 A, only 0.12 A inside the 4.0 cutoff, a coin-flip integer.
        # The ring fluorine's nearest residue is GLY97 @ 2.70 A, runner ILE96 @ 3.21 A
        # (gap 0.51): a robust, single-answer contact question -- what the ring F points
        # into -- pairing with probe-1's NH2 hinge anchor (ASP93).
        {"kind": "nearest", "smarts": "[F]", "group": "ring fluorine"},
    ],
    "ada": [
        {
            "kind": "nearest",
            "smarts": "[NX3H2][CX3]=O",
            "group": "imidazole carboxamide",
        },
        {"kind": "count", "smarts": "[OX2H]", "group": "secondary alcohol hydroxyl"},
    ],
    "nram": [
        {"kind": "count", "smarts": "[CX3](=O)[OX2H]", "group": "carboxylic acid"},
        {
            "kind": "count_name",
            "smarts": "[CX3](=O)[OX2H]",
            "group": "carboxylic acid",
            "prefixes": ("ARG",),
            "resname": "arginine",
        },
    ],
}


def probes(target: str) -> list[dict]:
    """Build a target's probe rows with computed answers."""
    j = MANIFEST[target]
    rows = []
    for i, p in enumerate(_PROBES[target], 1):
        if p["kind"] == "nearest":
            q = (
                f"Which receptor residue (give its name and number, e.g. ASP381) has "
                f"an atom closest to the ligand's {p['group']}?"
            )
            ans = nearest_answer(target, p["smarts"])
        elif p["kind"] == "count":
            q = (
                f"How many receptor residues have any atom within 4.0 Angstrom of the "
                f"ligand's {p['group']}?"
            )
            ans = count_answer(target, p["smarts"])
        else:  # count_name
            q = (
                f"How many {p['resname']} residues have any atom within 4.0 Angstrom "
                f"of the ligand's {p['group']}?"
            )
            ans = count_answer(target, p["smarts"], prefixes=p["prefixes"])
        rows.append(
            {
                "id": f"probe-{i}-{p['kind']}",
                "ligand": j["reference"],
                "receptor": j["receptor"],
                "question": q,
                "answer": ans,
            }
        )
    return rows


# --- 3D decoration + scaffold production --------------------------------------------
def decoration(target: str) -> dict:
    """One decoration row (paths + optimize instruction, abl1 style)."""
    j = MANIFEST[target]
    return {
        "id": target,
        "target": j["reference"],
        "receptor": j["receptor"],
        "instruction": "Optimize the scaffold to improve predicted binding.",
    }


def write_scaffold(target: str) -> tuple[str, int, int]:
    """Materialize the Murcko scaffold SDF (same code the runner uses); verify it."""
    j = MANIFEST[target]
    out = str(DATA / target / "scaffold.sdf")
    scaffold_smiles = strip_murcko.write_scaffold(j["reference"], out)
    scaffold = Chem.MolFromSmiles(scaffold_smiles)
    ligand = Chem.MolFromSmiles(flat(j["smiles"]))
    n_scaffold, n_ligand = scaffold.GetNumHeavyAtoms(), ligand.GetNumHeavyAtoms()
    assert n_scaffold > 0, f"{target}: empty scaffold"
    assert n_scaffold < n_ligand, f"{target}: scaffold not smaller than ligand"
    assert scaffold.GetRingInfo().NumRings() >= 1, f"{target}: scaffold has no ring"
    return scaffold_smiles, n_scaffold, n_ligand


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    """Generate all case files + scaffolds and print a verification report."""
    CASES.mkdir(exist_ok=True)
    n2d = nprobe = 0
    for t in TARGETS:
        start = flat(MANIFEST[t]["smiles"])
        e2d = edits_2d(t)
        for e in e2d:
            assert Chem.MolFromSmiles(e["gold"]) is not None, f"{t}/{e['id']}: bad gold"
            assert e["gold"] != canon(start), f"{t}/{e['id']}: gold == start"
        rows2d = [dict(e, smiles=start) for e in e2d]
        _write(CASES / f"{t}_2d.jsonl", rows2d)
        _write(CASES / f"{t}_3d_decoration.jsonl", [decoration(t)])
        prb = probes(t)
        _write(CASES / f"{t}_3d_probes.jsonl", prb)
        scaf_smiles, n_s, n_l = write_scaffold(t)
        n2d += len(rows2d)
        nprobe += len(prb)
        weak = " [WEAK: few R-atoms to recover]" if (n_l - n_s) < 4 else ""
        print(
            f"{t:7} 2D={len(rows2d)} probes={len(prb)} "
            f"scaffold={n_s}/{n_l} heavy (strip {n_l - n_s}){weak}  {scaf_smiles}"
        )
        for e in e2d:
            print(f"    2D [{e['id']:14}] {formula(start)}->{formula(e['gold'])}")
        for p in prb:
            print(f"    Q  [{p['id']:16}] ans={p['answer']:8} {p['question'][:60]}...")
    print(
        f"\nTOTAL: {n2d} 2D + {nprobe} probes + {len(TARGETS)} decoration "
        f"across {len(TARGETS)} targets"
    )


if __name__ == "__main__":
    main()
