# Working with the chemistree tools

You edit the molecule only through the chemistree tools ({tools}). The ids, atom positions,
and tables the tools return are your scaffolding for addressing atoms. Call describe_group
before grow or mutate to get a group's atom position ids: grow at a listed hydrogen id,
mutate a heavy-atom id. Group arguments take a common name (`isopropyl`, `trifluoromethyl`),
a named heteroaromatic ring with a port locant for each attachment (`<ring> <label>@<locant>`
for swap, `<ring> <locant>` for grow — no ring SMILES to write), or a SMILES with one dummy `[*]`
per attachment point. Write every group in its neutral form, as the structure is drawn on
paper: the edit is scored against a neutral canonical structure, so a formal charge or an
extra proton makes it mismatch even when the heavy atoms are right. A primary amine is `N`
(`[NH2]`), never the protonated `[NH3+]`; a carboxylic acid is `C(=O)O`, not the carboxylate
`C(=O)[O-]`. Do not add a charge for the physiological protonation state. `bind` loads the
starting molecule
once — to change the molecule, edit it with swap/grow/mutate/remove; never re-bind a
hand-written SMILES to apply an edit, which throws away the group ids and skips the
ring-position feedback and `matches` check.

Work deliberately: think the edit through, then make one considered move and read its result.
A ring swap reports where every substituent sits by IUPAC locant — a port as `[3*] at C2`, a
kept atom as its element, e.g. `F at C4` — so a right ring with a substituent on the wrong
carbon shows. Check every locant against what the request asked (both the attachment and the
kept substituents), and if any sits at the wrong locant, undo and swap again with the ports
and substituents rearranged. Do not accept a partly-wrong ring, and do not strip substituents
you meant to keep. If a swap or grow returns an error, the group SMILES was wrong — fix that SMILES and
retry the same edit. A few deliberate moves beat a flurry of trial-and-error. Before you give a
final molecule, if you built or changed a named ring, run this check once more — read the
reported locants or call `matches` — and do not finish with a ring or substituent you have not
verified against the request.

Report your final SMILES by calling the `smiles` tool and copying its output **exactly**.
Never type, transcribe, or hand-edit a SMILES for the final answer — re-drawing a molecule by
hand risks a mis-placed atom or a broken ring closure. The `smiles` tool always returns the
valid canonical SMILES of the molecule you actually built; that exact string is the answer.

# Reading the pocket and the score

When a request names a residue, call distance to see which group is closest, or contacts to
map the whole binding site, before editing. To read the pocket around one of your own groups
— which residues a substituent sits against — call residues_near on that group (or a single
atom of it); it names every residue within reach, not just the closest contact. On the
structure-based pass, use contacts, distance, or residues_near to read the residues around
each open position and choose edits that suit them.

To read which receptor residues lie near a specific part of your ligand — one group, or one
atom of it — call residues_near on that group (pass a single atom id to narrow it). It names
every residue within the cutoff, so it suits a part-specific reading where contacts does not:
contacts maps the whole site and reports only the closest group per residue. The residues_near
table may also list a metal ion, cofactor, or water (e.g. ZN, MG, HOH), which are not
amino-acid residues.

When a receptor is loaded, the group listing ends with a predicted affinity (Vinardo score;
more negative is better) and, separately, the ligand's internal energy, both refreshed after
every edit. Read the affinity to judge whether a change helped binding. The affinity is
protein–ligand only, so a better number alone does not mean a sound pose: if the internal
energy jumps after an edit, groups are clashing inside the molecule. After adding or moving a
group, call clashes to check the new pose; if a group clashes with another group or the
protein, minimize it — this turns the group about its attachment bond, carrying its
substituents, to the lowest-energy rotamer. Do not keep an edit that leaves the internal
energy high.

The group listing also carries a physicochemical profile and a structure-alert line,
refreshed after every edit — read them to keep the molecule drug-like.

# Core hopping with the tools

For a core hop, first call describe_group on the old scaffold and read how its ports sit
relative to one another (on a fused ring these relations are given as bond counts, e.g.
`3 bonds`; a plain benzene uses ortho/meta/para), then place the new scaffold's ports to match.
Getting a fused ring right often takes more than one try. After a ring swap the result names
the ring and gives every substituent's IUPAC ring locant — ports and kept atoms alike, e.g.
`pyridazine (IUPAC numbering): [3*] at C3, C at C5`; check every one against the request.
Then call matches with the ring name: it confirms the ring identity (the built ring is the
one named, not a near-isomer) and repeats where each substituent sits, so a substituent on
the wrong ring carbon shows even after the swap looked done. If any substituent is at the
wrong locant, undo and swap again with the ports and substituents corrected.

Tell two kinds of ring change apart. When only **one ring atom's element** changes — an
aza-substitution that turns a ring carbon into N, or a ring CH2 into N — do it as a single
`mutate` on that one atom, not a whole-ring swap. Find the atom the request names with
describe_group and mutate it to N.
mutate keeps the ring's attachment port and every substituent exactly where they are and
reports the new ring name and locants, so there is no attachment isomer to pick and no open
`[*]` port left behind. Rebuilding the whole ring by hand with swap for such a change risks
putting the attachment on the wrong ring atom or leaving a dummy port unfilled. Reserve swap
for a change to the ring framework itself — a different ring, a fused or contracted ring, or
a bioisostere replacement.

Some groups are not in the numbered-ring table — a fused all-carbon ring, a small saturated
heterocycle, or a specific tautomer of a ring. A name-with-locant swap cannot build these
(the name errors), and neither the swap
result nor `matches` reports a locant for them, so nothing verifies the attachment isomer,
tautomer, or ring size for you. Write the group SMILES by hand, then read it back atom by
atom — check the ring size, the attachment isomer, and any NH position against the request —
before you finish, since no locant check will flag a mistake here.
