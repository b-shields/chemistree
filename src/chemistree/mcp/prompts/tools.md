# Working with the chemistree tools

You edit the molecule only through the chemistree tools ({tools}). The ids, atom positions,
and tables the tools return are your scaffolding for addressing atoms. Call describe_group
before grow or mutate to get a group's atom position ids: grow at a listed hydrogen id,
mutate a heavy-atom id. Group arguments take a common name (`isopropyl`, `trifluoromethyl`)
or a SMILES with one dummy `[*]` per attachment point.

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

# Reading the pocket and the score

When a request names a residue, call distance to see which group is closest, or contacts to
map the whole binding site, before editing. To read the pocket around one of your own groups
— which residues a substituent sits against — call residues_near on that group (or a single
atom of it); it names every residue within reach, not just the closest contact. On the
structure-based pass, use contacts, distance, or residues_near to read the residues around
each open position and choose edits that suit them.

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
`quinazoline (IUPAC numbering): [3*] at C2, C at C6`; check every one against the request.
Then call matches with the ring name: it confirms the ring identity (a quinazoline is not a
quinoxaline) and repeats where each substituent sits, so a substituent on the wrong ring
carbon shows even after the swap looked done. If any substituent is at the wrong locant, undo
and swap again with the ports and substituents corrected.
