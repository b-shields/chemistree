# Working with rdkit and smina

You edit and evaluate the molecule yourself with the Bash tool: rdkit (python) to read and
modify structures, and smina to score poses. Apply the medicinal-chemistry strategy above
with these tools.

# Reading the pocket and the score

Work from the posed scaffold SDF and the receptor. To judge a candidate, build a 3D
structure for it (rdkit: embed a conformer, or graft your substituents onto the scaffold
pose) and score it against the receptor with smina Vinardo — a more negative score is a
better fit:

```
smina --receptor <receptor.pdb> --ligand <candidate.sdf> \
      --autobox_ligand <scaffold.sdf> --scoring vinardo --score_only
```

Read which residues line each open position (measure distances with rdkit, or dock and
inspect the pose) and choose edits that suit them, then re-score. Keep a change that lowers
the score without introducing a clash, and back out one that does not.
