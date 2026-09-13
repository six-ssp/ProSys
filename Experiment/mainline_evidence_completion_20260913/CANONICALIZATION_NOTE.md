# Canonicalization audit boundary

During the strict split audit, RDKit reported ring-closure errors when the
legacy helper parsed dot-separated substrings independently. This is not an
invalid whole reaction molecule: SMILES ring closures may connect atoms across
such substrings.

An independent scan (`scripts/audit_canonical_fragments.py`) parsed each whole
reaction side first, then used `Chem.GetMolFrags(..., asMols=True)` and canonical
SMILES to form complete connected components. Retained evidence is in
`canonical_fragment_audit.json`.

Findings on all six condition datasets:

- No whole reactant/product side failed RDKit parsing.
- Exactly two side strings differ from the legacy dot-splitting result: the
  reactants and product of the same Diels-Alder training record, ID `11028191`.
- No validation or test side differs in this scan.
- Train/validation, train/test and validation/test canonical-reaction
  intersections remain zero for every family with the complete-component keys.

The frozen mainline uses the legacy side helper in parts of its reaction
indexing/representation path. Its rare training-record representation loss is
a known implementation limitation, not evidence of test leakage. This audit
does not prove that retraining with a corrected helper would leave scores
unchanged. No frozen split, checkpoint or model feature was changed during the
identity-level reconstruction; fixing the helper should be versioned and
followed by a Diels-Alder sensitivity rerun rather than silently attributed to
the historical models.

The existing strict Stage 1/2 audit also exited successfully: family route
train/validation/test intersections are zero, family validation/test route keys
are anchored to condition validation/test keys, and family route train+validation
does not overlap condition test keys. That command does not audit the entire
USPTO pretraining corpus; do not extend its scope to a new USPTO leakage claim.
