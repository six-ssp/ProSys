# Validation-boundary decision: measured preview

No dataset, frozen split or historical model has been changed by this preview.
The user subsequently requested a freeze: neither proposed retraining protocol
is selected, no validation exclusion is applied, and no further training is
scheduled. The options below remain a record of the unresolved limitation.

## Fixed-base clean-validation option

Exclude the exact canonical reactions shared with either the retained USPTO
base-training or base-validation corpus from model-selection validation only.
Do not move excluded validation rows into training or test. Preserve every
existing condition test identity. Apply the exclusion to all relevant
selection paths: expert early stopping, neural condition/graph encoders,
XGBoost model selection and predicted-validation-route fusion/prior selection.
Simply changing the reported validation denominator after training is not a fix.

| Family | Original validation reactions | Exclude | Remaining reactions | Original condition rows | Exclude condition rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Beckmann | 234 | 1 | 233 | 515 | 1 |
| Buchwald-Hartwig | 1097 | 46 | 1051 | 1722 | 257 |
| Chan-Lam | 389 | 12 | 377 | 571 | 14 |
| Diels-Alder | 760 | 10 | 750 | 813 | 24 |
| Friedel-Crafts acylation | 475 | 25 | 450 | 648 | 54 |
| Friedel-Crafts alkylation | 893 | 4 | 889 | 1594 | 9 |
| Total | 3848 | 98 | 3750 | 5863 | 359 |

The exclusion is 2.55% of unique validation reactions and 6.12% of condition
rows. Multiple conditions per reaction explain the difference. The comparison
is reaction-disjoint, not product-disjoint. This is a new selection protocol,
not a retrospective certification of the original trained models.

All 39640 retained family validation source/target text pairs were re-encoded
with the stored dictionaries and matched against their actual binary tensors.
All matched exactly. The 98 overlaps correspond to 980 augmented pairs.
Unknown-token rows remain recorded separately in the preview; they are not
the same as empty or chemically invalid targets.

## Additional prepared-data defect

The binary checks exposed empty reactant targets after root alignment. They
are actual EOS-only sequences in the retained family binary inputs, not a
spreadsheet-display problem. Their original export lacked a final nonempty
pair check. The source export now rejects such pairs and writes a per-split
filter manifest; retained data are unchanged pending versioned rebuilding.

| Family | Empty training pairs | Empty validation pairs | Empty prepared test pairs |
| --- | ---: | ---: | ---: |
| Beckmann | 2390 | 0 | 30 |
| Buchwald-Hartwig | 350 | 0 | 20 |
| Chan-Lam | 210 | 10 | 40 |
| Diels-Alder | 6300 | 60 | 40 |
| Friedel-Crafts acylation | 930 | 0 | 10 |
| Friedel-Crafts alkylation | 4430 | 10 | 10 |
| Total | 14610 | 80 | 150 |

The retained USPTO base augmented train/validation inputs have **zero empty
pairs** among 7437290 and 1859060 pairs, respectively. Thus this new defect does
not by itself require USPTO base retraining. Cleaning the family prepared
inputs is required before new expert fits. Test-query manifests must remain
fixed: dropping an invalid prepared target is not permission to drop a failed
product from the route/system evaluation denominator. The maintained test
route inference reads the independent condition test manifest, not these
prepared target sequences.

## Full base-retraining option

Alternatively remove both condition-validation and condition-test overlaps
from USPTO before training a new base, and regenerate family prepared inputs
with the nonempty-pair fix. This avoids changing which condition validation
reactions are used, but requires new base training, expert training and
dependent evaluations. Historical raw-to-bin hashes cannot be reconstructed;
new preprocessing should persist row-level admission/exclusion manifests.

Either option must be versioned and chosen for data integrity, not selected
according to whichever produces higher held-out test scores.

Sources: `validation_preview/preview.json`, `validation_preview/summary.csv`,
`empty_stage1_pairs/audit.json`, and `uspto_audit/audit.json`.
