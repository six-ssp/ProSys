# ProSys preprocessing maintenance

ProSys uses the filtered USPTO-FULL base and six Reaxys family datasets. The
upstream examples below are historical examples, not the current data policy.
Whole reaction sides are parsed before connected components are split, so
cross-dot ring closures are not mistaken for disconnected products/reactants.
After root-aligned augmentation, empty source/target pairs are rejected and
their pre-filter indices are saved in `<split>.pair_filter.json`.

These repairs do not rewrite retained augmented/binarized inputs. See
`../../Experiment/project_completion_20260913/VALIDATION_DECISION.md` before
restarting expert training. Clean-validation selection is still pending.

## Upstream examples

- Download raw datasets and put them in the raw_datasets folder, and do SPE tokenization:

```
-  python preprocess_data.py -dataset USPTO_50K -augmentation 20 -processes 8 -spe -dropout 0 
-  python preprocess_data.py -dataset USPTO_FULL -augmentation 10 -processes 8 -spe -dropout 0
```

- Data processing using fairseq, for example, USPTO_50K,
```shell
sh binarize.sh ../datasets/USPTO_50K/aug20 dict.txt
```
