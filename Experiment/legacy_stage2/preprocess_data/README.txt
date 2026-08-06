Simple Usage Examples
"Convert a chemical name to SMILES"

Archived note:
the active project no longer depends on the deleted top-level `preprocess_data/`
directory. The current maintained pipeline uses `data_preprocess/preprocess.py`
plus `data_preprocess/name_to_smiles.tsv` instead. The instructions below are
kept only as legacy background for the old Stage-2 workflow.


(1) preprocess raw Reaxys .xlsx to .txt files:

python preprocess_reaxys.py

(2) convert chemical label names to smiles:

cd ../All_LCC_Data/processed_data_temp/unprocessed_class
[archived] OPSIN jar path from the deleted top-level `preprocess_data/` is omitted in
the maintained repo. If this legacy flow is ever revived, provide an OPSIN jar locally
and update the command path here.

(source: https://github.com/dan2097/opsin)

(3) use PubChem and ChemSpider to double check the chemical names and emerge the names and smiles:

python emerge.py --input_dir ../All_LCC_Data/processed_data_temp/unprocessed_class \
    --output_dir ../All_LCC_Data/processed_data_temp/label_processed

(3-2) manually preprocess the labels.
python manually_modify.py

(4) use the new label names to process all the train, validation split .txt files:

python process_all_data.py --target_dir ../All_LCC_Data/processed_data_temp

(5) Assign Atom Mapping if we need to use Chemprop condensed graph of reaction to encode reaction:
python AssignAtomMapping.py

/home/lungyi/rxn_yield_context/rxn_yield_context/All_LCC_Data/processed_data_temp/label_processed
/home/lungyi/rxn_yield_context/rxn_yield_context/All_LCC_data/processed_data_temp/label_processed
