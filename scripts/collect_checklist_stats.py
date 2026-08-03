"""Collect the statistics referenced by checklist.md for the current mainline."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess.preprocess import (  # noqa: E402
    DEFAULT_LABEL_FREQ_SCOPE,
    DEFAULT_MIN_LABEL_FREQ,
    DEFAULT_MIN_YIELD,
    KEEP_COLUMNS,
    REACTION_TYPES,
    SENTINEL_VALUES,
    apply_label_map,
    build_frequency_dict,
    build_label_name_map,
    canonical_smiles,
    deduplicate_condition_records,
    deduplicate_condition_strings,
    filter_rare_labels_all_families,
    highest_temperature,
    hydrate_strip,
    is_nan,
    is_valid_smiles,
    load_name_to_smiles,
    merge_reagent_catalyst,
    normalize_label,
    raw_load_file,
    resolve_name_to_smiles_path,
    split_reaction_smiles,
    standardize_labels_in_series,
)
from prosys_shared.mainline import (  # noqa: E402
    FAMILY_ORDER,
    display_family_name,
    evaluate_scored_frame,
    family_dir,
    load_split_rows,
    parse_families_arg,
    split_file_for_family,
    stage1_route_recall,
)
from stage1_retrosynthesis.utils.get_ranked_topk import (  # noqa: E402
    canonicalize_smiles_clear_map,
    process_input,
)


def _family_sort_key(family: str) -> tuple[int, str]:
    try:
        return (FAMILY_ORDER.index(family), family)
    except ValueError:
        return (len(FAMILY_ORDER), family)


def _write_csv(rows: list[dict], output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_file, index=False)
    return output_file


def _write_json(data: dict | list, output_file: Path) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return output_file


def _safe_numeric(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    return value


def _mean(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(np.asarray(numeric, dtype=np.float64)))


def _weighted_mean(rows: list[dict], value_key: str, weight_key: str) -> float | None:
    pairs = [
        (float(row[value_key]), float(row[weight_key]))
        for row in rows
        if row.get(value_key) is not None and row.get(weight_key) not in (None, 0)
    ]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def _count_tokens(text: str) -> int:
    if is_nan(text) or not text:
        return 0
    return len([token for token in str(text).split('; ') if token and token.lower() != 'nan'])


def _canonicalize_side(side: str) -> str:
    canon: list[str] = []
    for part in str(side).split('.'):
        part = part.strip()
        if not part:
            continue
        value = canonical_smiles(part)
        if value:
            canon.append(value)
    return '.'.join(sorted(canon))


def _canonical_route_key(reactants: str, product: str) -> str:
    reactant_key = _canonicalize_side(reactants)
    product_key = _canonicalize_side(product)
    if not reactant_key or not product_key:
        return ''
    return reactant_key + '>>' + product_key


def _split_validity_counts(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work = split_reaction_smiles(frame.copy())
    before = len(work)
    bad_reactants = work['reactants'].apply(lambda value: is_nan(value) or not is_valid_smiles(value))
    work = work.loc[~bad_reactants].reset_index(drop=True)
    bad_products = work['products'].apply(lambda value: is_nan(value) or not is_valid_smiles(value))
    work = work.loc[~bad_products].reset_index(drop=True)
    after_rdkit = len(work)
    bad_yield = work['Yield (numerical)'].apply(is_nan)
    work = work.loc[~bad_yield].reset_index(drop=True)
    after_yield = len(work)
    numeric_yield = pd.to_numeric(work['Yield (numerical)'], errors='coerce')
    work = work.loc[numeric_yield >= DEFAULT_MIN_YIELD].reset_index(drop=True)
    after_min_yield = len(work)
    bad_solvent = work['Solvent (Reaction Details)'].apply(is_nan)
    work = work.loc[~bad_solvent].reset_index(drop=True)
    after_solvent = len(work)
    return work, {
        'before_rdkit': before,
        'after_rdkit': after_rdkit,
        'after_yield': after_yield,
        'after_min_yield': after_min_yield,
        'after_solvent': after_solvent,
    }


def _normalize_labels(frame: pd.DataFrame, name_to_smiles: dict) -> pd.DataFrame:
    work = frame.copy()
    work['Reagent'] = standardize_labels_in_series(work['Reagent'], name_to_smiles)
    work['Solvent (Reaction Details)'] = standardize_labels_in_series(work['Solvent (Reaction Details)'], name_to_smiles)
    reagent_name_map = build_label_name_map(work['Reagent'], name_to_smiles)
    solvent_name_map = build_label_name_map(work['Solvent (Reaction Details)'], name_to_smiles)
    work['Reagent'] = apply_label_map(work['Reagent'], reagent_name_map)
    work['Solvent (Reaction Details)'] = apply_label_map(work['Solvent (Reaction Details)'], solvent_name_map)
    return work


def _apply_complexity_filter(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    bad_solvent = work['Solvent (Reaction Details)'].apply(lambda value: _count_tokens(value) > 2)
    bad_reagent = work['Reagent'].apply(lambda value: _count_tokens(value) > 3)
    return work.loc[~(bad_solvent | bad_reagent)].reset_index(drop=True)


def _load_family_input(payload: tuple[str, Path]) -> tuple[str, int, pd.DataFrame]:
    """Read each source file once and retain the Stage-2 columns used in production."""

    family, input_root = payload
    family_input = input_root / family
    raw_files = sorted(
        path
        for path in family_input.iterdir()
        if path.is_file() and path.suffix.lower() in {'.csv', '.xlsx'} and not path.name.startswith('~$')
    )
    raw_frames = [raw_load_file(str(path)) for path in raw_files]
    raw_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    keep_columns = [column for column in KEEP_COLUMNS if column in raw_df.columns]
    stage2_df = raw_df.loc[:, keep_columns].copy()
    if 'Number of Reaction Steps' in stage2_df.columns:
        stage2_df = stage2_df.loc[stage2_df['Number of Reaction Steps'] == 1].reset_index(drop=True)
    return family, len(raw_df), stage2_df


def _reassign_family_with_global_roles(
    payload: tuple[str, pd.DataFrame, dict[str, int], dict[str, int]],
) -> tuple[str, pd.DataFrame]:
    """Apply the production role-reassignment rule using fixed global frequencies."""

    family, frame, reagent_freq, solvent_freq = payload
    work = frame.copy()
    for row_index in work.index:
        reagents = (
            [value for value in str(work.at[row_index, 'Reagent']).split('; ') if value and value.lower() not in SENTINEL_VALUES]
            if not is_nan(work.at[row_index, 'Reagent']) else []
        )
        solvents = (
            [value for value in str(work.at[row_index, 'Solvent (Reaction Details)']).split('; ') if value and value.lower() not in SENTINEL_VALUES]
            if not is_nan(work.at[row_index, 'Solvent (Reaction Details)']) else []
        )
        new_reagents = [value for value in reagents if value not in solvent_freq]
        new_solvents = [value for value in solvents if value not in reagent_freq]
        for value in reagents:
            if value in solvent_freq and value not in new_solvents:
                new_solvents.append(value)
        for value in solvents:
            if value in reagent_freq and value not in new_reagents:
                new_reagents.append(value)
        work.at[row_index, 'Reagent'] = 'nan' if not new_reagents else '; '.join(new_reagents)
        work.at[row_index, 'Solvent (Reaction Details)'] = 'nan' if not new_solvents else '; '.join(new_solvents)

    bad_solvent = work['Solvent (Reaction Details)'].apply(is_nan)
    return family, work.loc[~bad_solvent].reset_index(drop=True)


def _clean_family_before_low_frequency(
    payload: tuple[str, pd.DataFrame, dict[str, int], dict],
) -> tuple[str, pd.DataFrame, dict[str, int]]:
    """Run family-local cleanup after the shared role reassignment step."""

    family, frame, metadata, name_to_smiles = payload
    valid_df, validity = _split_validity_counts(frame)
    valid_df['Temperature (Reaction Details) [C]'] = valid_df['Temperature (Reaction Details) [C]'].apply(highest_temperature)
    valid_df = merge_reagent_catalyst(valid_df)
    after_merge = len(valid_df)
    valid_df = deduplicate_condition_strings(valid_df)
    valid_df = _normalize_labels(valid_df, name_to_smiles)
    after_normalization = len(valid_df)
    valid_df = _apply_complexity_filter(valid_df)

    work = valid_df.copy()
    work['canonical_key'] = work.apply(lambda row: _canonical_route_key(row['reactants'], row['products']), axis=1)
    counts = {
        'raw_records': metadata['raw_records'],
        'single_step_records': metadata['single_step_records'],
        'after_role_reassignment': metadata['after_role_reassignment'],
        'after_rdkit': validity['after_rdkit'],
        'after_yield': validity['after_yield'],
        'after_min_yield': validity['after_min_yield'],
        'after_solvent': validity['after_solvent'],
        'after_merge': after_merge,
        'after_normalization': after_normalization,
        'after_complexity': len(valid_df),
    }
    return family, work, counts


def simulate_clean_pipeline(repo_root: Path, families: list[str], *, workers: int = 1) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    input_root = repo_root / 'data' / 'reaxys_input'
    mapping_path = resolve_name_to_smiles_path()
    name_to_smiles = load_name_to_smiles(str(mapping_path)) if mapping_path else {}

    single_step_frames: dict[str, pd.DataFrame] = {}
    family_metadata: dict[str, dict[str, int]] = {}
    pre_lowfreq_frames: dict[str, pd.DataFrame] = {}
    stats_rows: list[dict] = []
    label_space_rows: list[dict] = []
    split_cache: dict[str, dict] = {}

    input_payloads = [(family, input_root) for family in families]
    if workers > 1 and len(input_payloads) > 1:
        with mp.get_context('fork').Pool(processes=min(workers, len(input_payloads))) as pool:
            loaded_families = pool.map(_load_family_input, input_payloads)
    else:
        loaded_families = [_load_family_input(payload) for payload in input_payloads]

    for family, raw_records, single_step_df in loaded_families:
        single_step_frames[family] = single_step_df
        family_metadata[family] = {
            'raw_records': raw_records,
            'single_step_records': len(single_step_df),
        }

    # Match the actual preprocessing pipeline: role reassignment is performed
    # globally across all single-step family frames before family-local cleanup.
    all_data = pd.concat(single_step_frames.values(), ignore_index=True)
    reagent_freq = build_frequency_dict(all_data['Reagent'])
    solvent_freq = build_frequency_dict(all_data['Solvent (Reaction Details)'])
    for chemical in set(reagent_freq) & set(solvent_freq):
        if solvent_freq.get(chemical, 0) >= reagent_freq.get(chemical, 0):
            reagent_freq.pop(chemical, None)
        else:
            solvent_freq.pop(chemical, None)

    role_payloads = [
        (family, single_step_frames[family], reagent_freq, solvent_freq)
        for family in families
    ]
    if workers > 1 and len(role_payloads) > 1:
        with mp.get_context('fork').Pool(processes=min(workers, len(role_payloads))) as pool:
            reassigned_families = pool.map(_reassign_family_with_global_roles, role_payloads)
    else:
        reassigned_families = [_reassign_family_with_global_roles(payload) for payload in role_payloads]
    single_step_frames = {family: frame for family, frame in reassigned_families}

    for family in families:
        family_metadata[family]['after_role_reassignment'] = len(single_step_frames[family])

    cleanup_payloads = [(family, single_step_frames[family], family_metadata[family], name_to_smiles) for family in families]
    if workers > 1 and len(cleanup_payloads) > 1:
        with mp.get_context('fork').Pool(processes=min(workers, len(cleanup_payloads))) as pool:
            cleaned_families = pool.map(_clean_family_before_low_frequency, cleanup_payloads)
    else:
        cleaned_families = [_clean_family_before_low_frequency(payload) for payload in cleanup_payloads]

    for family, frame, counts in cleaned_families:
        pre_lowfreq_frames[family] = frame
        split_cache[family] = counts

    lowfreq_input = {family: frame.copy() for family, frame in pre_lowfreq_frames.items()}
    lowfreq_output = filter_rare_labels_all_families(
        lowfreq_input,
        min_freq=DEFAULT_MIN_LABEL_FREQ,
        scope=DEFAULT_LABEL_FREQ_SCOPE,
    )

    for family in families:
        dedup_df = deduplicate_condition_records(lowfreq_output[family].copy())
        pre_lowfreq_frames[family] = dedup_df
        counts = split_cache[family]
        stats_rows.append(
            {
                'family': family,
                'raw_records': counts['raw_records'],
                'single_step_records': counts['single_step_records'],
                'removed_multi_step': counts['raw_records'] - counts['single_step_records'],
                'after_role_reassignment': counts['after_role_reassignment'],
                'removed_role_reassignment': counts['single_step_records'] - counts['after_role_reassignment'],
                'rdkit_valid_records': counts['after_rdkit'],
                'removed_rdkit_fail': counts['after_role_reassignment'] - counts['after_rdkit'],
                'after_missing_yield_filter': counts['after_yield'],
                'removed_missing_yield': counts['after_rdkit'] - counts['after_yield'],
                'after_min_yield_filter': counts['after_min_yield'],
                'removed_low_yield': counts['after_yield'] - counts['after_min_yield'],
                'after_missing_solvent_filter': counts['after_solvent'],
                'removed_missing_solvent': counts['after_min_yield'] - counts['after_solvent'],
                'after_catalyst_merge': counts['after_merge'],
                'after_label_normalization': counts['after_normalization'],
                'after_complexity_filter': counts['after_complexity'],
                'after_low_frequency_filter': len(lowfreq_output[family]),
                'after_deduplication': len(dedup_df),
                'final_all_dataset': len(dedup_df),
            }
        )

        contexts = (
            dedup_df[['Reagent', 'Solvent (Reaction Details)']]
            .drop_duplicates()
            .shape[0]
            if not dedup_df.empty
            else 0
        )
        route_keys = dedup_df['canonical_key'].replace('', np.nan).dropna().unique().tolist() if 'canonical_key' in dedup_df.columns else []
        temp_values = pd.to_numeric(dedup_df['Temperature (Reaction Details) [C]'], errors='coerce')
        yield_values = pd.to_numeric(dedup_df['Yield (numerical)'], errors='coerce')
        label_space_rows.append(
            {
                'family': family,
                'final_reactions': len(dedup_df),
                'unique_routes': len(route_keys),
                'unique_reagent_labels': len(build_frequency_dict(dedup_df['Reagent'])),
                'unique_solvent_labels': len(build_frequency_dict(dedup_df['Solvent (Reaction Details)'])),
                'unique_contexts': int(contexts),
                'avg_reagents_per_reaction': float(np.mean(dedup_df['Reagent'].apply(_count_tokens))) if not dedup_df.empty else 0.0,
                'avg_solvents_per_reaction': float(np.mean(dedup_df['Solvent (Reaction Details)'].apply(_count_tokens))) if not dedup_df.empty else 0.0,
                'median_temperature': float(temp_values.dropna().median()) if temp_values.notna().any() else None,
                'temperature_valid_rate': float(temp_values.notna().mean()) if len(temp_values) else 0.0,
                'median_yield': float(yield_values.dropna().median()) if yield_values.notna().any() else None,
            }
        )

    return stats_rows, pre_lowfreq_frames, label_space_rows


def collect_split_stats(repo_root: Path, families: list[str]) -> list[dict]:
    rows: list[dict] = []
    for family in families:
        split_rows = {split: load_split_rows(split_file_for_family(repo_root, family, split)) for split in ['train', 'val', 'test']}
        train_contexts = {(row['reagent_norm'], row['solvent_norm']) for row in split_rows['train']}
        train_routes = {f"{row['reactants']}>>{row['product']}" for row in split_rows['train']}
        rows.append(
            {
                'family': family,
                'train_reactions': len(split_rows['train']),
                'valid_reactions': len(split_rows['val']),
                'test_reactions': len(split_rows['test']),
                'train_routes': len({f"{row['reaction_id']}|{row['reactants']}|{row['product']}" for row in split_rows['train']}),
                'valid_routes': len({f"{row['reaction_id']}|{row['reactants']}|{row['product']}" for row in split_rows['val']}),
                'test_routes': len({f"{row['reaction_id']}|{row['reactants']}|{row['product']}" for row in split_rows['test']}),
                'train_contexts': len(train_contexts),
                'valid_contexts': len({(row['reagent_norm'], row['solvent_norm']) for row in split_rows['val']}),
                'test_contexts': len({(row['reagent_norm'], row['solvent_norm']) for row in split_rows['test']}),
                'test_contexts_unseen_in_train': len({(row['reagent_norm'], row['solvent_norm']) for row in split_rows['test']} - train_contexts),
                'test_routes_unseen_in_train': len({f"{row['reactants']}>>{row['product']}" for row in split_rows['test']} - train_routes),
            }
        )
    return rows


def collect_uspto_stats(repo_root: Path) -> dict:
    raw_dir = repo_root / 'data' / 'editretro' / 'datasets' / 'USPTO_full'
    filtered_dir = repo_root / 'data' / 'editretro' / 'datasets' / 'USPTO_STAGE2_FILTERED'
    filtered_raw_dir = filtered_dir / 'raw'
    summary_file = filtered_dir / 'summary.json'
    if summary_file.exists():
        stats = json.loads(summary_file.read_text(encoding='utf-8'))
        return {
            'raw_reactions': stats.get('merged_raw'),
            'removed_invalid_smiles': stats.get('removed_invalid'),
            'removed_overlap_with_reaxys_test': stats.get('removed_overlap'),
            'removed_duplicates': stats.get('removed_duplicates'),
            'final_reactions': stats.get('after_dedup'),
            'train': stats.get('train'),
            'valid': stats.get('val'),
        }

    merged = []
    for name in ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']:
        path = raw_dir / name
        if path.exists():
            merged.append(pd.read_csv(path))
    if not merged:
        return {}
    raw_df = pd.concat(merged, ignore_index=True)
    raw_reactions = len(raw_df)
    rxn_series = raw_df['reactants>reagents>production'].astype(str).str.replace(r':\d+', '', regex=True)
    split = rxn_series.str.split('>>', n=1, expand=True)
    work = pd.DataFrame({'reactants': split[0], 'product': split[1]})
    work['canonical_key'] = work.apply(lambda row: _canonical_route_key(row['reactants'], row['product']), axis=1)
    valid_df = work.loc[work['canonical_key'] != ''].reset_index(drop=True)
    stage2_test_keys = set()
    for family in REACTION_TYPES:
        test_file = repo_root / 'data' / f'reaction_processed_{family}_catmerge' / 'For_second_part_model' / 'Splitted_second_test_labels_processed.txt'
        if not test_file.exists():
            continue
        for row in load_split_rows(test_file):
            route_key = _canonical_route_key(row['reactants'], row['product'])
            if route_key:
                stage2_test_keys.add(route_key)
    after_overlap = valid_df.loc[~valid_df['canonical_key'].isin(stage2_test_keys)].reset_index(drop=True)
    dedup_df = after_overlap.drop_duplicates(subset='canonical_key', keep='first').reset_index(drop=True)

    filtered_counts = {}
    for split_name in ['train', 'val']:
        path = filtered_raw_dir / f'raw_{split_name}.csv'
        if not path.exists():
            path = filtered_dir / f'raw_{split_name}.csv'
        filtered_counts[split_name] = len(pd.read_csv(path)) if path.exists() else 0

    return {
        'raw_reactions': raw_reactions,
        'removed_invalid_smiles': raw_reactions - len(valid_df),
        'removed_overlap_with_reaxys_test': len(valid_df) - len(after_overlap),
        'removed_duplicates': len(after_overlap) - len(dedup_df),
        'final_reactions': len(dedup_df),
        'train': filtered_counts.get('train', 0),
        'valid': filtered_counts.get('val', 0),
    }


def collect_stage1_route_dataset_stats(repo_root: Path, families: list[str]) -> list[dict]:
    rows: list[dict] = []
    input_root = repo_root / 'data' / 'reaxys_input'
    for family in families:
        raw_family_files = sorted(
            path for path in (input_root / family).iterdir()
            if path.is_file() and path.suffix.lower() in {'.csv', '.xlsx'} and not path.name.startswith('~$')
        )
        raw_routes: set[str] = set()
        stage2_complete_routes = set()
        for path in raw_family_files:
            frame = raw_load_file(str(path))
            if 'Reaction' not in frame.columns:
                continue
            split = frame['Reaction'].astype(str).str.split('>>', n=1, expand=True)
            frame['reactants'] = split[0]
            frame['products'] = split[1]
            if 'Number of Reaction Steps' in frame.columns:
                frame = frame.loc[frame['Number of Reaction Steps'] == 1].reset_index(drop=True)
            for _, row in frame.iterrows():
                reactants = str(row.get('reactants', ''))
                product = str(row.get('products', ''))
                if not is_valid_smiles(reactants) or not is_valid_smiles(product):
                    continue
                route_key = _canonical_route_key(reactants, product)
                if route_key:
                    raw_routes.add(route_key)
        for split in ['train', 'val', 'test']:
            for row in load_split_rows(split_file_for_family(repo_root, family, split)):
                route_key = _canonical_route_key(row['reactants'], row['product'])
                if route_key:
                    stage2_complete_routes.add(route_key)

        dataset_root = repo_root / 'data' / 'editretro' / 'datasets' / f'REAXYS_{family}_SINGLE_CATMERGE' / 'raw'
        train_df = pd.read_csv(dataset_root / 'raw_train.csv') if (dataset_root / 'raw_train.csv').exists() else pd.DataFrame()
        val_df = pd.read_csv(dataset_root / 'raw_val.csv') if (dataset_root / 'raw_val.csv').exists() else pd.DataFrame()
        test_df = pd.read_csv(dataset_root / 'raw_test.csv') if (dataset_root / 'raw_test.csv').exists() else pd.DataFrame()
        stage2_train_routes = len({
            f"{row['reaction_id']}|{row['reactants']}|{row['product']}"
            for row in load_split_rows(split_file_for_family(repo_root, family, 'train'))
        })
        rows.append(
            {
                'family': family,
                'single_step_route_records': len(raw_routes),
                'condition_complete_records': len(stage2_complete_routes),
                'extra_route_only_records_added_to_train': max(len(train_df) - stage2_train_routes, 0),
                'train_routes': len(train_df),
                'valid_routes': len(val_df),
                'test_routes': len(test_df),
            }
        )
    return rows


def collect_stage1_route_performance(route_root: Path, families: list[str]) -> tuple[list[dict], list[dict], list[dict]]:
    performance_rows: list[dict] = []
    base_vs_tuned_rows: list[dict] = []
    error_rows: list[dict] = []
    for family in families:
        cache_file = route_root / family / 'route_cache.json'
        if not cache_file.exists():
            continue
        cache = json.loads(cache_file.read_text(encoding='utf-8'))
        recall = stage1_route_recall(cache_file)
        reaction_rows = cache.get('reactions', [])
        route_counts = [len(row.get('routes', [])) for row in reaction_rows]

        generation_file = Path(cache.get('generation_file', ''))
        valid_smiles_rate = None
        duplicate_prediction_rate = None
        invalid_output_rate = None
        if generation_file.exists():
            predictions, _, _ = process_input(str(generation_file), False)
            total_predictions = len(predictions)
            valid_predictions = 0
            invalid_predictions = 0
            aug = int(cache.get('aug', 10))
            beam = int(cache.get('topk', 10))
            duplicates_per_product: list[float] = []
            for idx in range(0, total_predictions, aug * beam):
                block = predictions[idx:idx + aug * beam]
                canon = []
                for pred in block:
                    value = canonicalize_smiles_clear_map(pred, return_max_frag=False)
                    if value:
                        valid_predictions += 1
                        canon.append(value)
                    else:
                        invalid_predictions += 1
                if canon:
                    duplicates_per_product.append(1.0 - (len(set(canon)) / len(canon)))
            if total_predictions > 0:
                valid_smiles_rate = valid_predictions / total_predictions
                invalid_output_rate = invalid_predictions / total_predictions
            duplicate_prediction_rate = _mean(duplicates_per_product)

        correct_route_scores: list[float] = []
        no_route = 0
        for reaction in reaction_rows:
            gold = reaction.get('gold_reactants', '')
            if not reaction.get('routes'):
                no_route += 1
            gold_key = _canonicalize_side(gold)
            for route in reaction.get('routes', []):
                if _canonicalize_side(route.get('reactants', '')) == gold_key:
                    correct_route_scores.append(float(route.get('retro_score', 0.0)))
                    break

        performance_rows.append(
            {
                'family': family,
                'test_products': int(recall['n']),
                'route_at_1': recall.get('route_recall_top1'),
                'route_at_3': recall.get('route_recall_top3'),
                'route_at_5': recall.get('route_recall_top5'),
                'route_at_10': recall.get('route_recall_top10'),
                'valid_smiles_rate': valid_smiles_rate,
                'avg_unique_routes_per_product': float(np.mean(route_counts)) if route_counts else 0.0,
            }
        )
        error_rows.append(
            {
                'family': family,
                'invalid_output_rate': invalid_output_rate,
                'duplicate_prediction_rate': duplicate_prediction_rate,
                'no_route_generated_rate': (no_route / len(reaction_rows) if reaction_rows else 0.0),
                'gold_route_absent_top10_rate': 1.0 - recall.get('route_recall_top10', 0.0),
                'avg_route_score_of_correct_routes': _mean([float(score) for score in correct_route_scores]),
            }
        )
    return performance_rows, base_vs_tuned_rows, error_rows


def collect_base_vs_tuned(route_root: Path, base_route_root: Path, families: list[str]) -> list[dict]:
    rows: list[dict] = []
    for family in families:
        tuned_cache = route_root / family / 'route_cache.json'
        base_cache = base_route_root / family / 'route_cache.json'
        if not tuned_cache.exists() or not base_cache.exists():
            continue
        tuned = stage1_route_recall(tuned_cache)
        base = stage1_route_recall(base_cache)
        rows.append(
            {
                'family': family,
                'test_products': tuned.get('n'),
                'base_route_at_1': base.get('route_recall_top1'),
                'base_route_at_3': base.get('route_recall_top3'),
                'base_route_at_5': base.get('route_recall_top5'),
                'base_route_at_10': base.get('route_recall_top10'),
                'family_tuned_route_at_1': tuned.get('route_recall_top1'),
                'family_tuned_route_at_3': tuned.get('route_recall_top3'),
                'family_tuned_route_at_5': tuned.get('route_recall_top5'),
                'family_tuned_route_at_10': tuned.get('route_recall_top10'),
                'delta_route_at_1': tuned.get('route_recall_top1', 0.0) - base.get('route_recall_top1', 0.0),
                'delta_route_at_3': tuned.get('route_recall_top3', 0.0) - base.get('route_recall_top3', 0.0),
                'delta_route_at_5': tuned.get('route_recall_top5', 0.0) - base.get('route_recall_top5', 0.0),
                'delta_route_at_10': tuned.get('route_recall_top10', 0.0) - base.get('route_recall_top10', 0.0),
            }
        )
    return rows


def _candidate_route_context_stats(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            'test_samples': 0,
            'avg_candidate_systems_per_sample': 0.0,
            'avg_contexts_per_route': 0.0,
            'fallback_rate': 0.0,
        }
    systems_per_sample = frame.groupby('sample_index', sort=True).size().astype(float)
    route_group_cols = ['sample_index', 'reaction_id', 'reactants', 'product']
    contexts_per_route = frame.groupby(route_group_cols, sort=True).size().astype(float)
    route_knn = frame.groupby(route_group_cols, sort=True)['knn_similarity_max'].max()
    return {
        'test_samples': int(frame['sample_index'].nunique()),
        'avg_candidate_systems_per_sample': float(systems_per_sample.mean()) if not systems_per_sample.empty else 0.0,
        'avg_contexts_per_route': float(contexts_per_route.mean()) if not contexts_per_route.empty else 0.0,
        'fallback_rate': float((route_knn <= 0.0).mean()) if not route_knn.empty else 0.0,
    }


def collect_stage23_stats(repo_root: Path, mainline_root: Path, route_root: Path, families: list[str]) -> dict[str, list[dict]]:
    coverage_rows: list[dict] = []
    compression_rows: list[dict] = []
    support_rows: list[dict] = []
    ranking_rows: list[dict] = []
    loss_rows: list[dict] = []
    decomposed_rows: list[dict] = []
    temperature_rows: list[dict] = []

    for family in families:
        table_test = mainline_root / family / '_shared_knn' / 'training_tables' / 'test.csv'
        scored_test = mainline_root / family / 'knn_xgb' / 'non_oracle' / 'test_scored.csv'
        route_cache = route_root / family / 'route_cache.json'
        if not table_test.exists() or not scored_test.exists() or not route_cache.exists():
            continue
        table_frame = pd.read_csv(table_test)
        scored_frame = pd.read_csv(scored_test)
        recall = stage1_route_recall(route_cache)
        metrics = evaluate_scored_frame(scored_frame, score_column='xgb_score', temperature_column='xgb_temperature_pred')
        pool_stats = _candidate_route_context_stats(table_frame)
        train_rows = load_split_rows(split_file_for_family(repo_root, family, 'train'))
        global_contexts = len({(row['reagent_norm'], row['solvent_norm']) for row in train_rows})

        coverage_rows.append(
            {
                'family': family,
                'test_samples': pool_stats['test_samples'],
                'route_at_10': recall.get('route_recall_top10'),
                'avg_candidate_systems_per_sample': pool_stats['avg_candidate_systems_per_sample'],
                'avg_contexts_per_route': pool_stats['avg_contexts_per_route'],
                'route_coverage': metrics.get('pool_route_coverage'),
                'context_coverage': metrics.get('pool_context_coverage'),
                'full_system_pool_coverage': metrics.get('pool_coverage'),
            }
        )
        compression_rows.append(
            {
                'family': family,
                'global_train_contexts': global_contexts,
                'avg_retrieved_contexts_per_route': pool_stats['avg_contexts_per_route'],
                'avg_candidate_systems_per_product': pool_stats['avg_candidate_systems_per_sample'],
                'reduction_factor': (global_contexts / pool_stats['avg_contexts_per_route'] if pool_stats['avg_contexts_per_route'] > 0 else None),
                'pool_coverage': metrics.get('pool_coverage'),
            }
        )
        support_rows.append(
            {
                'family': family,
                'mean_max_similarity': _safe_numeric(pd.to_numeric(table_frame['knn_similarity_max'], errors='coerce').mean()) if 'knn_similarity_max' in table_frame.columns else None,
                'mean_similarity_sum': _safe_numeric(pd.to_numeric(table_frame['knn_similarity_sum'], errors='coerce').mean()) if 'knn_similarity_sum' in table_frame.columns else None,
                'mean_neighbor_count': _safe_numeric(pd.to_numeric(table_frame['knn_neighbor_count'], errors='coerce').mean()) if 'knn_neighbor_count' in table_frame.columns else None,
                'mean_weighted_yield': _safe_numeric(pd.to_numeric(table_frame['knn_weighted_mean_yield'], errors='coerce').mean()) if 'knn_weighted_mean_yield' in table_frame.columns else None,
                'fallback_rate': pool_stats['fallback_rate'],
            }
        )
        ranking_rows.append(
            {
                'family': family,
                'pool_coverage': metrics.get('pool_coverage'),
                'system_at_1': metrics.get('system_top1_all'),
                'system_at_3': metrics.get('system_top3_all'),
                'system_at_5': metrics.get('system_top5_all'),
                'system_at_10': metrics.get('system_top10_all'),
                'ndcg_at_10': metrics.get('system_ndcg10'),
                'mrr': metrics.get('system_mrr'),
                'num_slates': metrics.get('num_slates'),
            }
        )
        loss_rows.append(
            {
                'family': family,
                'pool_coverage': metrics.get('pool_coverage'),
                'system_at_10': metrics.get('system_top10_all'),
                'ranking_loss_at_10': metrics.get('pool_coverage', 0.0) - metrics.get('system_top10_all', 0.0),
                'system_at_1': metrics.get('system_top1_all'),
                'top1_loss': metrics.get('pool_coverage', 0.0) - metrics.get('system_top1_all', 0.0),
            }
        )
        decomposed_rows.append(
            {
                'family': family,
                'route_at_10': metrics.get('route_top10_all'),
                'context_at_10': metrics.get('context_top10_all'),
                'system_at_10': metrics.get('system_top10_all'),
                'gap_route_to_system': metrics.get('route_top10_all', 0.0) - metrics.get('system_top10_all', 0.0),
                'gap_context_to_system': metrics.get('context_top10_all', 0.0) - metrics.get('system_top10_all', 0.0),
            }
        )
        temp = metrics.get('temperature', {})
        temperature_rows.append(
            {
                'family': family,
                'temp_valid_samples': temp.get('n'),
                'temp_mae': temp.get('mae'),
                'temp_rmse': temp.get('rmse'),
                'temp_within_5c': temp.get('within_5c'),
                'temp_within_10c': temp.get('within_10c'),
                'temp_within_20c': temp.get('within_20c'),
            }
        )

    return {
        'coverage': coverage_rows,
        'compression': compression_rows,
        'support': support_rows,
        'ranking': ranking_rows,
        'loss': loss_rows,
        'decomposed': decomposed_rows,
        'temperature': temperature_rows,
    }


def append_macro_and_weighted(rows: list[dict], *, sample_key: str | None = None) -> list[dict]:
    if not rows:
        return rows
    metric_keys = [key for key in rows[0].keys() if key != 'family']
    macro = {'family': 'MACRO-AVG'}
    for key in metric_keys:
        macro[key] = _mean([_safe_numeric(row.get(key)) for row in rows])
    result = list(rows) + [macro]
    if sample_key is not None and sample_key in rows[0]:
        weighted = {'family': 'WEIGHTED-AVG'}
        for key in metric_keys:
            if key == sample_key:
                weighted[key] = sum(int(row.get(sample_key, 0) or 0) for row in rows)
            else:
                weighted[key] = _weighted_mean(rows, key, sample_key)
        result.append(weighted)
    return result


def render_markdown_report(sections: dict[str, list[dict]], output_file: Path) -> Path:
    lines = ['# Checklist Statistics', '']
    for title, rows in sections.items():
        lines.append(f'## {title}')
        lines.append('')
        if not rows:
            lines.append('No rows collected.')
            lines.append('')
            continue
        frame = pd.DataFrame(rows)
        lines.append(frame.to_markdown(index=False))
        lines.append('')
    output_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect checklist statistics for the maintained ProSys mainline.')
    parser.add_argument('--repo_root', type=str, default='.')
    parser.add_argument('--families', type=str, default='all')
    parser.add_argument('--output_root', type=str, default='outputs/checklist_stats')
    parser.add_argument('--route_root', type=str, default='outputs/stage1_routes')
    parser.add_argument('--base_route_root', type=str, default='outputs/stage1_routes_base')
    parser.add_argument('--mainline_root', type=str, default='outputs/stage23_mainline')
    parser.add_argument('--workers', type=int, default=1, help='Parallel family-cleanup workers; 1 preserves serial execution.')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    route_root = Path(args.route_root).resolve()
    base_route_root = Path(args.base_route_root).resolve()
    mainline_root = Path(args.mainline_root).resolve()
    families = parse_families_arg(args.families)

    print(f'[checklist] families={families}', flush=True)
    print('[checklist] simulate raw-to-clean pipeline', flush=True)
    raw_to_clean_rows, _, label_space_rows = simulate_clean_pipeline(repo_root, families, workers=max(1, args.workers))
    print('[checklist] collect split stats', flush=True)
    split_rows = collect_split_stats(repo_root, families)
    print('[checklist] collect USPTO stats', flush=True)
    uspto_stats = collect_uspto_stats(repo_root)
    print('[checklist] collect Stage1 route dataset stats', flush=True)
    stage1_dataset_rows = collect_stage1_route_dataset_stats(repo_root, families)
    print('[checklist] collect Stage1 route performance', flush=True)
    stage1_perf_rows, _, stage1_error_rows = collect_stage1_route_performance(route_root, families)
    print('[checklist] collect base-vs-tuned route stats', flush=True)
    base_vs_tuned_rows = collect_base_vs_tuned(route_root, base_route_root, families)
    print('[checklist] collect Stage2/Stage3 stats', flush=True)
    stage23 = collect_stage23_stats(repo_root, mainline_root, route_root, families)

    raw_to_clean_rows = append_macro_and_weighted(raw_to_clean_rows)
    label_space_rows = append_macro_and_weighted(label_space_rows)
    split_rows = append_macro_and_weighted(split_rows)
    stage1_dataset_rows = append_macro_and_weighted(stage1_dataset_rows)
    stage1_perf_rows = append_macro_and_weighted(stage1_perf_rows, sample_key='test_products')
    base_vs_tuned_rows = append_macro_and_weighted(base_vs_tuned_rows, sample_key='test_products')
    stage1_error_rows = append_macro_and_weighted(stage1_error_rows)
    for key in list(stage23):
        sample_key = 'num_slates' if key == 'ranking' else None
        stage23[key] = append_macro_and_weighted(stage23[key], sample_key=sample_key)

    _write_csv(raw_to_clean_rows, output_root / '01_raw_to_clean.csv')
    _write_csv(label_space_rows, output_root / '02_label_space.csv')
    _write_csv(split_rows, output_root / '03_split_stats.csv')
    _write_json(uspto_stats, output_root / '04_uspto_stats.json')
    _write_csv(stage1_dataset_rows, output_root / '05_stage1_route_dataset.csv')
    _write_csv(stage1_perf_rows, output_root / '06_stage1_route_performance.csv')
    _write_csv(base_vs_tuned_rows, output_root / '07_stage1_base_vs_tuned.csv')
    _write_csv(stage1_error_rows, output_root / '08_stage1_error_stats.csv')
    _write_csv(stage23['coverage'], output_root / '09_stage2_coverage.csv')
    _write_csv(stage23['compression'], output_root / '10_stage2_search_space.csv')
    _write_csv(stage23['support'], output_root / '11_stage2_knn_support.csv')
    _write_csv(stage23['ranking'], output_root / '12_stage3_ranking.csv')
    _write_csv(stage23['loss'], output_root / '13_stage3_loss_breakdown.csv')
    _write_csv(stage23['decomposed'], output_root / '14_stage3_decomposed.csv')
    _write_csv(stage23['temperature'], output_root / '15_temperature.csv')

    render_markdown_report(
        {
            'Raw-to-clean': raw_to_clean_rows,
            'Label space': label_space_rows,
            'Split stats': split_rows,
            'USPTO filtered': [uspto_stats] if uspto_stats else [],
            'Stage1 route dataset': stage1_dataset_rows,
            'Stage1 performance': stage1_perf_rows,
            'Stage1 base vs tuned': base_vs_tuned_rows,
            'Stage1 error stats': stage1_error_rows,
            'Stage2 coverage': stage23['coverage'],
            'Stage2 search-space compression': stage23['compression'],
            'Stage2 KNN support': stage23['support'],
            'Stage3 ranking': stage23['ranking'],
            'Stage3 loss breakdown': stage23['loss'],
            'Stage3 decomposed': stage23['decomposed'],
            'Temperature': stage23['temperature'],
        },
        output_root / 'checklist_stats.md',
    )
    print(f'[checklist] wrote {output_root}', flush=True)


if __name__ == '__main__':
    main()
