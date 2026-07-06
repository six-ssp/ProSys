"""Utilities for parsing family-specific Stage 2 reaction-condition tables."""

from __future__ import annotations

from typing import Iterable, List


def remove_duplicated_records(records: str) -> str:
    """Remove duplicate labels while preserving their original order."""
    return '; '.join(dict.fromkeys(str(records).split('; ')))


def sort_out_data(data: Iterable[str | List[str]]) -> list[list]:
    """
    Group a flat condition table into reaction-level records.

    Input rows are expected in the following tab-separated layout:
    ``reaction_id, reactants, product, yield, reagent, solvent, temperature``.
    Consecutive rows with the same reaction id are merged into a single reaction
    entry with a context list.
    """
    rows: list[list[str]] = []
    for line in data:
        if isinstance(line, str):
            parts = line.strip('\n').split('\t')
        else:
            parts = list(line)
        if len(parts) < 7:
            continue
        rows.append(parts[:7])

    sorted_data = []
    rxn_id = ''
    current_index = -1

    for row in rows:
        row_reaction_id, reactants, product, yield_, reagent, solvent, temp = row
        reagent = remove_duplicated_records(reagent)
        solvent = remove_duplicated_records(solvent)

        if rxn_id != row_reaction_id:
            current_index += 1
            sorted_data.append([row_reaction_id, reactants, product, []])
            sorted_data[current_index][3].append((yield_, reagent, solvent, temp))
            rxn_id = row_reaction_id
            last_reagent = reagent
            last_solvent = solvent
            continue

        if last_reagent == reagent and last_solvent == solvent:
            continue

        sorted_data[current_index][3].append((yield_, reagent, solvent, temp))
        last_reagent = reagent
        last_solvent = solvent

    return sorted_data
