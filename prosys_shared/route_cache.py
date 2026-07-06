"""Route-record helpers shared by the maintained ProSys pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RouteRecord:
    sample_index: int
    reaction_id: str
    reactants: str
    product: str
    family: str
    retro_rank: int = 1
    retro_score: float = 1.0
    retro_probability: float = 1.0


def load_route_records_from_split(split_file: str | Path, family: str) -> list[RouteRecord]:
    records: list[RouteRecord] = []
    seen = set()
    with open(split_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            reaction_id, reactants, product = parts[:3]
            dedup_key = (reaction_id, reactants, product)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            records.append(
                RouteRecord(
                    sample_index=len(records),
                    reaction_id=reaction_id,
                    reactants=reactants,
                    product=product,
                    family=family,
                )
            )
    return records


def load_route_records_from_cache(route_cache_file: str | Path, family: str) -> list[RouteRecord]:
    """Load Non-Oracle route records from a Stage 1 route cache."""

    import json

    with open(route_cache_file, 'r', encoding='utf-8') as handle:
        cache = json.load(handle)

    records: list[RouteRecord] = []
    for reaction in cache.get('reactions', []):
        sample_index = int(reaction['sample_index'])
        reaction_id = str(reaction['reaction_id'])
        product = str(reaction['product'])
        for route in reaction.get('routes', []):
            records.append(
                RouteRecord(
                    sample_index=sample_index,
                    reaction_id=reaction_id,
                    reactants=str(route['reactants']),
                    product=product,
                    family=family,
                    retro_rank=int(route.get('retro_rank', 1)),
                    retro_score=float(route.get('retro_score', 1.0)),
                    retro_probability=float(route.get('retro_probability', 1.0)),
                )
            )
    return records
