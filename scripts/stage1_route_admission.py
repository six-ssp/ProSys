"""Verify retained guarded-inference evidence before downstream publication."""

import json
from pathlib import Path

from scripts.audit_stage1_base_augmented import sha

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'product_identity_fallback_v1'


def verify_guard(cache_path, root=ROOT):
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text())
    path = cache_path.parent / 'augmentation_guard.json'
    guard = json.loads(path.read_text())
    reference = cache.get('augmentation_guard', {})
    if (guard.get('pass') is not True or guard.get('protocol') != PROTOCOL or
            reference.get('protocol') != PROTOCOL or reference.get('receipt_sha256') != sha(path)):
        raise ValueError('Missing or mismatched guarded-inference receipt')
    count, aug = len(cache['reactions']), cache['aug']
    events = guard.get('events', [])
    if (type(aug) is not int or aug < 1 or count <= 0 or guard.get('query_count') != count or
            guard.get('augmented_input_count') != count * aug or len(events) != count):
        raise ValueError('Guarded-inference support differs from route cache')
    for index, event in enumerate(events):
        slots = event.get('replaced_slots', [])
        if (event.get('query_index') != index or event.get('augmentation_count') != aug or
                type(event.get('input_normalized')) is not bool or
                any(type(slot) is not int or slot < 1 or slot >= aug for slot in slots) or
                slots != sorted(set(slots))):
            raise ValueError('Invalid per-query augmentation evidence')
    replaced = sum(len(event['replaced_slots']) for event in events)
    if (guard.get('replaced_variant_count') != replaced or reference.get('replaced_variant_count') != replaced or
            guard.get('normalized_input_count') != sum(event['input_normalized'] for event in events)):
        raise ValueError('Guard aggregate counts differ from individual events')
    sources = guard.get('source_sha256', {})
    if not sources:
        raise ValueError('Missing guard source bindings')
    for name, digest in sources.items():
        source = (Path(root) / name).resolve()
        source.relative_to(Path(root).resolve())
        if sha(source) != digest:
            raise ValueError('Guard source changed: ' + name)
    if reference.get('wrapper_sha256') != sha(Path(root) / 'scripts/build_stage1_guarded_routes.py'):
        raise ValueError('Guarded route builder changed')
    return {'protocol': PROTOCOL, 'query_count': count, 'guard_sha256': sha(path),
            'route_cache_sha256': sha(cache_path), 'source_sha256': sources,
            'replaced_variants': replaced, 'normalized_queries': guard['normalized_input_count']}


def require_paired_checkpoints(test_cache, validation_cache):
    test, validation = (json.loads(Path(path).read_text()) for path in (test_cache, validation_cache))
    paths = [Path(cache['checkpoint']).resolve() for cache in (test, validation)]
    if paths[0] != paths[1] or test['family'] != validation['family']:
        raise ValueError('Validation/test routes must use the same family expert checkpoint')
    if any(test[key] != validation[key] for key in ('aug', 'topk', 'n_best')):
        raise ValueError('Validation/test decode settings differ')
    return sha(paths[0])
