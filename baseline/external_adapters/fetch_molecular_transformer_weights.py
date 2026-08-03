"""Download a verified MolecularTransformer checkpoint from an explicit direct URL.

The historical IBM Box collection linked by the upstream README is no longer a
working direct download. This command intentionally requires both a direct URL
and its SHA256 checksum before it writes a checkpoint into the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

from .contracts import project_root


def download_verified(url: str, destination: Path, expected_sha256: str) -> Path:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
        raise ValueError('sha256 must be a 64-character hexadecimal digest.')

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={'User-Agent': 'ProSys-baseline-fetcher/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
        observed = digest.hexdigest()
        if observed != expected:
            raise ValueError(f'SHA256 mismatch: expected {expected}, got {observed}')
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Verified direct checkpoint URL, not a collection page.')
    parser.add_argument('--sha256', required=True, help='Expected SHA256 for the downloaded checkpoint.')
    parser.add_argument(
        '--output',
        type=Path,
        default=project_root() / 'baseline' / 'MolecularTransformer' / 'checkpoints' / 'model.pt',
    )
    args = parser.parse_args()
    destination = download_verified(args.url, args.output, args.sha256)
    print(f'Downloaded verified checkpoint to {destination}')


if __name__ == '__main__':
    main()
