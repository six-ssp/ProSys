"""Archived Stage 2 code kept for legacy FNN/neural-V2 reproduction."""

from __future__ import annotations

import sys

# Keep historical absolute imports like `from stage2.train_multilabel ...`
# working after the archive moved under Experiment/legacy_stage2.
sys.modules.setdefault('stage2', sys.modules[__name__])
