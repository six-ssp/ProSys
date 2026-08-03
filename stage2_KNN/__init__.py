"""Compatibility shim for the maintained Stage 2 package.

The current implementation lives in ``stage2_ReaFNN``. This package keeps the
historical import path ``stage2_KNN`` working for scripts that still reference
it.
"""

from stage2_ReaFNN import KNNContextPoolBuilder, ReaFNNConfig, ReaFNNSelector, train_reafnn_selector

__all__ = ['KNNContextPoolBuilder', 'ReaFNNConfig', 'ReaFNNSelector', 'train_reafnn_selector']
