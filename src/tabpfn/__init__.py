"""TabPFN: A transformer that solves small tabular classification problems in seconds.

This package provides a scikit-learn compatible interface to TabPFN,
a prior-fitted network for tabular data classification and regression.
"""

from __future__ import annotations

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tabpfn")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from tabpfn.classifier import TabPFNClassifier
from tabpfn.regressor import TabPFNRegressor

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
]
