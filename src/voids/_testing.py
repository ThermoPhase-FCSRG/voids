"""Internal testing utilities for voids package."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Sets seeds for both the standard library's random module and NumPy's
    random number generator.
    
    Parameters
    ----------
    seed : int
        Seed value for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
