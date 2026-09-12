"""Global reproducibility helpers.

A single `set_seed()` call seeds Python's `random`, NumPy, and (if installed)
PyTorch/CUDA, so that stratified sampling, TF-IDF fitting order, LinearSVC,
and transformer training all become reproducible given the same seed.
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed every RNG source used anywhere in this project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic where possible. This can slow training
        # down slightly but keeps runs reproducible across reruns/resumes.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
