from __future__ import annotations

import os
import random
import torch
import numpy as np


DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
