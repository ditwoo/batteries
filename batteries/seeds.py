import random
import numpy as np
from packaging.version import parse, Version


def seed_all(seed: int) -> None:
    """Fix all seeds so results can be reproducible.

    Args:
        seed (int): random seed
    """
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        if parse(tf.__version__) >= Version("2.0.0"):
            tf.random.set_seed(seed)
        elif parse(tf.__version__) <= Version("1.13.2"):
            tf.set_random_seed(seed)
        else:
            tf.compat.v1.set_random_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
