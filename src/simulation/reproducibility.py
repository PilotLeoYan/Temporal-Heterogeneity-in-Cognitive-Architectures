import os
import platform
import random
import sys
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_seeds(base_seed: int, count: int) -> list[int]:
    ss = np.random.SeedSequence(base_seed)
    return [int(s) for s in ss.generate_state(count)]


def get_environment_metadata() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": np.__version__,
    }


@contextmanager
def reproducible_scope(seed: int) -> Generator[None]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
