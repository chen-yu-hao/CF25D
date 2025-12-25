from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_mn15_fit_file(
    file_path: str | Path,
    *,
    expected_n_values: int = 78,
    dtype: Any = np.float64,
) -> dict[str, np.ndarray]:
    """Read one `Database_MN15/fit/*` dataset file into a dict.

    Each non-empty line is expected to be:
      <name> <v1> <v2> ... <v_expected_n_values>

    Parameters
    ----------
    file_path
        Path to a single dataset file such as
        `CF25D-fit/Database_MN15/fit/NC15`.
    expected_n_values
        Number of float values after the name on each line. The MN15 fit files
        in this project use 78.
    dtype
        Numpy dtype (or dtype-like) used to store the parsed values.

    Returns
    -------
    dict
        Mapping `{system_name: values}`, where `values.shape == (expected_n_values,)`.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    out: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                name, rest = line.split(maxsplit=1)
            except ValueError:
                raise ValueError(f"Invalid line (missing values) at {path}:{line_num}") from None

            values = np.fromstring(rest, sep=" ", dtype=dtype)
            if values.size != expected_n_values:
                raise ValueError(
                    f"Expected {expected_n_values} floats at {path}:{line_num}, got {values.size}"
                )
            out[name] = values

    return out

