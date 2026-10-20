from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import librosa
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from utmosv2._settings._config import Config

INFERENCE_PHASES = frozenset({"test", "prediction"})
"""Phases where the segment selection is deterministic and covers the whole audio.

Note that `"valid"` is intentionally excluded: validation during training must keep
the same random cropping as training so that model selection stays comparable, and
its dataloader uses `cfg.batch_size`, which cannot collate variable-length samples.
"""


def load_audio(cfg: Config, file: Path) -> np.ndarray:
    try:
        y, sr = librosa.load(file, sr=None)
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sr)
    except Exception:
        y = np.load(file)
    return y


def extend_audio(y: np.ndarray, length: int, method: str) -> np.ndarray:
    if y.shape[0] > length:
        return y
    elif method == "tile":
        n = length // y.shape[0] + 1
        y = np.tile(y, n)
        return y
    else:
        raise NotImplementedError


def select_random_start(y: np.ndarray, length: int) -> np.ndarray:
    start = np.random.randint(0, y.shape[0] - length)
    return y[start : start + length]


def num_starts_to_cover(
    total: int, length: int, hop: int, multiple_of: int = 1
) -> int:
    """Return how many windows of `length` are needed to cover `total` with `hop` stride.

    The result is rounded up to a multiple of `multiple_of` so that the windows can be
    grouped without duplicating or dropping any of them.

    Args:
        total (int): Total number of samples to cover.
        length (int): Window length in samples.
        hop (int): Maximum stride between consecutive windows, in samples.
        multiple_of (int): Round the count up to a multiple of this value.
    """
    span = total - length
    n = 1 if span <= 0 else math.ceil(span / hop) + 1
    return math.ceil(n / multiple_of) * multiple_of


def select_starts(y: np.ndarray, length: int, n: int) -> list[int]:
    """Return `n` evenly spaced start positions covering the whole of `y`.

    Both ends are always included, so the effective hop is at most the one requested by
    `num_starts_to_cover`. Unlike `select_random_start`, this is deterministic and never
    raises when `y` is exactly `length` samples long.
    """
    span = y.shape[0] - length
    if span <= 0:
        return [0] * n
    if n == 1:
        return [span // 2]
    return [round(i * span / (n - 1)) for i in range(n)]


def get_dataset_map(cfg: Config) -> dict[str, int]:
    if cfg.data_config:
        with open(cfg.data_config, "r") as f:
            datasets = json.load(f)
        return {d["name"]: i for i, d in enumerate(datasets["data"])}
    else:
        return {
            "bvcc": 0,
            "sarulab": 1,
            "blizzard2008": 2,
            "blizzard2009": 3,
            "blizzard2010-EH1": 4,
            "blizzard2010-EH2": 5,
            "blizzard2010-ES1": 6,
            "blizzard2010-ES3": 7,
            "blizzard2011": 8,
            "somos": 9,
        }


def get_dataset_num(cfg: Config) -> int:
    return len(get_dataset_map(cfg))
