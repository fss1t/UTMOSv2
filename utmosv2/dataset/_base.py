from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

from utmosv2.dataset._schema import InMemoryData
from utmosv2.dataset._utils import INFERENCE_PHASES, get_dataset_map, load_audio
from utmosv2.preprocess._preprocess import remove_silent_section

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from utmosv2._settings._config import Config
    from utmosv2.dataset._schema import DatasetItem


class BaseDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
        self,
        cfg: Config,
        data: pd.DataFrame | list[DatasetItem] | InMemoryData,
        phase: str,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.data = data
        self.phase = phase
        self.transform = transform

    def _get_audio_and_mos(self, idx: int) -> tuple[np.ndarray, torch.Tensor]:
        if isinstance(self.data, InMemoryData):
            y = self.data.data[idx]
            mos = None
        else:
            row = self.data[idx] if isinstance(self.data, list) else self.data.iloc[idx]
            file = row.file_path
            assert file is not None
            y = load_audio(self.cfg, file)
            mos = row.mos

        if getattr(self.cfg.dataset, "remove_silent_section", None):
            y = remove_silent_section(y)

        if self.phase in INFERENCE_PHASES:
            y = self._limit_duration(y)

        return y, torch.tensor(mos or 0.0, dtype=torch.float32)

    def _limit_duration(self, y: np.ndarray) -> np.ndarray:
        """Crop the head of `y` so that the whole-audio inference stays tractable.

        At inference the SSL branch consumes the audio without cropping, and its
        self-attention is O(T^2) in the number of frames (~50 frames per second), so an
        unbounded input can exhaust memory on long recordings. Applying the cap here in
        the shared loading path also bounds the mel branch, whose number of windows grows
        linearly with the duration, and keeps both branches looking at the same region.
        """
        max_duration = getattr(self.cfg.dataset, "max_duration", None) or 60.0
        max_len = int(max_duration * self.cfg.sr)
        if y.shape[0] <= max_len:
            return y
        warnings.warn(
            f"The audio is {y.shape[0] / self.cfg.sr:.1f} seconds long, which exceeds "
            f"max_duration ({max_duration:.1f} seconds). "
            f"Only the first {max_duration:.1f} seconds are used for the prediction."
        )
        return y[:max_len]

    def __len__(self) -> int:
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        pass


class DataDomainMixin:
    data: pd.DataFrame | list[DatasetItem] | InMemoryData

    def __init__(self, cfg: Config) -> None:
        self.dataset_map = get_dataset_map(cfg)

    def _get_data_domain_embedding(self, idx: int) -> torch.Tensor:
        dataset_name = (
            self.data[idx].dataset_name
            if isinstance(self.data, list)
            else self.data.dataset_name
            if isinstance(self.data, InMemoryData)
            else self.data.iloc[idx].dataset
        )

        d = torch.zeros(len(self.dataset_map), dtype=torch.float32)
        d[self.dataset_map[dataset_name]] = 1.0
        return d
