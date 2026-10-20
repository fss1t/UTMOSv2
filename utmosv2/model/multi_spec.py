from __future__ import annotations

from typing import TYPE_CHECKING, cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utmosv2.dataset._utils import get_dataset_num

if TYPE_CHECKING:
    from utmosv2._settings._config import Config


def _split_frame_groups(x: torch.Tensor, per_group: int) -> torch.Tensor:
    """Fold groups of frames into the batch dimension.

    At inference a sample carries `num_groups * per_group` frames covering the whole
    audio, whereas the model was trained on exactly `per_group` frames. Stacking the
    groups along the batch axis keeps each group's pooling statistics identical to
    training (`torch.max` over the concatenated frames grows with the number of frames,
    so a single wide forward would bias the output by the duration) and lets the
    backbones run with a batch of `num_groups`.

    Args:
        x (torch.Tensor): Input of shape (1, num_groups * per_group, channels, height, width).
        per_group (int): Number of frames the model was trained with.

    Returns:
        torch.Tensor: Input of shape (num_groups, per_group, channels, height, width).
    """
    if x.shape[0] != 1:
        raise ValueError(
            "Whole-audio inference requires a batch size of 1, "
            f"but got a batch of {x.shape[0]}."
        )
    if x.shape[1] % per_group != 0:
        raise ValueError(
            f"The number of frames ({x.shape[1]}) must be a multiple of "
            f"num_frames * len(specs) ({per_group})."
        )
    return x.reshape(-1, per_group, *x.shape[2:])


class MultiSpecModelV2(nn.Module):
    """
    A multi-spectrogram model (version 2) that processes multiple spectrograms
    and combines their outputs using learnable weights. This model supports
    attention-based pooling and a flexible number of spectrogram frames.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing model and dataset settings.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbones = nn.ModuleList(
            [
                timm.create_model(
                    cfg.model.multi_spec.backbone,
                    pretrained=True,
                    num_classes=0,
                )
                for _ in range(len(cfg.dataset.specs))
            ]
        )
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1) if self.cfg.model.multi_spec.atten else 1,  # type: ignore
            pool_type=self.cfg.model.multi_spec.pool_type,
            flatten=False,
        )

        if self.cfg.model.multi_spec.atten:
            self.attn = nn.MultiheadAttention(
                embed_dim=cast(int, self.backbones[0].num_features)
                * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1),
                num_heads=8,
                dropout=0.2,
                batch_first=True,
            )

        fc_in_features = (
            cast(int, self.backbones[0].num_features)
            * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1)
            * (2 if self.cfg.model.multi_spec.atten else 1)
        )

        self.fc: nn.Linear | nn.Identity = nn.Linear(
            fc_in_features, cfg.model.multi_spec.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiSpecModelV2.

        Args:
            x (torch.Tensor):
                Input tensor of shape (batch_size, num_frames, channels, width, height).
                At inference the second dimension may hold several groups of `num_frames`
                covering the whole audio, in which case the groups are averaged.

        Returns:
            torch.Tensor:
                Output tensor after applying backbones, pooling, and fully connected layers.
        """
        per_group = self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
        grouped = x.shape[1] != per_group
        if grouped:
            x = _split_frame_groups(x, per_group)
        xl = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        xl = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(xl)
        ]
        xl = [
            sum(
                [
                    xl[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(xl, dim=3)
        x = self.pooling(x).squeeze(3)
        if self.cfg.model.multi_spec.atten:
            xt = torch.permute(x, (0, 2, 1))
            y, _ = self.attn(xt, xt, xt)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(x)
        if grouped:
            x = x.mean(dim=0, keepdim=True)
        return x


class MultiSpecExtModel(nn.Module):
    """
    An extended version of the MultiSpecModel that incorporates data-domain id
    in addition to the spectrograms. This model allows the fusion of
    data-domain embeddings with multi-spectrogram features.

    Args:
        cfg (SimpleNamespace | ModuleType):
            Configuration object containing model and dataset settings.

    Returns:
        torch.Tensor:
            The model's output after processing the input and data-domain id
            through backbones, pooling, and fully connected layers.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbones = nn.ModuleList(
            [
                timm.create_model(
                    cfg.model.multi_spec.backbone,
                    pretrained=True,
                    num_classes=0,
                )
                for _ in range(len(cfg.dataset.specs))
            ]
        )
        for backbone in self.backbones:
            backbone.global_pool = nn.Identity()

        self.weights = nn.Parameter(
            F.softmax(torch.randn(len(cfg.dataset.specs)), dim=0)
        )

        self.pooling = timm.layers.SelectAdaptivePool2d(
            output_size=(None, 1) if self.cfg.model.multi_spec.atten else 1,  # type: ignore
            pool_type=self.cfg.model.multi_spec.pool_type,
            flatten=False,
        )

        if self.cfg.model.multi_spec.atten:
            self.attn = nn.MultiheadAttention(
                embed_dim=cast(int, self.backbones[0].num_features)
                * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1),
                num_heads=8,
                dropout=0.2,
                batch_first=True,
            )

        fc_in_features = (
            cast(int, self.backbones[0].num_features)
            * (2 if self.cfg.model.multi_spec.pool_type == "catavgmax" else 1)
            * (2 if self.cfg.model.multi_spec.atten else 1)
        )

        self.num_dataset = get_dataset_num(cfg)

        self.fc: nn.Linear | nn.Identity = nn.Linear(
            fc_in_features + self.num_dataset, cfg.model.multi_spec.num_classes
        )

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        per_group = self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
        grouped = x.shape[1] != per_group
        if grouped:
            x = _split_frame_groups(x, per_group)
            d = d.expand(x.shape[0], -1)
        xl = [
            x[:, i, :, :, :].squeeze(1)
            for i in range(
                self.cfg.dataset.spec_frames.num_frames * len(self.cfg.dataset.specs)
            )
        ]
        xl = [
            self.backbones[i % len(self.cfg.dataset.specs)](t) for i, t in enumerate(xl)
        ]
        xl = [
            sum(
                [
                    xl[i * len(self.cfg.dataset.specs) + j] * w
                    for j, w in enumerate(self.weights)
                ]
            )
            for i in range(self.cfg.dataset.spec_frames.num_frames)
        ]
        x = torch.cat(xl, dim=3)
        x = self.pooling(x).squeeze(3)
        if self.cfg.model.multi_spec.atten:
            xt = torch.permute(x, (0, 2, 1))
            y, _ = self.attn(xt, xt, xt)
            x = torch.cat([torch.mean(y, dim=1), torch.max(x, dim=2).values], dim=1)
        x = self.fc(torch.cat([x, d], dim=1))
        if grouped:
            # `fc` is affine, so averaging here is identical to averaging the predictions
            # of each group. When this model is the spectrogram branch of a fusion model
            # its `fc` is an Identity and the averaged features are what gets returned.
            x = x.mean(dim=0, keepdim=True)
        return x
