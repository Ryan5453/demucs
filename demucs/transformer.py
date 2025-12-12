# Copyright (c) 2019-present, Meta, Inc.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# First author is Simon Rouard.

import math
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_sin_embedding(
    length: int, dim: int, shift: int = 0, device="cpu", max_period=10000
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model)
    )
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe[None, :].to(device)


def create_sin_embedding_cape(
    length: int,
    dim: int,
    batch_size: int,
    mean_normalize: bool,
    augment: bool,  # True during training
    max_global_shift: float = 0.0,  # delta max
    max_local_shift: float = 0.0,  # epsilon max
    max_scale: float = 1.0,
    device: str = "cpu",
    max_period: float = 10000.0,
):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = 1.0 * torch.arange(length).view(-1, 1, 1)  # (length, 1, 1)
    pos = pos.repeat(1, batch_size, 1)  # (length, batch_size, 1)
    if mean_normalize:
        pos -= torch.nanmean(pos, dim=0, keepdim=True)

    pos = pos.to(device)

    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    ).float()


class ScaledEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 1.0,
        boost: float = 3.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data *= scale / boost
        self.boost = boost

    @property
    def weight(self):
        return self.embedding.weight * self.boost

    def forward(self, x):
        return self.embedding(x) * self.boost


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class MyGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: (B, T, C)
        if num_groups=1: Normalisation on all T and C together for each B
        """
        x = x.transpose(1, 2)
        return super().forward(x).transpose(1, 2)


class MyTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        group_norm=0,
        norm_first=False,
        norm_out=False,
        layer_norm_eps=1e-5,
        layer_scale=False,
        init_values=1e-4,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )
        if group_norm:
            self.norm1 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)
        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        if batch_first = False, src shape is (T, B, C)
        the case where batch_first=True is not covered
        """
        x = src

        if self.norm_first:
            x = x + self.gamma_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            )
            x = x + self.gamma_2(self._ff_block(self.norm2(x)))

            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(
                x + self.gamma_1(self._sa_block(x, src_mask, src_key_padding_mask))
            )
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x


class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation=F.relu,
        layer_norm_eps: float = 1e-5,
        layer_scale: bool = False,
        init_values: float = 1e-4,
        norm_first: bool = False,
        group_norm: bool = False,
        norm_out: bool = False,
        device=None,
        dtype=None,
        batch_first=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.cross_attn: nn.Module
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1: nn.Module
        self.norm2: nn.Module
        self.norm3: nn.Module
        if group_norm:
            self.norm1 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm2 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
            self.norm3 = MyGroupNorm(
                int(group_norm), d_model, eps=layer_norm_eps, **factory_kwargs
            )
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
            self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.norm_out = None
        if self.norm_first & norm_out:
            self.norm_out = MyGroupNorm(num_groups=int(norm_out), num_channels=d_model)

        self.gamma_1 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )
        self.gamma_2 = (
            LayerScale(d_model, init_values, True) if layer_scale else nn.Identity()
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    def forward(self, q, k, mask=None):
        """
        Args:
            q: tensor of shape (T, B, C)
            k: tensor of shape (S, B, C)
            mask: tensor of shape (T, S)

        """
        if self.norm_first:
            x = q + self.gamma_1(self._ca_block(self.norm1(q), self.norm2(k), mask))
            x = x + self.gamma_2(self._ff_block(self.norm3(x)))
            if self.norm_out:
                x = self.norm_out(x)
        else:
            x = self.norm1(q + self.gamma_1(self._ca_block(q, k, mask)))
            x = self.norm2(x + self.gamma_2(self._ff_block(x)))

        return x

    # cross-attention block
    def _ca_block(self, q, k, attn_mask=None):
        x = self.cross_attn(q, k, k, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# ----------------- MULTI-BLOCKS MODELS: -----------------------


class CrossTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        emb: str = "sin",
        hidden_scale: float = 4.0,
        num_heads: int = 8,
        num_layers: int = 6,
        cross_first: bool = False,
        dropout: float = 0.0,
        max_positions: int = 1000,
        norm_in: bool = True,
        norm_in_group: bool = False,
        group_norm: int = False,
        norm_first: bool = False,
        norm_out: bool = False,
        max_period: float = 10000.0,
        weight_decay: float = 0.0,
        lr: float | None = None,
        layer_scale: bool = False,
        gelu: bool = True,
        sin_random_shift: int = 0,
        weight_pos_embed: float = 1.0,
        cape_mean_normalize: bool = True,
        cape_augment: bool = True,
        cape_glob_loc_scale: list = [5000.0, 1.0, 1.4],
    ):
        super().__init__()
        """
        """
        assert dim % num_heads == 0

        hidden_dim = int(dim * hidden_scale)

        self.num_layers = num_layers
        # classic parity = 1 means that if idx%2 == 1 there is a
        # classical encoder else there is a cross encoder
        self.classic_parity = 1 if cross_first else 0
        self.emb = emb
        self.max_period = max_period
        self.weight_decay = weight_decay
        self.weight_pos_embed = weight_pos_embed
        self.sin_random_shift = sin_random_shift
        if emb == "cape":
            self.cape_mean_normalize = cape_mean_normalize
            self.cape_augment = cape_augment
            self.cape_glob_loc_scale = cape_glob_loc_scale
        if emb == "scaled":
            self.position_embeddings = ScaledEmbedding(max_positions, dim, scale=0.2)

        self.lr = lr

        activation: Any = F.gelu if gelu else F.relu

        self.norm_in: nn.Module
        self.norm_in_t: nn.Module
        if norm_in:
            self.norm_in = nn.LayerNorm(dim)
            self.norm_in_t = nn.LayerNorm(dim)
        elif norm_in_group:
            self.norm_in = MyGroupNorm(int(norm_in_group), dim)
            self.norm_in_t = MyGroupNorm(int(norm_in_group), dim)
        else:
            self.norm_in = nn.Identity()
            self.norm_in_t = nn.Identity()

        # spectrogram layers
        self.layers = nn.ModuleList()
        # temporal layers
        self.layers_t = nn.ModuleList()

        kwargs_common = {
            "d_model": dim,
            "nhead": num_heads,
            "dim_feedforward": hidden_dim,
            "dropout": dropout,
            "activation": activation,
            "group_norm": group_norm,
            "norm_first": norm_first,
            "norm_out": norm_out,
            "layer_scale": layer_scale,
            "batch_first": True,
        }

        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(MyTransformerEncoderLayer(**kwargs_common))
                self.layers_t.append(
                    MyTransformerEncoderLayer(**kwargs_common)
                )

            else:
                self.layers.append(CrossTransformerEncoderLayer(**kwargs_common))

                self.layers_t.append(
                    CrossTransformerEncoderLayer(**kwargs_common)
                )

    def forward(self, x, xt):
        B, C, Fr, T1 = x.shape
        pos_emb_2d = create_2d_sin_embedding(
            C, Fr, T1, x.device, self.max_period
        )  # (1, C, Fr, T1)
        pos_emb_2d = pos_emb_2d.permute(0, 2, 3, 1).reshape(
            B, T1 * Fr, C
        )  # "b c fr t1 -> b (t1 fr) c"
        x = x.permute(0, 2, 3, 1).reshape(B, T1 * Fr, C)  # "b c fr t1 -> b (t1 fr) c"
        x = self.norm_in(x)
        x = x + self.weight_pos_embed * pos_emb_2d

        B, C, T2 = xt.shape
        xt = xt.permute(0, 2, 1)  # "b c t2 -> b t2 c"
        pos_emb = self._get_pos_embedding(T2, B, C, x.device)
        pos_emb = pos_emb.permute(1, 0, 2)  # "t2 b c -> b t2 c"
        xt = self.norm_in_t(xt)
        xt = xt + self.weight_pos_embed * pos_emb

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)
                xt = self.layers_t[idx](xt)
            else:
                old_x = x
                x = self.layers[idx](x, xt)
                xt = self.layers_t[idx](xt, old_x)

        x = x.reshape(B, T1, Fr, C).permute(0, 3, 2, 1)  # "b (t1 fr) c -> b c fr t1"
        xt = xt.permute(0, 2, 1)  # "b t2 c -> b c t2"
        return x, xt

    def _get_pos_embedding(self, T, B, C, device):
        if self.emb == "sin":
            shift = random.randrange(self.sin_random_shift + 1)
            pos_emb = create_sin_embedding(
                T, C, shift=shift, device=device, max_period=self.max_period
            )
        elif self.emb == "cape":
            pos_emb = create_sin_embedding_cape(
                T,
                C,
                B,
                device=device,
                max_period=self.max_period,
                mean_normalize=self.cape_mean_normalize,
                augment=False,
            )

        elif self.emb == "scaled":
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embeddings(pos)[:, None]

        return pos_emb
