# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import TypeAlias
from enum import Enum

import julius
import torch
import torchaudio
from torch import Tensor

# Type alias for path-like objects
PathLike: TypeAlias = str | Path


class ClipMode(str, Enum):
    rescale = "rescale"
    clamp = "clamp"
    tanh = "tanh"
    none = "none"


def convert_audio_channels(wav, channels=2):
    """Convert audio to the given number of channels."""
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, but the stream have multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file have
        # one single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file have
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError(
            "The audio file has less channels than requested but is not mono."
        )
    return wav


def convert_audio(wav, from_samplerate, to_samplerate, channels) -> Tensor:
    """Convert audio from a given samplerate to a target one and target number of channels."""
    wav = convert_audio_channels(wav, channels)
    return julius.resample_frac(wav, from_samplerate, to_samplerate)


def prevent_clip(wav, mode: ClipMode = ClipMode.rescale):
    """
    Different strategies for avoiding raw clipping.
    """
    if mode is None or mode == ClipMode.none:
        return wav
    assert wav.dtype.is_floating_point, "too late for clipping"
    if mode == ClipMode.rescale:
        wav = wav / max(1.01 * wav.abs().max(), 1)
    elif mode == ClipMode.clamp:
        wav = wav.clamp(-0.99, 0.99)
    elif mode == ClipMode.tanh:
        wav = torch.tanh(wav)
    else:
        raise InvalidClipModeError(f"Invalid mode {mode}")
    return wav


def save_audio(
    wav: Tensor,
    path: PathLike,
    samplerate: int,
    clip: ClipMode = ClipMode.rescale,
):
    """Save audio file as 32-bit float WAV (native model output format),
    automatically preventing clipping if necessary based on the given `clip` strategy.
    """
    path = Path(path)

    # Ensure tensor is on CPU before any operations
    if wav.device.type != "cpu":
        wav = wav.cpu()

    wav = prevent_clip(wav, mode=clip)

    # Always save as 32-bit float (native model format)
    torchaudio.save(
        path,
        wav,
        sample_rate=samplerate,
        encoding="PCM_F",
        bits_per_sample=32,
    )
