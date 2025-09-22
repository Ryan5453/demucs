# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor

from .exceptions import InvalidClipModeError


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

    # If sample rates are the same, no resampling needed
    if from_samplerate == to_samplerate:
        return wav

    # Calculate the resampling ratio
    ratio = to_samplerate / from_samplerate

    # Resample using PyTorch's interpolate function
    # wav shape: [..., channels, samples]
    original_shape = wav.shape
    wav_reshaped = wav.view(-1, 1, original_shape[-1])  # [batch*channels, 1, samples]

    # Use linear interpolation for audio resampling
    resampled = F.interpolate(
        wav_reshaped, scale_factor=ratio, mode="linear", align_corners=False
    )

    # Reshape back to original format
    new_length = resampled.shape[-1]
    return resampled.view(*original_shape[:-1], new_length)


def prevent_clip(audio: Tensor, mode: ClipMode = ClipMode.rescale) -> Tensor:
    """
    Different strategies for avoiding raw clipping.

    :param audio: The audio tensor to prevent clipping from
    :param mode: The mode to use for preventing clipping
    :return: The audio tensor with clipping prevented
    :raises InvalidClipModeError: If the clippingmode is invalid
    """
    if mode == ClipMode.none:
        return audio
    elif mode == ClipMode.rescale:
        return audio / max(1.01 * audio.abs().max(), 1)
    elif mode == ClipMode.clamp:
        return audio.clamp(-0.99, 0.99)
    elif mode == ClipMode.tanh:
        return torch.tanh(audio)
    else:
        raise InvalidClipModeError(f"Invalid mode {mode}")
