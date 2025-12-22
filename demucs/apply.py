# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import (
    Any,
    Callable,
    TypeAlias,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .blocks import center_trim
from .hdemucs import HDemucs
from .htdemucs import HTDemucs

Model: TypeAlias = HDemucs | HTDemucs


class ModelEnsemble(nn.Module):
    def __init__(
        self,
        models: list[Model],
        weights: list[list[float]] | None = None,
        segment: float | None = None,
    ):
        """
        Represents a model ensemble with specific weights.
        You should call `apply_model` rather than calling directly the forward here for
        optimal performance.

        Args:
            models (list[nn.Module]): list of Demucs/HDemucs models.
            weights (list[list[float]]): list of weights. If None, assumed to
                be all ones, otherwise it should be a list of N list (N number of models),
                each containing S floats (S number of sources).
            segment (None or float): overrides the `segment` attribute of each model
                (this is performed inplace, be careful is you reuse the models passed).
        """
        super().__init__()
        assert len(models) > 0
        first = models[0]
        for other in models:
            assert other.sources == first.sources
            assert other.samplerate == first.samplerate
            assert other.audio_channels == first.audio_channels
            if segment is not None:
                if (
                    not isinstance(other, HTDemucs)
                    or segment <= other.max_allowed_segment
                ):
                    other.max_allowed_segment = segment

        self.audio_channels = first.audio_channels
        self.samplerate = first.samplerate
        self.sources = first.sources
        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [[1.0 for _ in first.sources] for _ in models]
        else:
            assert len(weights) == len(models)
            for weight in weights:
                assert len(weight) == len(first.sources)
        self.weights = weights

    @property
    def max_allowed_segment(self) -> float:
        max_allowed_segment = float("inf")
        for model in self.models:
            if isinstance(model, HTDemucs):
                max_allowed_segment = min(
                    max_allowed_segment, float(model.max_allowed_segment)
                )
        return max_allowed_segment

    def forward(self, x):
        raise NotImplementedError("Call `apply_model` on this.")


class TensorChunk:
    def __init__(self, tensor, offset=0, length=None):
        total_length = tensor.shape[-1]
        assert offset >= 0
        assert offset < total_length

        if length is None:
            length = total_length - offset
        else:
            length = min(total_length - offset, length)

        if isinstance(tensor, TensorChunk):
            self.tensor = tensor.tensor
            self.offset = offset + tensor.offset
        else:
            self.tensor = tensor
            self.offset = offset
        self.length = length
        self.device = tensor.device

    @property
    def shape(self):
        shape = list(self.tensor.shape)
        shape[-1] = self.length
        return shape

    def padded(self, target_length):
        delta = target_length - self.length
        total_length = self.tensor.shape[-1]
        assert delta >= 0

        start = self.offset - delta // 2
        end = start + target_length

        correct_start = max(0, start)
        correct_end = min(total_length, end)

        pad_left = correct_start - start
        pad_right = end - correct_end

        out = F.pad(self.tensor[..., correct_start:correct_end], (pad_left, pad_right))
        assert out.shape[-1] == target_length
        return out


def tensor_chunk(tensor_or_chunk):
    if isinstance(tensor_or_chunk, TensorChunk):
        return tensor_or_chunk
    else:
        assert isinstance(tensor_or_chunk, Tensor)
        return TensorChunk(tensor_or_chunk)


def apply_model(
    model: ModelEnsemble | Model,
    mix: Tensor | TensorChunk,
    device=None,
    shifts=0,
    split=False,
    overlap=0.25,
    transition_power=1.0,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    segment=None,
    use_only_stem: str | None = None,
):
    """
    Apply model to a given mixture.

    Args:
        shifts (int): if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec
            and apply the opposite shift to the output. This is repeated `shifts` time and
            all predictions are averaged. This effectively makes the model time equivariant
            and improves SDR by up to 0.2 points.
        split (bool): if True, the input will be broken down in 8 seconds extracts
            and predictions will be performed individually on each and concatenated.
            Useful for model with large memory footprint like Tasnet.
        progress_callback: Optional callback function for progress updates during split processing.
                          Called with (event_type, data) where event_type is one of:
                          - "processing_start": data = {"total_chunks": int}
                          - "chunk_complete": data = {"completed_chunks": int, "total_chunks": int}
                          - "processing_complete": data = {"total_chunks": int}
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
        segment (float or None): override the model segment parameter.
        use_only_stem (str or None): if specified and model is a ModelEnsemble, only use
            the sub-model specialized for this stem (performance optimization).
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    kwargs = {
        "shifts": shifts,
        "split": split,
        "overlap": overlap,
        "transition_power": transition_power,
        "progress_callback": progress_callback,
        "device": device,
        "segment": segment,
        "use_only_stem": use_only_stem,
    }
    out: float | Tensor
    res: float | Tensor
    if isinstance(model, ModelEnsemble):
        # Special treatment for model ensemble.
        # We explicitely apply multiple times `apply_model` so that the random shifts
        # are different for each model.

        # Optimization: If use_only_stem is specified, only run the specialized model
        if use_only_stem:
            # Find which model specializes in this stem
            try:
                stem_index = model.sources.index(use_only_stem)
            except ValueError:
                # Stem doesn't exist, fall through to run all models
                pass
            else:
                # Find the model that specializes in this stem
                model_index = None
                for i, weights in enumerate(model.weights):
                    if (
                        len(weights) > stem_index
                        and abs(weights[stem_index] - 1.0) < 1e-6
                        and all(
                            abs(w) < 1e-6
                            for j, w in enumerate(weights)
                            if j != stem_index
                        )
                    ):
                        model_index = i
                        break

                if model_index is not None:
                    # Run only the specialized model
                    sub_model = model.models[model_index]
                    original_model_device = next(iter(sub_model.parameters())).device
                    sub_model.to(device)

                    # Remove use_only_stem for the recursive call
                    sub_kwargs = dict(kwargs)
                    sub_kwargs.pop("use_only_stem")
                    result = apply_model(sub_model, mix, **sub_kwargs)

                    sub_model.to(original_model_device)
                    return result

        # Default behavior: run all models in the bag
        estimates: float | Tensor = 0.0
        totals = [0.0] * len(model.sources)
        for sub_model, model_weights in zip(model.models, model.weights):
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)

            # Remove use_only_stem for recursive calls
            sub_kwargs = dict(kwargs)
            sub_kwargs.pop("use_only_stem")
            res = apply_model(sub_model, mix, **sub_kwargs)
            out = res

            # Only move back to original device if it's different from the target device
            # This allows "pinning" models to the GPU for better performance
            if original_model_device != device:
                sub_model.to(original_model_device)

            for k, inst_weight in enumerate(model_weights):
                out[:, k, :, :] *= inst_weight
                totals[k] += inst_weight
            estimates += out
            del out

        assert isinstance(estimates, Tensor)
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] /= totals[k]
        return estimates

    model.to(device)
    model.eval()
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    if shifts:
        kwargs["shifts"] = 0
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.0
        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            res = apply_model(model, shifted, **kwargs)
            shifted_out = res
            out += shifted_out[..., max_shift - offset :]
        out /= shifts
        assert isinstance(out, Tensor)
        return out
    elif split:
        kwargs["split"] = False
        out = torch.zeros(
            batch, len(model.sources), channels, length, device=mix.device
        )
        sum_weight = torch.zeros(length, device=mix.device)
        if segment is None:
            segment = model.max_allowed_segment
        assert segment is not None and segment > 0.0
        segment_length: int = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        # We start from a triangle shaped weight, with maximal weight in the middle
        # of the segment. Then we normalize and take to the power `transition_power`.
        # Large values of transition power will lead to sharper transitions.
        weight = torch.cat(
            [
                torch.arange(1, segment_length // 2 + 1, device=device),
                torch.arange(
                    segment_length - segment_length // 2, 0, -1, device=device
                ),
            ]
        )
        assert len(weight) == segment_length
        # If the overlap < 50%, this will translate to linear transition when
        # transition_power is 1.
        weight = (weight / weight.max()) ** transition_power
        weight_on_device = weight.to(mix.device)

        # Process chunks sequentially (PyTorch handles internal threading)
        total_chunks = len(offsets)

        # Notify callback about processing start
        if progress_callback:
            progress_callback("processing_start", {"total_chunks": total_chunks})

        completed_chunks = 0
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment_length)
            chunk_out = apply_model(
                model,
                chunk,
                **kwargs,
            )
            chunk_length = chunk_out.shape[-1]
            out[..., offset : offset + segment_length] += weight_on_device[
                :chunk_length
            ] * chunk_out.to(mix.device)
            sum_weight[offset : offset + segment_length] += weight_on_device[
                :chunk_length
            ]

            completed_chunks += 1
            if progress_callback:
                progress_callback(
                    "chunk_complete",
                    {
                        "completed_chunks": completed_chunks,
                        "total_chunks": total_chunks,
                    },
                )

        # Notify callback about processing completion
        if progress_callback:
            progress_callback("processing_complete", {"total_chunks": total_chunks})
        assert sum_weight.min() > 0
        out /= sum_weight
        assert isinstance(out, Tensor)
        return out
    else:
        valid_length: int
        if isinstance(model, HTDemucs) and segment is not None:
            valid_length = int(segment * model.samplerate)
        elif hasattr(model, "valid_length"):
            valid_length = model.valid_length(length)  # type: ignore
        else:
            valid_length = length
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(valid_length).to(device)
        with torch.no_grad():
            out = model(padded_mix)
        assert isinstance(out, Tensor)
        return center_trim(out, length)
