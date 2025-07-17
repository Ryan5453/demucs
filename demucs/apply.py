# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Hashable,
    List,
    Optional,
    TypeAlias,
)

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch import Tensor
from torch.nn import functional as F

from .demucs import Demucs
from .hdemucs import HDemucs
from .htdemucs import HTDemucs
from .utils import center_trim, DummyPoolExecutor

# Type alias for all model types
Model: TypeAlias = Demucs | HDemucs | HTDemucs

console = Console()

# Global weight tensor cache - this optimization is actually useful
_weight_cache = {}
_weight_cache_lock = threading.Lock()


def _get_weight_tensor(
    segment_length: int, transition_power: float, device: torch.device
):
    """Get or create a cached weight tensor for segmented processing."""
    key = (segment_length, transition_power, device.type)

    with _weight_cache_lock:
        if key not in _weight_cache:
            # Create triangle shaped weight
            weight = torch.cat(
                [
                    torch.arange(1, segment_length // 2 + 1, device=device),
                    torch.arange(
                        segment_length - segment_length // 2, 0, -1, device=device
                    ),
                ]
            )
            # Normalize and apply transition power
            weight = (weight / weight.max()) ** transition_power
            _weight_cache[key] = weight.clone()

    # Return a copy on the correct device
    cached_weight = _weight_cache[key]
    if cached_weight.device != device:
        return cached_weight.to(device)
    return cached_weight


class BagOfModels(nn.Module):
    def __init__(
        self,
        models: List[Model],
        weights: Optional[List[List[float]]] = None,
        segment: Optional[float] = None,
    ):
        """
        Represents a bag of models with specific weights.
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
                if not isinstance(other, HTDemucs) or segment <= other.segment:
                    other.segment = segment

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
                max_allowed_segment = min(max_allowed_segment, float(model.segment))
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


def _replace_dict(_dict: Optional[dict], *subs: tuple[Hashable, Any]) -> dict:
    if _dict is None:
        _dict = {}
    else:
        _dict = copy.copy(_dict)
    for key, value in subs:
        _dict[key] = value
    return _dict


def apply_model(
    model: BagOfModels | Model,
    mix: Tensor | TensorChunk,
    device=None,
    shifts=0,
    split=False,
    overlap=0.25,
    transition_power=1.0,
    progress=False,
    segment=None,
    num_workers=0,
    callback=None,
    callback_arg=None,
    compile_model=False,
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
        progress (bool): if True, show a progress bar (requires split=True)
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
        num_workers (int): if non zero, device is 'cpu', how many threads to
            use in parallel.
        segment (float or None): override the model segment parameter.
        compile_model (bool): if True, use torch.compile for optimization.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    if callback_arg is None:
        callback_arg = {}
    callback_arg["model_idx_in_bag"] = 0
    callback_arg["shift_idx"] = 0
    callback_arg["segment_offset"] = 0

    out: float | Tensor
    res: float | Tensor
    if isinstance(model, BagOfModels):
        # Special treatment for bag of model.
        # We explicitely apply multiple times `apply_model` so that the random shifts
        # are different for each model.
        estimates: float | Tensor = 0.0
        totals = [0.0] * len(model.sources)
        callback_arg["models"] = len(model.models)
        for sub_model, model_weights in zip(model.models, model.weights):
            # Create callback for this specific model
            model_callback = (
                lambda d, i=callback_arg["model_idx_in_bag"]: callback(
                    _replace_dict(d, ("model_idx_in_bag", i))
                )
                if callback
                else None
            )
            original_model_device = next(iter(sub_model.parameters())).device
            sub_model.to(device)

            # Simple recursive call - just pass what we need
            res = apply_model(
                sub_model,
                mix,
                device=device,
                shifts=shifts,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
                progress=progress,
                segment=segment,
                num_workers=num_workers,
                callback=model_callback,
                callback_arg=callback_arg,
                compile_model=compile_model,
            )
            out = res
            sub_model.to(original_model_device)
            for k, inst_weight in enumerate(model_weights):
                out[:, k, :, :] = out[:, k, :, :] * inst_weight  # Non-inplace
                totals[k] += inst_weight
            estimates += out
            del out
            callback_arg["model_idx_in_bag"] += 1

        assert isinstance(estimates, Tensor)
        for k in range(estimates.shape[1]):
            estimates[:, k, :, :] = estimates[:, k, :, :] / totals[k]  # Non-inplace
        return estimates

    if "models" not in callback_arg:
        callback_arg["models"] = 1
    # Only move model to device if necessary
    current_device = next(iter(model.parameters())).device
    if current_device != device:
        model.to(device)
    model.eval()

    # Optionally compile the model for better performance
    if compile_model and hasattr(torch, "compile"):
        try:
            # Check if model is already compiled
            if not hasattr(model, "_is_compiled"):
                model = torch.compile(model, mode="reduce-overhead")
                model._is_compiled = True
        except Exception:
            # Fall back to uncompiled model if compilation fails
            pass
    assert transition_power >= 1, "transition_power < 1 leads to weird behavior."
    batch, channels, length = mix.shape
    if shifts:
        max_shift = int(0.5 * model.samplerate)
        mix = tensor_chunk(mix)
        assert isinstance(mix, TensorChunk)
        padded_mix = mix.padded(length + 2 * max_shift)
        out = 0.0
        for shift_idx in range(shifts):
            offset = random.randint(0, max_shift)
            shifted = TensorChunk(padded_mix, offset, length + max_shift - offset)
            # Create callback for this specific shift
            shift_callback = (
                lambda d, i=shift_idx: callback(_replace_dict(d, ("shift_idx", i)))
                if callback
                else None
            )
            # Simple recursive call - no shifts this time
            res = apply_model(
                model,
                shifted,
                device=device,
                shifts=0,
                split=split,
                overlap=overlap,
                transition_power=transition_power,
                progress=progress,
                segment=segment,
                num_workers=num_workers,
                callback=shift_callback,
                callback_arg=callback_arg,
                compile_model=compile_model,
            )
            shifted_out = res
            out += shifted_out[..., max_shift - offset :]
        out /= shifts
        assert isinstance(out, Tensor)
        return out
    elif split:
        out = torch.zeros(
            batch, len(model.sources), channels, length, device=mix.device
        )
        sum_weight = torch.zeros(length, device=mix.device)
        if segment is None:
            segment = model.segment
        assert segment is not None and segment > 0.0
        segment_length: int = int(model.samplerate * segment)
        stride = int((1 - overlap) * segment_length)
        offsets = range(0, length, stride)
        # Get cached weight tensor for efficiency
        weight = _get_weight_tensor(segment_length, transition_power, device)
        assert len(weight) == segment_length
        # Create thread pool only when needed
        if num_workers > 0 and device.type == "cpu":
            pool = ThreadPoolExecutor(max_workers=num_workers)
        else:
            pool = DummyPoolExecutor()

        futures = []
        for offset in offsets:
            chunk = TensorChunk(mix, offset, segment_length)
            # Create callback for this specific chunk
            chunk_callback = (
                lambda d, i=offset: callback(_replace_dict(d, ("segment_offset", i)))
                if callback
                else None
            )
            future = pool.submit(
                apply_model,
                model,
                chunk,
                device=device,
                shifts=shifts,
                split=False,
                overlap=overlap,
                transition_power=transition_power,
                progress=progress,
                segment=segment,
                num_workers=num_workers,
                callback=chunk_callback,
                callback_arg=callback_arg,
                compile_model=compile_model,
            )
            futures.append((future, offset))
            offset += segment_length
        if progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green"),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
                refresh_per_second=10,
            ) as progress_bar:
                task = progress_bar.add_task("Processing audio", total=len(futures))
                for future, offset in futures:
                    try:
                        chunk_out = future.result()  # type: Tensor
                        chunk_length = chunk_out.shape[-1]
                        out[..., offset : offset + segment_length] += (
                            weight[:chunk_length] * chunk_out
                        ).to(mix.device)
                        sum_weight[offset : offset + segment_length] += weight[
                            :chunk_length
                        ].to(mix.device)
                        progress_bar.update(task, advance=1)
                    except Exception:
                        pool.shutdown(wait=True, cancel_futures=True)
                        raise
        else:
            for future, offset in futures:
                try:
                    chunk_out = future.result()  # type: Tensor
                    chunk_length = chunk_out.shape[-1]
                    out[..., offset : offset + segment_length] += (
                        weight[:chunk_length] * chunk_out
                    ).to(mix.device)
                    sum_weight[offset : offset + segment_length] += weight[
                        :chunk_length
                    ].to(mix.device)
                except Exception:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
        assert sum_weight.min() > 0
        out /= sum_weight

        # Clean up the thread pool
        pool.shutdown(wait=True)

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
        padded_mix = mix.padded(valid_length)
        # Only transfer to device if necessary
        if padded_mix.device != device:
            padded_mix = padded_mix.to(device)

        # Simple callback execution - no locks needed
        if callback is not None:
            callback(_replace_dict(callback_arg, ("state", "start")))  # type: ignore

        # Use mixed precision for compatible devices
        autocast_enabled = device.type in ["cuda", "xpu"]
        autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

        with torch.inference_mode():
            if autocast_enabled:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    out = model(padded_mix)
            else:
                out = model(padded_mix)

        if callback is not None:
            callback(_replace_dict(callback_arg, ("state", "end")))  # type: ignore
        assert isinstance(out, Tensor)
        return center_trim(out, length)
