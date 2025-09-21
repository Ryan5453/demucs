# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import inspect
import sys
import warnings
from pathlib import Path

import torch


# Quantization support has been removed from this version of Demucs
# These functions are kept as stubs to maintain API compatibility

def _check_diffq():
    """Legacy function - quantization support has been removed."""
    raise ImportError(
        "Quantization support has been removed from this version of Demucs. "
        "Please use non-quantized models instead."
    )


def get_quantizer(model, args, optimizer=None):
    """Legacy function - quantization support has been removed."""
    return None


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, "cpu", weights_only=False)
    else:
        raise ValueError(f"Invalid type for {path_or_package}.")

    klass = package["klass"]
    args = package["args"]
    kwargs = package["kwargs"]

    if strict:
        model = klass(*args, **kwargs)
    else:
        sig = inspect.signature(klass)
        for key in list(kwargs):
            if key not in sig.parameters:
                warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def set_state(model, state, quantizer=None):
    """Set the state on a given model."""
    if state.get("__quantized"):
        raise ImportError(
            "Quantized model detected but quantization support has been removed. "
            "Please use non-quantized models instead."
        )
    else:
        model.load_state_dict(state)
    return state


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
