# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import inspect
import warnings
from pathlib import Path

import torch

# Known deprecated parameters that are present in older model checkpoints
# but are no longer used in the current model classes. These are silently ignored.
_DEPRECATED_PARAMS = frozenset({
    # Legacy Wiener filtering parameters
    "wiener_iters",
    "end_iters",
    "wiener_residual",
    # Removed sparse attention parameters (xformers APIs deprecated in 0.0.34)
    "t_sparse_self_attn",
    "t_sparse_cross_attn",
    "t_mask_type",
    "t_mask_random_seed",
    "t_sparse_attn_window",
    "t_global_window",
    "t_sparsity",
    "t_auto_sparsity",
})


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
                if key not in _DEPRECATED_PARAMS:
                    warnings.warn("Dropping inexistant parameter " + key)
                del kwargs[key]
        model = klass(*args, **kwargs)

    state = package["state"]

    set_state(model, state)
    return model


def set_state(model, state):
    """Set the state on a given model."""
    model.load_state_dict(state)
    return state


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
