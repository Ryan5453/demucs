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
from rich.console import Console

console = Console(stderr=True)


def _check_diffq():
    try:
        import diffq  # noqa
    except ImportError:
        console.print(
            "[bold red]Trying to use DiffQ, but diffq is not installed.\n"
            "On Windows run: python.exe -m pip install diffq \n"
            "On Linux/Mac, run: python3 -m pip install diffq[/bold red]"
        )
        raise ImportError("diffq is not installed")


def get_quantizer(model, args, optimizer=None):
    """Return the quantizer given the XP quantization args."""
    quantizer = None
    if args.diffq:
        _check_diffq()
        from diffq import DiffQuantizer

        quantizer = DiffQuantizer(
            model, min_size=args.min_size, group_size=args.group_size
        )
        if optimizer is not None:
            quantizer.setup_optimizer(optimizer)
    elif args.qat:
        _check_diffq()
        from diffq import UniformQuantizer

        quantizer = UniformQuantizer(model, bits=args.qat, min_size=args.min_size)
    return quantizer


def load_model(path_or_package, strict=False):
    """Load a model from the given serialized model, either given as a dict (already loaded)
    or a path to a file on disk."""
    if isinstance(path_or_package, dict):
        package = path_or_package
    elif isinstance(path_or_package, (str, Path)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = path_or_package
            package = torch.load(path, "cpu")
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
        if quantizer is not None:
            quantizer.restore_quantized_state(model, state["quantized"])
        else:
            _check_diffq()
            from diffq import restore_quantized_state

            restore_quantized_state(model, state)
    else:
        model.load_state_dict(state)
    return state


def capture_init(init):
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__
