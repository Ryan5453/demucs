# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from pathlib import Path
from typing import Optional, Callable, Any, Dict

from .repo import ModelRepository
from .states import _check_diffq

logger = logging.getLogger(__name__)
METADATA_PATH = Path(__file__).parent / "metadata.json"

SOURCES = ["drums", "bass", "other", "vocals"]


def get_model(
    name: str,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
):
    """
    Load a model by name from the model repository.

    Args:
        name: Model name from the model repository
        progress_callback: Optional callback function for progress updates.
                          Called with (event_type, data) where event_type is one of:
                          - "download_start": data = {"model_name": str, "total_layers": int}
                          - "layer_start": data = {"model_name": str, "layer_index": int, "total_layers": int}
                          - "layer_progress": data = {"model_name": str, "layer_index": int, "total_layers": int, "progress_percent": float}
                          - "layer_complete": data = {"model_name": str, "layer_index": int, "total_layers": int}
                          - "download_complete": data = {"model_name": str, "total_layers": int}
    """
    model_repo = ModelRepository()

    try:
        model = model_repo.get_model(name, progress_callback=progress_callback)
    except ImportError as exc:
        if "diffq" in exc.args[0]:
            _check_diffq()
        raise

    model.eval()
    return model
