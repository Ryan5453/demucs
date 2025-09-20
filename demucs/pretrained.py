# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, TaskID

from .repo import ModelRepository
from .states import _check_diffq

logger = logging.getLogger(__name__)
METADATA_PATH = Path(__file__).parent / "metadata.json"
console = Console()

SOURCES = ["drums", "bass", "other", "vocals"]


def get_model(
    name: str,
    progress_bar: Optional[Progress] = None,
    task_id: Optional[TaskID] = None,
):
    """
    Load a model by name from the model repository.

    Args:
        name: Model name from the model repository
        progress_bar: Optional Progress instance for download progress
        task_id: Optional TaskID for the progress bar
    """
    model_repo = ModelRepository()

    try:
        model = model_repo.get_model(name, progress_bar=progress_bar, task_id=task_id)
    except ImportError as exc:
        if "diffq" in exc.args[0]:
            _check_diffq()
        raise

    model.eval()
    return model


