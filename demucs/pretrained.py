# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models from GitHub releases."""

import logging
import typing as tp
from pathlib import Path

from dora.log import bold, fatal

from .repo import ModelRepository
from .states import _check_diffq

logger = logging.getLogger(__name__)
METADATA_PATH = Path(__file__).parent / "metadata.json"

SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_MODEL = "htdemucs"


def get_model(name: str, repo: tp.Optional[Path] = None):
    """
    `name` must be a model name, signature or collection name from the model repository.
    If `repo` is provided, will look for models in that local directory first.
    """
    model_repo = ModelRepository(METADATA_PATH, repo)

    try:
        model = model_repo.get_model(name)
    except ImportError as exc:
        if "diffq" in exc.args[0]:
            _check_diffq()
        raise

    model.eval()
    return model


def get_model_from_args(args):
    """
    Load local model package or pre-trained model.
    """
    if args.name is None:
        args.name = DEFAULT_MODEL
        print(
            bold("Important: the default model was recently changed to `htdemucs`"),
            "the latest Hybrid Transformer Demucs model. In some cases, this model can "
            "actually perform worse than previous models. To get back the old default model "
            "use `-n mdx_extra_q`.",
        )
    return get_model(name=args.name, repo=args.repo)


def _check_diffq():
    try:
        import diffq  # noqa
    except ImportError:
        fatal(
            "You need to install diffq to use this model. "
            "Please run `pip install diffq`."
        )
