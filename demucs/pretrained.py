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

from .repo import (
    AnyModelRepo,
    CollectionRepo,
    GitHubRepo,
    LocalRepo,
    ModelLoadingError,
)  # noqa
from .states import _check_diffq

logger = logging.getLogger(__name__)
METADATA_PATH = Path(__file__).parent / "metadata.json"
REMOTE_ROOT = Path("https://github.com/Ryan5453/demucs/releases/download/v5.0.0-models")

SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_MODEL = "htdemucs"

# Export DEFAULT_MODEL to be used in separate.py
__all__ = [
    "ModelLoadingError",
    "get_model",
    "get_model_from_args",
    "SOURCES",
    "DEFAULT_MODEL",
]


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-s", "--sig", help="Locally trained XP signature.")
    group.add_argument(
        "-n",
        "--name",
        default="htdemucs",
        help="Pretrained model name or signature. Default is htdemucs.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        help="Folder containing all pre-trained models for use with -n.",
    )


def get_model(name: str, repo: tp.Optional[Path] = None):
    """`name` must be a collection of models name or a pretrained signature
    from the GitHub model repo or the specified local repo if `repo` is not None.
    """
    model_repo: tp.Union[GitHubRepo, LocalRepo]
    if repo is None:
        model_repo = GitHubRepo(METADATA_PATH)
        collection_repo = CollectionRepo(METADATA_PATH, model_repo)
    else:
        if not repo.is_dir():
            fatal(f"{repo} must exist and be a directory.")
        model_repo = LocalRepo(repo)
        collection_repo = CollectionRepo(METADATA_PATH, model_repo)
    any_repo = AnyModelRepo(model_repo, collection_repo)
    try:
        model = any_repo.get_model(name)
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
