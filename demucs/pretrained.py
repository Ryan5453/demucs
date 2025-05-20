# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Loading pretrained models from GitHub releases."""

import logging
import typing as tp
import urllib.request
from pathlib import Path

from dora.log import bold, fatal

from .hdemucs import HDemucs
from .repo import (
    AnyModelRepo,
    CollectionRepo,
    LocalRepo,
    GitHubRepo,
    ModelLoadingError,
)  # noqa
from .states import _check_diffq

logger = logging.getLogger(__name__)
METADATA_PATH = Path(__file__).parent / "metadata.json"
REMOTE_ROOT = Path("https://github.com/Ryan5453/demucs/releases/download/v5.0.0-models")

SOURCES = ["drums", "bass", "other", "vocals"]
DEFAULT_MODEL = "htdemucs"

def _parse_remote_files(path_or_url):
    """Parse the files.txt from remote repository and returns a dict mapping
    model names to URLs for download."""
    if isinstance(path_or_url, str) and (path_or_url.startswith('http://') or path_or_url.startswith('https://')):
        with urllib.request.urlopen(path_or_url) as response:
            lines = response.read().decode('utf-8').split('\n')
    else:
        with open(path_or_url, 'r') as f:
            lines = f.readlines()
    
    files = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        file_name, url = line.split()
        files[file_name] = url
    return files


def demucs_unittest():
    model = HDemucs(channels=4, sources=SOURCES)
    return model


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
    if name == "demucs_unittest":
        return demucs_unittest()
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