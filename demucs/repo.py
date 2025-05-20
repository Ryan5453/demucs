"""
Repository system for managing pretrained models.
"""

import json
import typing as tp
from hashlib import sha256
from pathlib import Path

import torch

from .apply import BagOfModels, Model
from .states import load_model

AnyModel = tp.Union[Model, BagOfModels]


class ModelLoadingError(RuntimeError):
    pass


def check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, "rb") as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[: len(checksum)]
    if actual_checksum != checksum:
        raise ModelLoadingError(
            f"Invalid checksum for file {path}, "
            f"expected {checksum} but got {actual_checksum}"
        )


class ModelRepository:
    """Unified repository system for accessing models and collections."""

    def __init__(self, metadata_path: Path, local_repo_path: tp.Optional[Path] = None):
        """
        Initialize the repository.

        Args:
            metadata_path: Path to metadata.json containing model information
            local_repo_path: Path to local model repository (if None, use remote)
        """
        self.metadata_path = metadata_path
        self.local_repo_path = local_repo_path

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self._models = self.metadata["models"]
        self._collections = self.metadata["collections"]

        # Scan local repository if provided
        self._local_models = {}
        self._local_checksums = {}
        if local_repo_path is not None:
            self._scan_local_repo()

    def _scan_local_repo(self):
        """Scan local repository for model files."""
        for file in self.local_repo_path.iterdir():
            if file.suffix == ".th":
                if "-" in file.stem:
                    sig, checksum = file.stem.split("-")
                    self._local_checksums[sig] = checksum
                else:
                    sig = file.stem
                if sig in self._local_models:
                    raise ModelLoadingError(
                        f"Duplicate pre-trained model exist for signature {sig}. "
                        "Please delete all but one."
                    )
                self._local_models[sig] = file

    def has_model(self, name: str) -> bool:
        """Check if a model or collection exists."""
        # Check local models first
        if self.local_repo_path is not None and name in self._local_models:
            return True

        # Then check remote models
        if name in self._models:
            return True

        # Finally check collections
        return name in self._collections

    def get_model(self, name: str) -> AnyModel:
        """
        Get a model or collection by name.

        Args:
            name: Model name, signature, or collection name

        Returns:
            The requested model or model collection
        """
        # Try loading from local repository first
        if self.local_repo_path is not None and name in self._local_models:
            file = self._local_models[name]
            if name in self._local_checksums:
                check_checksum(file, self._local_checksums[name])
            return load_model(file)

        # Try loading as a remote model
        if name in self._models:
            model_info = self._models[name]
            url = model_info["url"]
            pkg = torch.hub.load_state_dict_from_url(
                url, map_location="cpu", check_hash=True
            )
            return load_model(pkg)

        # Try loading as a collection
        if name in self._collections:
            collection = self._collections[name]
            signatures = collection["models"]
            models = [self.get_model(sig) for sig in signatures]
            weights = collection.get("weights")
            segment = collection.get("segment")
            return BagOfModels(models, weights, segment)

        # If we got here, the model doesn't exist
        raise ModelLoadingError(
            f"Could not find a model or collection with name {name}."
        )

    def list_models(self) -> tp.Dict[str, tp.Dict]:
        """
        List all available models and collections.

        Returns:
            Dictionary mapping model/collection names to their metadata
        """
        result = {}

        # Add remote models
        for sig, info in self._models.items():
            result[sig] = {"type": "remote_model", "info": info}

        # Add local models
        if self.local_repo_path is not None:
            for sig, path in self._local_models.items():
                result[sig] = {"type": "local_model", "path": str(path)}

        # Add collections
        for name, info in self._collections.items():
            result[name] = {"type": "collection", "info": info}

        return result
