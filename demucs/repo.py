"""
Represents a model repository, including pre-trained models and bags of models.
A repo can either be the GitHub repository or a local repository with your own models.
"""

import typing as tp
import json
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


class ModelOnlyRepo:
    """Base class for all model only repos."""

    def has_model(self, sig: str) -> bool:
        raise NotImplementedError()

    def get_model(self, sig: str) -> Model:
        raise NotImplementedError()

    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        raise NotImplementedError()


class GitHubRepo(ModelOnlyRepo):
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self._models = self.metadata["models"]

    def has_model(self, sig: str) -> bool:
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        try:
            model_info = self._models[sig]
            url = model_info["url"]
        except KeyError:
            raise ModelLoadingError(
                f"Could not find a pre-trained model with signature {sig}."
            )
        pkg = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=True
        )  # type: ignore
        return load_model(pkg)

    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return {sig: info["url"] for sig, info in self._models.items()}


class LocalRepo(ModelOnlyRepo):
    def __init__(self, root: Path):
        self.root = root
        self.scan()

    def scan(self):
        self._models = {}
        self._checksums = {}
        for file in self.root.iterdir():
            if file.suffix == ".th":
                if "-" in file.stem:
                    xp_sig, checksum = file.stem.split("-")
                    self._checksums[xp_sig] = checksum
                else:
                    xp_sig = file.stem
                if xp_sig in self._models:
                    raise ModelLoadingError(
                        f"Duplicate pre-trained model exist for signature {xp_sig}. "
                        "Please delete all but one."
                    )
                self._models[xp_sig] = file

    def has_model(self, sig: str) -> bool:
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        try:
            file = self._models[sig]
        except KeyError:
            raise ModelLoadingError(
                f"Could not find pre-trained model with signature {sig}."
            )
        if sig in self._checksums:
            check_checksum(file, self._checksums[sig])
        return load_model(file)

    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return self._models


class CollectionRepo:
    """Handles collections of models (previously represented by YAML files)"""

    def __init__(self, metadata_path: Path, model_repo: ModelOnlyRepo):
        self.metadata_path = metadata_path
        self.model_repo = model_repo
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self._collections = self.metadata["collections"]

    def has_model(self, name: str) -> bool:
        return name in self._collections

    def get_model(self, name: str) -> BagOfModels:
        try:
            collection = self._collections[name]
        except KeyError:
            raise ModelLoadingError(
                f"{name} is neither a single pre-trained model or a collection of models."
            )
        signatures = collection["models"]
        models = [self.model_repo.get_model(sig) for sig in signatures]
        weights = collection.get("weights")
        segment = collection.get("segment")
        return BagOfModels(models, weights, segment)

    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return self._collections


class RemoteRepo(ModelOnlyRepo):
    """Repository for models stored remotely, accessible through URLs."""
    
    def __init__(self, files_dict: tp.Dict[str, str]):
        self._models = files_dict
        
    def has_model(self, sig: str) -> bool:
        return sig in self._models
        
    def get_model(self, sig: str) -> Model:
        try:
            url = self._models[sig]
        except KeyError:
            raise ModelLoadingError(
                f"Could not find a pre-trained model with signature {sig}."
            )
        pkg = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", check_hash=True
        )  # type: ignore
        return load_model(pkg)
        
    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return self._models


class BagOnlyRepo:
    """Repository that handles only bag of models."""
    
    def __init__(self, root: tp.Union[Path, str], model_repo: ModelOnlyRepo):
        self.root = root
        self.model_repo = model_repo
        self._bags = {}  # This would typically be populated by scanning files
        
    def has_model(self, name: str) -> bool:
        return name in self._bags
        
    def get_model(self, name: str) -> BagOfModels:
        try:
            bag_info = self._bags[name]
            sigs = bag_info["models"]
        except KeyError:
            raise ModelLoadingError(f"Could not find a bag of models named {name}.")
            
        models = [self.model_repo.get_model(sig) for sig in sigs]
        weights = bag_info.get("weights")
        segment = bag_info.get("segment")
        return BagOfModels(models, weights, segment)
        
    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        return self._bags


class AnyModelRepo:
    def __init__(self, model_repo: ModelOnlyRepo, collection_repo: CollectionRepo):
        self.model_repo = model_repo
        self.collection_repo = collection_repo

    def has_model(self, name_or_sig: str) -> bool:
        return self.model_repo.has_model(name_or_sig) or self.collection_repo.has_model(
            name_or_sig
        )

    def get_model(self, name_or_sig: str) -> AnyModel:
        if self.model_repo.has_model(name_or_sig):
            return self.model_repo.get_model(name_or_sig)
        else:
            return self.collection_repo.get_model(name_or_sig)

    def list_model(self) -> tp.Dict[str, tp.Union[str, Path]]:
        models = self.model_repo.list_model()
        for key, value in self.collection_repo.list_model().items():
            models[key] = value
        return models