"""
Repository system for managing pretrained models.
"""

import json
import os
import shutil
import tempfile
import typing as tp
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import httpx
import torch
from rich.progress import Progress, TaskID

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


def format_file_size(size_bytes):
    """Format file size in a human-readable way"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


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

    def get_cache_info(self) -> tp.Dict[str, tp.Dict]:
        """
        Get information about cached models.

        Returns:
            Dictionary with information about cached models
        """
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        cached_models = {}

        # Check which remote models are downloaded
        for sig, info in self._models.items():
            if "url" in info:
                url = info["url"]
                filename = os.path.basename(url)
                model_path = cache_dir / filename

                if model_path.exists():
                    size_bytes = model_path.stat().st_size
                    cached_models[sig] = {
                        "path": str(model_path),
                        "size_bytes": size_bytes,
                        "size_mb": size_bytes / (1024 * 1024),
                    }

        # Handle collections
        for name, info in self._collections.items():
            if "models" in info:
                component_models = info["models"]
                all_cached = True
                total_size = 0
                components = {}

                for component in component_models:
                    if component in cached_models:
                        components[component] = cached_models[component]
                        total_size += cached_models[component]["size_bytes"]
                    else:
                        all_cached = False

                if all_cached:
                    cached_models[name] = {
                        "type": "collection",
                        "components": components,
                        "size_bytes": total_size,
                        "size_mb": total_size / (1024 * 1024),
                    }

        return cached_models

    def _download_and_load_model(
        self,
        url: str,
        cache_path: Path,
        expected_hash: tp.Optional[str] = None,
        progress_bar: tp.Optional[Progress] = None,
        task_id: tp.Optional[TaskID] = None,
    ) -> AnyModel:
        """Download and load a model from a URL."""
        # Download the file to memory first
        try:
            with httpx.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                # Download to memory first
                buffer = BytesIO()
                downloaded_size = 0

                # Update progress bar for download phase (0-90%)
                if progress_bar and task_id:
                    progress_bar.update(task_id, total=100, completed=0)
                    if total_size:
                        progress_bar.update(
                            task_id,
                            description=f"Downloading {os.path.basename(url)} ({format_file_size(total_size)})",
                        )
                    else:
                        progress_bar.update(
                            task_id, description=f"Downloading {os.path.basename(url)}"
                        )

                for chunk in response.iter_bytes(chunk_size=8192):
                    buffer.write(chunk)
                    downloaded_size += len(chunk)

                    # Update progress to show download progress (0-90%)
                    if progress_bar and task_id and total_size:
                        progress = (
                            downloaded_size / total_size
                        ) * 90  # Use 90% for download
                        progress_bar.update(task_id, completed=progress)

                buffer.seek(0)

                # Try to load as a PyTorch model directly from memory
                try:
                    if progress_bar and task_id:
                        progress_bar.update(
                            task_id, description="Loading model...", completed=95
                        )

                    # Save to a temporary file first
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".th"
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        tmp_file.write(buffer.getvalue())

                    # Load the model
                    model_data = torch.load(tmp_path, map_location="cpu")

                    if progress_bar and task_id:
                        progress_bar.update(
                            task_id,
                            description="Verifying and caching...",
                            completed=98,
                        )

                    # If successful, save to cache
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(tmp_path), str(cache_path))

                    # Verify checksum if available
                    if expected_hash:
                        try:
                            check_checksum(cache_path, expected_hash)
                        except ModelLoadingError:
                            cache_path.unlink()
                            raise ModelLoadingError(
                                "Downloaded file has invalid checksum"
                            )

                    if progress_bar and task_id:
                        progress_bar.update(task_id, completed=100)

                    return load_model(model_data)

                except Exception as e:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()
                    raise ModelLoadingError(f"Failed to load model: {str(e)}")

        except httpx.HTTPError as e:
            raise ModelLoadingError(f"Failed to download {url}: {str(e)}")

    def get_model(
        self,
        name: str,
        progress_bar: tp.Optional[Progress] = None,
        task_id: tp.Optional[TaskID] = None,
        collection_name: tp.Optional[str] = None,
    ) -> AnyModel:
        """
        Get a model or collection by name.

        Args:
            name: Model name, signature, or collection name
            progress_bar: Optional rich.progress.Progress instance for download progress
            task_id: Optional TaskID for the progress bar
            collection_name: Optional name of parent collection (for nested progress)

        Returns:
            The requested model or model collection
        """
        # Try loading from local repository first
        if self.local_repo_path is not None and name in self._local_models:
            file = self._local_models[name]
            if name in self._local_checksums:
                check_checksum(file, self._local_checksums[name])
            return load_model(torch.load(file, map_location="cpu"))

        # Try loading as a remote model
        if name in self._models:
            model_info = self._models[name]
            url = model_info["url"]

            # Determine cache path
            cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)
            filename = os.path.basename(url)
            cache_path = cache_dir / filename

            # Get expected hash from model info
            expected_hash = model_info.get("hash")
            if expected_hash is None and "-" in filename:
                # Try to get hash from filename (model-<hash>.th format)
                expected_hash = filename.split("-")[-1].split(".")[0]

            # Check if file exists and verify hash
            if cache_path.exists():
                if expected_hash:
                    try:
                        check_checksum(cache_path, expected_hash)
                        return load_model(torch.load(cache_path, map_location="cpu"))
                    except ModelLoadingError:
                        # Invalid checksum, delete and redownload
                        cache_path.unlink()
                else:
                    # No hash available, use existing file
                    return load_model(torch.load(cache_path, map_location="cpu"))

            # Update progress bar description
            if progress_bar and task_id:
                desc = f"Downloading {name}"
                if collection_name:
                    desc = f"Downloading {collection_name} - {name}"
                progress_bar.update(task_id, description=desc)

            # Download and load the model
            return self._download_and_load_model(
                url=url,
                cache_path=cache_path,
                expected_hash=expected_hash,
                progress_bar=progress_bar,
                task_id=task_id,
            )

        # Try loading as a collection
        if name in self._collections:
            collection = self._collections[name]
            signatures = collection["models"]

            # For collections, download each model
            models = []
            for sig in signatures:
                if progress_bar and task_id:
                    # Update description to show collection progress
                    desc = f"Collection {name} ({len(models) + 1}/{len(signatures)})"
                    progress_bar.update(task_id, description=desc)

                # Download each model, passing the collection name for nested progress
                model = self.get_model(
                    sig,
                    progress_bar=progress_bar,
                    task_id=task_id,
                    collection_name=name,
                )
                models.append(model)

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

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the cache.

        Args:
            name: Model name or signature

        Returns:
            True if the model was successfully removed, False otherwise
        """
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"

        # For single models
        if name in self._models:
            url = self._models[name].get("url")
            if url:
                filename = os.path.basename(url)
                model_path = cache_dir / filename
                if model_path.exists():
                    model_path.unlink()
                    return True

        # For collections, remove all component models
        elif name in self._collections:
            removed_any = False
            for component in self._collections[name].get("models", []):
                if component in self._models:
                    url = self._models[component].get("url")
                    if url:
                        filename = os.path.basename(url)
                        model_path = cache_dir / filename
                        if model_path.exists():
                            model_path.unlink()
                            removed_any = True
            return removed_any

        return False
