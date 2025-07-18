# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json
import os
import shutil
import tempfile
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, TypeAlias

import httpx
import torch
from rich.progress import Progress, TaskID

from .apply import BagOfModels, Model
from .states import load_model
from .errors import ModelLoadingError

# Type alias for models
AnyModel: TypeAlias = Model | BagOfModels

# Base URL for model downloads
BASE_MODEL_URL = "https://github.com/Ryan5453/demucs/releases/download/v5.0.0-models/"


def check_checksum(path: Path, checksum: str):
    """
    Verifies that a file matches an expected checksum.

    :param path: Path to the file to check
    :param checksum: Expected SHA-256 checksum (first 8 characters)
    :raises ModelLoadingError: If the actual checksum does not match the expected one
    """
    sha = sha256()
    with open(path, "rb") as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:8]
    if actual_checksum != checksum:
        raise ModelLoadingError(
            f"Invalid checksum for file {path}, "
            f"expected {checksum} but got {actual_checksum}"
        )


def format_file_size(size_bytes: int) -> str:
    """
    Formats a file size in a human-readable way

    :param size_bytes: The size of the file in bytes
    :return: A string representing the size of the file in a human-readable way
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_cache_dir() -> Path:
    """
    Get the cache directory for Demucs models.

    :return: Path to the cache directory
    """
    home = Path.home()
    cache_dir = home / ".demucs" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def generate_model_url(checksum: str) -> str:
    """
    Generate a model download URL from a checksum.

    :param checksum: Model checksum identifier
    :return: Full download URL for the model
    """
    return f"{BASE_MODEL_URL}{checksum}.th"


class ModelRepository:
    """Repository system for accessing models."""

    def __init__(self):
        """
        Initialize the repository.
        """
        # Determine metadata_path relative to this file
        current_file_path = Path(__file__)
        self.metadata_path = current_file_path.parent / "metadata.json"

        # Load metadata
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        if "models" in self.metadata:
            self._models = self.metadata["models"]
        else:
            raise ModelLoadingError(
                "Invalid metadata structure: 'models' key not found in metadata.json. "
                "The expected format is a top-level 'models' dictionary."
            )

        # Generate layer URLs dynamically from model checksums
        self._layer_urls = {}
        for model_name, model_info in self._models.items():
            if "models" in model_info:
                for checksum in model_info["models"]:
                    self._layer_urls[checksum] = generate_model_url(checksum)

    def has_model(self, name: str) -> bool:
        """Check if a model exists."""
        return name in self._models

    def get_cache_info(self) -> Dict[str, Dict]:
        """
        Get information about cached models.

        Returns:
            Dictionary with information about cached models
        """
        cache_dir = get_cache_dir()
        cached_models = {}

        # Check which layer files are downloaded
        cached_layers = {}
        for checksum, url in self._layer_urls.items():
            filename = f"{checksum}.th"
            layer_path = cache_dir / filename

            if layer_path.exists():
                size_bytes = layer_path.stat().st_size
                cached_layers[checksum] = {
                    "path": str(layer_path),
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024 * 1024),
                }

        # Handle models
        for name, info in self._models.items():
            if "models" in info:
                component_layers = info["models"]
                all_cached = True
                total_size = 0
                components = {}

                for component in component_layers:
                    if component in cached_layers:
                        components[component] = cached_layers[component]
                        total_size += cached_layers[component]["size_bytes"]
                    else:
                        all_cached = False

                if all_cached:
                    cached_models[name] = {
                        "type": "model",
                        "layers": components,
                        "size_bytes": total_size,
                        "size_mb": total_size / (1024 * 1024),
                    }

        return cached_models

    def _download_and_load_layer(
        self,
        url: str,
        cache_path: Path,
        expected_checksum: str,
        progress_bar: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> AnyModel:
        """Download and load a model layer from a URL."""
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

                    # Load the model to verify it's valid
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

                    # Verify checksum
                    try:
                        check_checksum(cache_path, expected_checksum)
                    except ModelLoadingError:
                        cache_path.unlink()
                        raise

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
        progress_bar: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
    ) -> AnyModel:
        """
        Get a model by name.

        Args:
            name: Model name
            progress_bar: Optional rich.progress.Progress instance for download progress
            task_id: Optional TaskID for the progress bar

        Returns:
            The requested model
        """
        # Only load models, not individual layers
        if name not in self._models:
            raise ModelLoadingError(
                f"Could not find a model with name {name}. "
                f"Available models: {', '.join(self._models.keys())}"
            )

        model_info = self._models[name]
        layer_checksums = model_info["models"]

        # Download each layer
        layers = []
        cache_dir = get_cache_dir()

        for layer_checksum in layer_checksums:
            if layer_checksum not in self._layer_urls:
                raise ModelLoadingError(f"Layer {layer_checksum} not found in metadata")

            url = self._layer_urls[layer_checksum]

            # Determine cache path
            filename = f"{layer_checksum}.th"
            cache_path = cache_dir / filename

            # Check if file exists and validate its integrity
            if cache_path.exists():
                try:
                    # Validate checksum
                    check_checksum(cache_path, layer_checksum)

                    # Try to load the model
                    layer = load_model(torch.load(cache_path, map_location="cpu"))
                    layers.append(layer)
                    continue
                except (ModelLoadingError, Exception):
                    # If validation or loading fails, delete and redownload
                    if cache_path.exists():
                        cache_path.unlink()

            # Update progress bar description
            if progress_bar and task_id:
                desc = f"Downloading layer {layer_checksum} ({len(layers) + 1}/{len(layer_checksums)})"
                progress_bar.update(task_id, description=desc)

            # Download and load the layer
            layer = self._download_and_load_layer(
                url=url,
                cache_path=cache_path,
                expected_checksum=layer_checksum,
                progress_bar=progress_bar,
                task_id=task_id,
            )
            layers.append(layer)

        # Create BagOfModels from the layers
        weights = model_info.get("weights")
        segment = model_info.get("segment")
        return BagOfModels(layers, weights, segment)

    def list_models(self) -> Dict[str, Dict]:
        """
        List all available models.

        Returns:
            Dictionary mapping model names to their metadata
        """
        result = {}

        # Add models
        for name, info in self._models.items():
            result[name] = {"type": "model", "info": info}

        return result

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the cache.

        Args:
            name: Model name

        Returns:
            True if the model was successfully removed, False otherwise
        """
        if name not in self._models:
            return False

        cache_dir = get_cache_dir()
        removed_any = False

        # Remove all layer files for this model
        for layer_checksum in self._models[name].get("models", []):
            filename = f"{layer_checksum}.th"
            layer_path = cache_dir / filename
            if layer_path.exists():
                layer_path.unlink()
                removed_any = True

        return removed_any
