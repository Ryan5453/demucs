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

BASE_CDN_URL = "https://dl.fbaipublicfiles.com/demucs"


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

        # Generate layer URLs from model remote paths
        self._layer_urls = {}
        for model_name, model_info in self._models.items():
            if "models" in model_info:
                for model_entry in model_info["models"]:
                    checksum = model_entry["checksum"]
                    remote_path = model_entry["remote"]
                    self._layer_urls[checksum] = f"{BASE_CDN_URL}/{remote_path}"

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
                    checksum = component["checksum"]
                    if checksum in cached_layers:
                        components[checksum] = cached_layers[checksum]
                        total_size += cached_layers[checksum]["size_bytes"]
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
        model_name: str = "",
        layer_index: int = 1,
        total_layers: int = 1,
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

                # Calculate base progress from previous layers
                layer_base_progress = ((layer_index - 1) / total_layers) * 100
                layer_max_progress = (layer_index / total_layers) * 100

                # Update progress bar for download phase
                if progress_bar and task_id:
                    if total_size:
                        desc = f"[cyan]Downloading {model_name}[/cyan] - Layer {layer_index}/{total_layers} ({format_file_size(total_size)})"
                    else:
                        desc = f"[cyan]Downloading {model_name}[/cyan] - Layer {layer_index}/{total_layers}"

                    progress_bar.update(
                        task_id, description=desc, completed=layer_base_progress
                    )

                chunk_counter = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    buffer.write(chunk)
                    downloaded_size += len(chunk)
                    chunk_counter += 1

                    # Update progress every few chunks to avoid too frequent updates
                    if (
                        progress_bar
                        and task_id is not None
                        and (chunk_counter % 20 == 0)
                    ):
                        if total_size and total_size > 0:
                            # Calculate progress within this layer (0-90% of layer progress)
                            layer_download_progress = (
                                downloaded_size / total_size
                            ) * 0.9
                            current_progress = layer_base_progress + (
                                layer_download_progress
                                * (layer_max_progress - layer_base_progress)
                            )
                        else:
                            # If no total_size, show incremental progress based on downloaded chunks
                            # Use a simple heuristic: assume each chunk represents some progress
                            chunk_count = (
                                downloaded_size // 8192
                            )  # Number of 8KB chunks downloaded
                            # Estimate progress based on chunks (more chunks = more progress, but cap at 90% of layer)
                            estimated_progress = min(
                                chunk_count * 0.5, 90
                            )  # 0.5% per chunk, max 90%
                            current_progress = layer_base_progress + (
                                estimated_progress / 100
                            ) * (layer_max_progress - layer_base_progress)

                        # Update progress
                        progress_bar.update(task_id, completed=current_progress)

                buffer.seek(0)

                # Try to load as a PyTorch model directly from memory
                try:
                    if progress_bar and task_id:
                        # 90% of this layer's progress for loading
                        loading_progress = layer_base_progress + (
                            0.9 * (layer_max_progress - layer_base_progress)
                        )
                        progress_bar.update(
                            task_id,
                            description=f"[cyan]Downloading {model_name}[/cyan] - Layer {layer_index}/{total_layers} (loading...)",
                            completed=loading_progress,
                        )

                    # Save to a temporary file first
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".th"
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        tmp_file.write(buffer.getvalue())

                    # Load the model to verify it's valid
                    model_data = torch.load(
                        tmp_path, map_location="cpu", weights_only=False
                    )

                    if progress_bar and task_id:
                        # 95% of this layer's progress for verification
                        verify_progress = layer_base_progress + (
                            0.95 * (layer_max_progress - layer_base_progress)
                        )
                        progress_bar.update(
                            task_id,
                            description=f"[cyan]Downloading {model_name}[/cyan] - Layer {layer_index}/{total_layers} (verifying...)",
                            completed=verify_progress,
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
                        # This layer is complete
                        progress_bar.update(
                            task_id,
                            completed=layer_max_progress,
                            description=f"[cyan]Downloading {model_name}[/cyan] - Layer {layer_index}/{total_layers} (complete)",
                        )

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
        layer_checksums = [entry["checksum"] for entry in model_info["models"]]

        # Download each layer
        layers = []
        cache_dir = get_cache_dir()
        total_layers = len(layer_checksums)

        for i, layer_checksum in enumerate(layer_checksums):
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
                    layer = load_model(
                        torch.load(cache_path, map_location="cpu", weights_only=False)
                    )
                    layers.append(layer)

                    # Update progress for cached layer
                    if progress_bar and task_id:
                        # Calculate overall progress: (completed_layers / total_layers) * 100
                        layer_progress = ((i + 1) / total_layers) * 100
                        progress_bar.update(
                            task_id,
                            completed=layer_progress,
                            description=f"[cyan]Downloading {name}[/cyan] - Layer {i + 1}/{total_layers} (cached)",
                        )
                    continue
                except (ModelLoadingError, Exception):
                    # If validation or loading fails, delete and redownload
                    if cache_path.exists():
                        cache_path.unlink()

            # Update progress bar description for download
            if progress_bar and task_id is not None:
                progress_bar.update(
                    task_id,
                    description=f"[cyan]Downloading {name}[/cyan] - Layer {i + 1}/{total_layers}",
                )

            # Download and load the layer
            layer = self._download_and_load_layer(
                url=url,
                cache_path=cache_path,
                expected_checksum=layer_checksum,
                progress_bar=progress_bar,
                task_id=task_id,
                model_name=name,
                layer_index=i + 1,
                total_layers=total_layers,
            )
            layers.append(layer)

            # Update progress after successful layer download
            if progress_bar and task_id:
                layer_progress = ((i + 1) / total_layers) * 100
                progress_bar.update(
                    task_id,
                    completed=layer_progress,
                    description=f"[cyan]Downloading {name}[/cyan] - Layer {i + 1}/{total_layers} (complete)",
                )
        if progress_bar and task_id:
            progress_bar.update(
                task_id,
                completed=100,
                description=f"[green]Downloaded {name}[/green] - All {total_layers} layers complete",
            )

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
        for layer_info in self._models[name].get("models", []):
            layer_checksum = layer_info["checksum"]
            filename = f"{layer_checksum}.th"
            layer_path = cache_dir / filename
            if layer_path.exists():
                layer_path.unlink()
                removed_any = True

        return removed_any
