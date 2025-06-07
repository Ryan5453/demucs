# Demucs Python API

Demucs has a Python API that allows you to use the various pre-trained models to separate audio files into stems.

## `Separator`

The `Separator` class is the main class for separating audio files into stems.
It handles model loading, audio processing, and applying the separation model.

**Note:** Requires FFmpeg to be installed for audio file loading.
Install with: `conda install -c conda-forge 'ffmpeg<7'`

### Example

```python
from demucs.api import Separator

# Initialize Separator with a specific model
separator = Separator(model="htdemucs_ft", segment=4)

# Separate an audio file
separated_sources = separator.separate_audio_file("my_song.mp3")

# Save all stems to a directory
saved_paths = separated_sources.save_all_stems(output_dir="separated_output/")

# Isolate a stem and its complement
vocals_and_accompaniment = separated_sources.isolate_stem("vocals")
vocals_and_accompaniment.save_stem("vocals", "vocals_only.wav")
vocals_and_accompaniment.save_stem("no_vocals", "accompaniment_only.wav")
```

### Constructor

```python
separator = Separator(
    model: Union[str, AnyModel] = DEFAULT_MODEL,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    shifts: int = 1,
    overlap: float = 0.25,
    split: bool = True,
    segment: Optional[int] = None, # Default is different for each model
    jobs: int = 0,
    verbose: bool = False,
)
```

**Parameters:**

*   `model`: Model to use. Can be a model name string (e.g., `"htdemucs"`, `"mdx_q"`) or a pre-loaded model instance. Defaults to `demucs.pretrained.DEFAULT_MODEL`.
*   `device`: Device for processing (`"cuda"`, `"mps"`, `"cpu"`). Auto-detected if not specified.
*   `shifts`: Number of random shifts for equivariant stabilization. Higher values improve quality but increase processing time.
*   `overlap`: Overlap between processing chunks (0.0 to 1.0).
*   `split`: Whether to split input into chunks.
*   `segment`: Length (in seconds) of each chunk (if `split=True`). If `None`, model-specific default is used. Note: Transformer models have a maximum segment length.
*   `jobs`: Number of parallel jobs (0 for automatic).
*   `verbose`: Show progress bars (default `False`).

### Methods

#### `update_parameter(...)`

Updates separation parameters dynamically.
Accepts any of the constructor parameters as keyword arguments.

```python
separator.update_parameter(shifts=2, overlap=0.5)
```

#### `separate_audio_file(file: Union[str, Path]) -> SeparatedSources`

Separates an audio file into stems.

*   `file`: Path to the audio file.
*   **Returns:** A `SeparatedSources` object.
*   **Raises:** `LoadAudioError` if the audio file cannot be loaded.

#### `separate_audio_bytes(audio_bytes: bytes) -> SeparatedSources`

Separates audio from raw bytes (e.g., from an uploaded file or API request).

*   `audio_bytes`: Raw audio bytes.
*   **Returns:** A `SeparatedSources` object.
*   **Raises:** `LoadAudioError` if the audio bytes cannot be loaded.

#### `separate_tensor(wav: Tensor, sr: Optional[int] = None) -> SeparatedSources`

Separates a pre-loaded audio tensor.

*   `wav`: Audio tensor of shape `[channels, samples]`.
*   `sr`: Sample rate of `wav`. If `None`, assumes it matches the model's sample rate.
*   **Returns:** A `SeparatedSources` object.

## `SeparatedSources`

A container object returned by `Separator` methods, holding the separated audio stems.
It provides methods to access, save, and export these stems.

### Methods

#### `save_stem(stem_name: str, path: Union[str, Path], **kwargs) -> Path`

Saves a specific stem to a file as a 32-bit float WAV.

*   `stem_name`: Name of the stem to save.
*   `path`: Output file path. `.wav` extension is added if not present.
*   `**kwargs`: Additional arguments for `demucs.audio.save_audio` (e.g., `clip=ClipMode.clamp`).
*   **Returns:** `Path` object of the saved file.

#### `save_all_stems(output_dir: Union[str, Path], filename_template: str = "{stem_name}", **kwargs) -> Dict[str, Path]`

Saves all stems to a directory.

*   `output_dir`: Directory to save stems.
*   `filename_template`: Filename template. `{stem_name}` is replaced by the stem name (e.g., `"{stem_name}_mix.wav"`).
*   `**kwargs`: Additional arguments for `demucs.audio.save_audio`.
*   **Returns:** A dictionary mapping stem names to their saved `Path` objects.

#### `export_stem(stem_name: str, format: str = "wav", clip: ClipMode = ClipMode.rescale) -> bytes`

Exports a stem as raw audio bytes in memory.

*   `stem_name`: Name of the stem.
*   `format`: Desired audio format (e.g., `"wav"`, `"flac"`, `"mp3"`).
*   `clip`: Clipping mode from `demucs.audio.ClipMode` (e.g., `ClipMode.rescale`, `ClipMode.clamp`).
*   **Returns:** Raw audio `bytes`.

#### `export_all_stems(format: str = "wav", clip: ClipMode = ClipMode.rescale) -> Dict[str, bytes]`

Exports all stems as raw audio bytes.

*   `format`: Audio format.
*   `clip`: Clipping mode.
*   **Returns:** Dictionary mapping stem names to `bytes`.

#### `add_complement_stem(name: str, method: OtherMethod = OtherMethod.minus) -> SeparatedSources`

Adds the complement of a specified stem (e.g., "no_vocals") to the `SeparatedSources` object in-place.
The complement is everything else in the mix.

*   `name`: Name of the stem to create a complement for (e.g., `"vocals"` will create `"no_vocals"`).
*   `method`: `OtherMethod.minus` (original - stem) or `OtherMethod.add` (sum of other stems).
*   **Returns:** The same `SeparatedSources` object, now modified.

#### `isolate_stem(name: str, method: OtherMethod = OtherMethod.minus) -> SeparatedSources`

Creates a *new* `SeparatedSources` object containing only the specified stem and its complement.

*   `name`: Name of the stem to isolate.
*   `method`: `OtherMethod.minus` or `OtherMethod.add` for complement calculation.
*   **Returns:** A new `SeparatedSources` object.

## Model Management

Demucs provides utilities for managing models, including listing available models, checking cache status, downloading models, and more.

### `list_models() -> Dict[str, Dict]`

Lists all available pre-trained models from the Demucs model repository.

*   **Returns:** A dictionary where keys are model names and values are dictionaries of their metadata.

```python
from demucs.api import list_models

available_models = list_models()
print(f"Available models: {list(available_models.keys())}")
```

### `get_version() -> str`

Returns the installed version of Demucs.

*   **Returns:** Version string.

```python
from demucs.api import get_version

print(f"Demucs version: {get_version()}")
```

### `ModelRepository`

For advanced model management, Demucs provides the `ModelRepository` class that gives direct access to the model repository.

```python
from demucs.api import ModelRepository

# Initialize the repository
repo = ModelRepository()

# List available models
models = repo.list_models()
print(f"Available models: {list(models.keys())}")

# Get information about cached models
cache_info = repo.get_cache_info()
print(f"Downloaded models: {list(cache_info.keys())}")

# Download a model
model = repo.get_model("htdemucs")

# Remove a model from cache
repo.remove_model("htdemucs")
```

#### `list_models() -> Dict[str, Dict]`

Lists all available models in the repository.

* **Returns:** A dictionary mapping model names to their metadata.

#### `get_cache_info() -> Dict[str, Dict]`

Gets information about currently cached models.

* **Returns:** A dictionary mapping model names to cache information (path, size, etc.).

#### `get_model(name: str) -> AnyModel`

Downloads (if necessary) and loads a model by name.

* `name`: Name of the model to get.
* **Returns:** The loaded model.
* **Raises:** `ModelLoadingError` if the model cannot be loaded.

#### `remove_model(name: str) -> bool`

Removes a model from the cache.

* `name`: Name of the model to remove.
* **Returns:** `True` if the model was successfully removed, `False` otherwise.

#### `has_model(name: str) -> bool`

Checks if a model exists in the repository.

* `name`: Name of the model to check.
* **Returns:** `True` if the model exists, `False` otherwise.

### `get_cache_dir() -> Path`

Returns the path to the Demucs model cache directory.

* **Returns:** Path object for the cache directory.

```python
from demucs.api import get_cache_dir

cache_path = get_cache_dir()
print(f"Models are stored in: {cache_path}")
```

## Exception Classes

Demucs API defines several exception classes that you might encounter when working with the library.

### `ModelLoadingError`

Raised when a model cannot be loaded, either due to download issues, invalid checksums, or other model-related problems.

### `LoadAudioError`

Raised when an audio file or audio bytes cannot be loaded or processed.

### `LoadModelError`

Raised when there's an issue loading a model, typically related to model format or compatibility.

### `SegmentValidationError`

Raised when the provided segment parameter is invalid for the selected model, particularly for transformer models with maximum segment length restrictions.