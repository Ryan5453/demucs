# Demucs API

The Demucs Python API is primarily comprised of two classes: `Separator` and `SeparatedSources`.

## Separator

The `Separator` class is a higher level representation of a Demucs audio source separation model. When you want to separate an audio file into its constituent stems, you will first need to create an instance of the `Separator` class which will load the model into memory for use.

```python
separator = Separator(
    model: str | AnyModel = "htdemucs", 
    device: str = "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
)
```

A `Separator` only takes in two parameters, the model to use for separation, which backend to use for loading and running the model. Demucs can usually auto-detect the best backend to use based on the availability of the hardware.

Once you have a `Separator` instance, you can use the `separate` method to separate an audio file into its constituent stems.

```python
def separate(
    audio: Tensor | Path | str | bytes,
    shifts: int = 1,
    overlap: float = 0.25,
    split: bool = True,
    segment: int | None = None,
    sample_rate: int | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> SeparatedSources:
```

When separating audio, you primarily only need to care about the `audio` parameter. The `audio` parameter can be a `Tensor` matching the model expectations, a file path, or raw audio bytes.

**Note**: The `Separator` API operates silently and never prints to the console. If you need progress updates, use the callback system described in the [Progress Callbacks](#progress-callbacks) section.

## SeparatedSources

After running the `Separator`'s `separate` method, you will be returned a `SeparatedSources` instance. This instance contains the separated audio sources, the sample rate of the audio, and the original audio.

If you're happy with the pure audio stems, you have the ability to export them to an audio container (rather than the Tensors that are stored in the `SeparatedSources` instance). 

```python
def export_stem(
    stem_name: str,
    path: Path | str | None = None, # Format extension will be added if not provided
    format: str = "wav",
    clip: ClipMode = ClipMode.rescale,
    encoding: str | None = None, # Only used for WAV and FLAC
    bits_per_sample: int | None = None, # Only used for WAV and FLAC
) -> Path | bytes:
```

## Progress Callbacks

Demucs provides a callback-based system for monitoring progress during long-running operations like model downloads and audio processing. This system is designed to be UI-agnostic, allowing you to implement progress displays in any framework.

### Model Download Progress

When using the `Separator` class with models that need to be downloaded, you can provide a progress callback to the `ModelRepository`:

```python
from demucs.repo import ModelRepository

def progress_callback(event_type: str, data: dict):
    if event_type == "download_start":
        print(f"Starting download of {data['model_name']} ({data['total_layers']} layers)")
    elif event_type == "layer_start":
        print(f"Downloading layer {data['layer_index']}/{data['total_layers']}")
    elif event_type == "layer_progress":
        print(f"Layer {data['layer_index']}: {data['progress_percent']:.1f}%")
    elif event_type == "layer_complete":
        if data.get("cached"):
            print(f"Layer {data['layer_index']} (cached)")
        else:
            print(f"Layer {data['layer_index']} complete")
    elif event_type == "download_complete":
        print(f"Download complete: {data['model_name']}")

model_repo = ModelRepository()
model = model_repo.get_model("htdemucs", progress_callback=progress_callback)
model.eval()
```

#### Model Download Events

- **`download_start`**: Emitted when model download begins
  - `model_name`: Name of the model being downloaded
  - `total_layers`: Total number of layers to download

- **`layer_start`**: Emitted when each layer download starts
  - `model_name`: Name of the model
  - `layer_index`: Current layer number (1-based)
  - `total_layers`: Total number of layers
  - `layer_size_bytes`: Size of the layer in bytes

- **`layer_progress`**: Emitted during layer download with progress updates
  - `model_name`: Name of the model
  - `layer_index`: Current layer number (1-based)
  - `total_layers`: Total number of layers
  - `progress_percent`: Download progress for this layer (0-100)
  - `downloaded_bytes`: Bytes downloaded so far
  - `total_bytes`: Total bytes for this layer
  - `phase` (optional): Current phase ("loading", "verifying")

- **`layer_complete`**: Emitted when each layer finishes downloading
  - `model_name`: Name of the model
  - `layer_index`: Layer number that completed (1-based)
  - `total_layers`: Total number of layers
  - `cached` (optional): `True` if layer was already cached

- **`download_complete`**: Emitted when entire model download is finished
  - `model_name`: Name of the model
  - `total_layers`: Total number of layers downloaded

### Audio Processing Progress

When using `apply_model()` directly with split processing, you can monitor chunk processing progress:

```python
from demucs.apply import apply_model

def processing_callback(event_type: str, data: dict):
    if event_type == "processing_start":
        print(f"Processing {data['total_chunks']} chunks")
    elif event_type == "chunk_complete":
        print(f"Completed {data['completed_chunks']}/{data['total_chunks']} chunks")
    elif event_type == "processing_complete":
        print("Audio processing complete")

result = apply_model(
    model, 
    audio_tensor, 
    split=True, 
    progress_callback=processing_callback
)
```

#### Audio Processing Events

- **`processing_start`**: Emitted when chunk processing begins
  - `total_chunks`: Number of audio chunks to process

- **`chunk_complete`**: Emitted after each chunk is processed
  - `completed_chunks`: Number of chunks completed so far
  - `total_chunks`: Total number of chunks

- **`processing_complete`**: Emitted when all chunks are finished
  - `total_chunks`: Total number of chunks processed

### Example: Rich Progress Bar

Here's an example of how to create a Rich progress bar using the callback system:

```python
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from demucs.repo import ModelRepository

def create_rich_callback(progress_bar, task):
    def callback(event_type: str, data: dict):
        if event_type == "layer_start":
            progress_bar.update(
                task,
                description=f"Downloading {data['model_name']} - Layer {data['layer_index']}/{data['total_layers']}"
            )
        elif event_type == "layer_progress":
            # Calculate overall progress across all layers
            layer_base = (data['layer_index'] - 1) / data['total_layers'] * 100
            layer_progress = data['progress_percent'] / data['total_layers']
            overall_progress = layer_base + layer_progress
            progress_bar.update(task, completed=overall_progress)
        elif event_type == "download_complete":
            progress_bar.update(task, completed=100)
    return callback

with Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
) as progress:
    task = progress.add_task("Downloading model...", total=100)
    callback = create_rich_callback(progress, task)
    model_repo = ModelRepository()
    model = model_repo.get_model("htdemucs", progress_callback=callback)
    model.eval()
```

This callback system allows you to integrate Demucs progress reporting into any UI framework, whether it's a CLI tool, web application, desktop GUI, or any other interface.
