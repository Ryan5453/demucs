# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from typing_extensions import Annotated

from . import __version__
from .api import Separator, select_model
from .exceptions import ModelLoadingError
from .repo import ModelRepository

METADATA_PATH = Path(__file__).parent / "metadata.json"


class DeviceType(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class ModelName(str, Enum):
    auto = "auto"
    hdemucs_mmi = "hdemucs_mmi"
    htdemucs = "htdemucs"
    htdemucs_ft = "htdemucs_ft"
    htdemucs_6s = "htdemucs_6s"


class StemName(str, Enum):
    drums = "drums"
    bass = "bass"
    other = "other"
    vocals = "vocals"
    guitar = "guitar"
    piano = "piano"


class ClipMode(str, Enum):
    rescale = "rescale"
    clamp = "clamp"
    tanh = "tanh"
    none = "none"


console = Console()


def _get_common_audio_extensions() -> set[str]:
    """
    Get a set of common audio file extensions.
    This is used as a fast heuristic filter - the actual file validation
    is done by torchcodec's AudioDecoder when loading.
    """
    # Common audio/video formats that typically contain audio streams
    # Note: This is just a heuristic for performance - torchcodec determines actual support
    return {
        ".mp3",
        ".wav",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".mp4",
        ".webm",
        ".mkv",
        ".avi",
        ".mov",
        ".wma",
        ".alac",
    }


def _looks_like_audio_file(path: Path) -> bool:
    """
    Fast heuristic check if a file might be audio based on extension.
    This is just for performance - actual validation happens when torchcodec loads the file.
    """
    return path.suffix.lower() in _get_common_audio_extensions()


def _expand_paths_to_audio_files(paths: list[Path]) -> list[Path]:
    """
    Expand directory paths to include all audio files, keep regular files as-is.
    Uses common extensions as a fast filter - torchcodec validates actual support at load time.
    """
    audio_files = []

    for path in paths:
        if path.is_file():
            # For individual files, just add them and let torchcodec handle validation
            # This allows users to try any file they want
            audio_files.append(path)
        elif path.is_dir():
            # For directories, use extension heuristic for performance
            # Otherwise we'd have to probe every single file which could be very slow
            all_files = [
                f for f in path.iterdir() if f.is_file() and not f.name.startswith(".")
            ]
            found_files = [f for f in all_files if _looks_like_audio_file(f)]

            if found_files:
                # Sort files for consistent processing order
                found_files.sort()
                audio_files.extend(found_files)
            else:
                console.print(
                    f"[yellow]Warning:[/yellow] No audio files found in '{path}'"
                )
        else:
            console.print(f"[red]Error:[/red] Path '{path}' does not exist")

    return audio_files


# Progress bar helper functions
def _create_model_progress_bar():
    """Create a standardized progress bar for model operations."""

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=2,
        expand=True,
    )


def _create_progress_callback(progress_bar, task):
    """Create a progress callback that updates a Rich progress bar for model downloads."""

    def callback(event_type: str, data: dict):
        if event_type == "layer_start":
            progress_bar.update(
                task,
                description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']}",
            )
        elif event_type == "layer_progress":
            # Calculate overall progress: (completed_layers + current_layer_progress) / total_layers
            layer_base = (data["layer_index"] - 1) / data["total_layers"] * 100
            layer_progress = data["progress_percent"] / data["total_layers"]
            overall_progress = layer_base + layer_progress

            phase_text = ""
            if "phase" in data:
                phase_text = f" ({data['phase']})"

            progress_bar.update(
                task,
                completed=overall_progress,
                description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']}{phase_text}",
            )
        elif event_type == "layer_complete":
            if data.get("cached"):
                progress_bar.update(
                    task,
                    completed=(data["layer_index"] / data["total_layers"]) * 100,
                    description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']} (cached)",
                )
            else:
                progress_bar.update(
                    task,
                    completed=(data["layer_index"] / data["total_layers"]) * 100,
                    description=f"[cyan]Downloading {data['model_name']}[/cyan] - Layer {data['layer_index']}/{data['total_layers']} (complete)",
                )
        elif event_type == "download_complete":
            progress_bar.update(
                task,
                completed=100,
                description=f"[green]Downloaded {data['model_name']}[/green] - All {data['total_layers']} layers complete",
            )

    return callback


def _create_file_progress_bar():
    """Create a progress bar optimized for processing multiple files."""
    return Progress(
        SpinnerColumn(finished_text="[green]✓[/green]"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
        refresh_per_second=4,
        expand=True,
    )


class FileProgressTracker:
    """Tracks progress across multiple files with improved UX."""

    def __init__(self, total_files: int):
        self.total_files = total_files
        self.completed_files = 0
        self.progress_bar = None
        self.current_task = None
        self.file_tasks = {}

    def __enter__(self):
        self.progress_bar = _create_file_progress_bar()
        self.progress_bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.__exit__(exc_type, exc_val, exc_tb)

    def start_file(self, filename: str) -> int:
        """Start processing a new file."""
        # Create task for this file
        task_id = self.progress_bar.add_task(filename.strip(), total=100, completed=0)
        self.file_tasks[filename] = task_id
        return task_id

    def update_file_progress(self, filename: str, event_type: str, data: dict):
        """Update progress for a specific file."""
        if filename not in self.file_tasks:
            return

        task_id = self.file_tasks[filename]

        if event_type == "processing_start":
            self.progress_bar.update(
                task_id,
                description=filename.strip(),
                total=data["total_chunks"],
                completed=0,
            )
        elif event_type == "chunk_complete":
            self.progress_bar.update(
                task_id,
                completed=data["completed_chunks"],
                description=filename.strip(),
            )
        elif event_type == "processing_complete":
            # Complete the task which will show the checkmark in the spinner column
            self.progress_bar.update(
                task_id, completed=data["total_chunks"], description=filename.strip()
            )
            self.completed_files += 1

    def error_file(self, filename: str, error_msg: str):
        """Mark a file as having an error."""
        if filename not in self.file_tasks:
            return

        task_id = self.file_tasks[filename]
        # Complete the task to stop spinner and show red error mark
        self.progress_bar.update(
            task_id, completed=100, description=f"[red]✗[/red] {filename.strip()}"
        )
        self.completed_files += 1

    def create_audio_callback(self, filename: str):
        """Create a callback for audio processing progress."""

        def callback(event_type: str, data: dict):
            self.update_file_progress(filename, event_type, data)

        return callback


def _download_model_with_progress(name: str) -> bool:
    """
    Download a single model with progress display.
    Returns True if successful, False otherwise.
    """

    models = get_models()

    # Show initial message
    if name in models:
        layer_count = len(models[name]["models"])
        layer_word = "layer" if layer_count == 1 else "layers"
        console.print(
            f"[bold]Downloading {name} ({layer_count} {layer_word})...[/bold]"
        )

    with _create_model_progress_bar() as progress_bar:
        try:
            start_time = time.time()

            # Create progress task
            if name in models:
                layer_count = len(models[name]["models"])
                layer_word = "layer" if layer_count == 1 else "layers"
                task = progress_bar.add_task(
                    f"[cyan]Downloading {name} ({layer_count} {layer_word})[/cyan]",
                    total=100,
                    completed=0,
                )
            else:
                task = progress_bar.add_task(
                    f"[cyan]Downloading {name}[/cyan]",
                    total=100,
                    completed=0,
                )

            # Download the model using callback
            callback = _create_progress_callback(progress_bar, task)
            model_repo = ModelRepository()
            model = model_repo.get_model(name=name, progress_callback=callback)
            model.eval()

            progress_bar.remove_task(task)

            # Show success message
            download_time = time.time() - start_time
            model_repo = ModelRepository()
            cache_info = model_repo.get_cache_info()

            num_sources = len(model.sources)
            if name in models:
                layer_count = len(models[name]["models"])
                layer_word = "layer" if layer_count == 1 else "layers"
                model_type = f"{layer_count} {layer_word}"
            else:
                model_type = "Model"

            size_str = ""
            speed_str = ""
            if name in cache_info:
                size_bytes = cache_info[name]["size_bytes"]
                size_str = f" ({format_file_size(size_bytes)})"

                if download_time > 0.1:
                    speed = size_bytes / download_time
                    speed_str = f" at {format_file_size(speed)}/s"

            console.print(
                f"[green]✓[/green] [bold]{name}[/bold]: {model_type} with {num_sources} sources{size_str}{speed_str}"
            )
            return True

        except Exception as error:
            console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")
            return False


def _ensure_model_available(name: str) -> bool:
    """
    Ensure a model is available, downloading if necessary.
    Returns True if model is available, False otherwise.
    """
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    if name in cache_info:
        return True

    # Model needs to be downloaded
    return _download_model_with_progress(name)


def _download_models_batch(model_names: list[str]) -> None:
    """
    Download multiple models, showing progress for each.
    This is the unified download logic used by both download and separate commands.
    """
    # Create a ModelRepository to check if models are already downloaded
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    # Get models info
    models = get_models()

    # Filter out already downloaded models
    to_download = []
    for name in model_names:
        if name in cache_info:
            # Show already downloaded message
            if name in models:
                layer_count = len(models[name]["models"])
                layer_word = "layer" if layer_count == 1 else "layers"
                size_str = ""
                if name in cache_info:
                    size_bytes = cache_info[name]["size_bytes"]
                    size_str = f" ({format_file_size(size_bytes)})"
                console.print(
                    f"[green]✓[/green] [bold]{name}[/bold]: Already downloaded ({layer_count} {layer_word}{size_str})"
                )
        else:
            to_download.append(name)

    if not to_download:
        console.print("[green]All specified models are already downloaded.[/green]")
        return

    # Show download summary for multiple models
    if len(to_download) > 1:
        total_layers = sum(
            len(models[name]["models"]) for name in to_download if name in models
        )
        console.print(
            f"[bold]Downloading {len(to_download)} models ({total_layers} total layers)...[/bold]"
        )

    # Download each model using the shared progress logic
    with _create_model_progress_bar() as progress_bar:
        for name in to_download:
            _download_single_model_in_batch(name, models, progress_bar)

    console.print("[bold green]Download complete![/bold green]")


def _download_single_model_in_batch(name: str, models: dict, progress_bar) -> None:
    """Download a single model within an existing progress bar context."""

    try:
        start_time = time.time()

        if name in models:
            layer_count = len(models[name]["models"])
            layer_word = "layer" if layer_count == 1 else "layers"
            task = progress_bar.add_task(
                f"[cyan]Downloading {name} ({layer_count} {layer_word})[/cyan]",
                total=100,
                completed=0,
            )
        else:
            task = progress_bar.add_task(
                f"[cyan]Downloading {name}[/cyan]",
                total=100,
                completed=0,
            )

        # Download the model using callback
        callback = _create_progress_callback(progress_bar, task)
        model_repo = ModelRepository()
        model = model_repo.get_model(name=name, progress_callback=callback)
        model.eval()

        progress_bar.remove_task(task)

        # Show success message with timing info
        download_time = time.time() - start_time
        model_repo = ModelRepository()
        cache_info = model_repo.get_cache_info()

        num_sources = len(model.sources)
        if name in models:
            layer_count = len(models[name]["models"])
            layer_word = "layer" if layer_count == 1 else "layers"
            model_type = f"{layer_count} {layer_word}"
        else:
            model_type = "Model"

        size_str = ""
        speed_str = ""
        if name in cache_info:
            size_bytes = cache_info[name]["size_bytes"]
            size_str = f" ({format_file_size(size_bytes)})"

            if download_time > 0.1:
                speed = size_bytes / download_time
                speed_str = f" at {format_file_size(speed)}/s"

        console.print(
            f"[green]✓[/green] [bold]{name}[/bold]: {model_type} with {num_sources} sources{size_str}{speed_str}"
        )

    except ModelLoadingError as error:
        console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")
    except Exception as e:
        console.print(f"[red]✗[/red] [bold]{name}[/bold]: Unexpected error: {str(e)}")


def version_command():
    """
    Show the installed version of Demucs
    """
    typer.echo(f"Demucs version: {__version__}")


def list_models_command():
    """
    List all available models and show which ones are downloaded.
    """
    # Create a ModelRepository to manage models
    model_repo = ModelRepository()

    # Get models from metadata.json
    models = get_models()

    # Get cache info to determine which models are downloaded
    cache_info = model_repo.get_cache_info()

    # Create a table with detailed model information
    table = Table(title="Available Demucs Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Layers", style="blue")
    table.add_column("Segment", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Status", style="bright_green")

    # Use the order from metadata.json
    for name in models.keys():
        info = models[name]

        # Get model details
        layer_count = len(info.get("models", []))
        segment = info.get("segment", "N/A")

        # Determine if model is downloaded and its size
        is_downloaded = name in cache_info
        model_size = "N/A"
        if is_downloaded:
            model_size = format_file_size(cache_info[name]["size_bytes"])

        table.add_row(
            name,
            str(layer_count) + (" layer" if layer_count == 1 else " layers"),
            str(segment),
            model_size,
            "[green]Downloaded[/green]"
            if is_downloaded
            else "[red]Not Downloaded[/red]",
        )

    console.print(table)


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


def format_output_path(
    template: str, model: str, track: Path, stem: str, ext: str = "wav"
) -> Path:
    """Format output path template with variables"""
    now = datetime.now()
    # Template variables
    variables = {
        "model": model,
        "track": track.name.rsplit(".", 1)[0],
        "stem": stem,
        "ext": ext,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H-%M-%S"),
        "timestamp": str(int(now.timestamp())),
    }

    # Replace all variables in the template
    formatted_path = template
    for var, value in variables.items():
        formatted_path = formatted_path.replace(f"{{{var}}}", value)

    return Path(formatted_path)


def download_models_command(
    names: Annotated[
        list[str],
        typer.Argument(help="Model names to download."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option(
            "--all", help="Download all available models (may take some time)"
        ),
    ] = False,
):
    """
    Download and cache the specified models for offline use.
    """
    # Require either model names or --all flag
    if not all_models and (names is None or not names):
        console.print("[red]Error:[/red] No models specified for download.")
        console.print("Please either:")
        console.print("  1. Specify one or more model names to download")
        console.print("  2. Use [bold]--all[/bold] to download all available models")
        console.print("\nTo see available models, run: [bold]demucs models list[/bold]")
        return

    # Get model names to download
    if all_models:
        models = get_models()
        model_names = list(models.keys())
    else:
        model_names = names

    # Use the shared download logic
    _download_models_batch(model_names)


def remove_models_command(
    names: Annotated[
        list[str],
        typer.Argument(help="Model names to remove."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Remove all downloaded models"),
    ] = False,
):
    """
    Remove models from the cache to free up space.
    """

    # Create a ModelRepository to manage models
    model_repo = ModelRepository()

    # Get models to remove
    if all_models:
        # Get all downloaded models
        cache_info = model_repo.get_cache_info()
        model_names = list(cache_info.keys())
    else:
        if names is None or not names:
            console.print(
                "[yellow]No models specified for removal. Please specify at least one model name.[/yellow]"
            )
            return
        else:
            model_names = names

    if not model_names:
        console.print("[yellow]No models found to remove.[/yellow]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress_bar:
        task = progress_bar.add_task(
            "[yellow]Removing models...", total=len(model_names)
        )

        for name in model_names:
            progress_bar.update(task, description=f"[cyan]Removing {name}...[/cyan]")

            success = model_repo.remove_model(name)
            if success:
                console.print(f"[green]✓[/green] Removed model [bold]{name}[/bold]")
            else:
                console.print(
                    f"[yellow]![/yellow] Model [bold]{name}[/bold] not found in cache"
                )

            progress_bar.update(task, advance=1)

    console.print("[bold green]Model removal complete![/bold green]")


# Add a function to get models directly from metadata.json
def get_models() -> dict[str, dict]:
    """Get models from metadata.json"""
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    # Support both old and new metadata structure
    if "models" in metadata:
        return metadata["models"]
    elif "collections" in metadata:
        return metadata["collections"]
    else:
        raise RuntimeError("Invalid metadata structure: no models or collections found")


def main_command(
    # Input/Output
    tracks: Annotated[
        list[Path] | None,
        typer.Argument(
            help="Path to audio files or directories containing audio files",
            show_default=False,
        ),
    ] = None,
    # Model Selection
    model: Annotated[
        ModelName,
        typer.Option(
            "-m",
            "--model",
            help="Model to use for separation",
            rich_help_panel="Model Selection",
        ),
    ] = ModelName.htdemucs,
    # Processing Options
    device: Annotated[
        DeviceType,
        typer.Option(
            "-d",
            "--device",
            help="Device to process separation on",
            rich_help_panel="Processing",
        ),
    ] = (
        DeviceType.cuda
        if torch.cuda.is_available()
        else DeviceType.mps
        if torch.backends.mps.is_available()
        else DeviceType.cpu
    ),
    shifts: Annotated[
        int,
        typer.Option(
            min=1,
            max=20,
            help="Number of random shifts for equivariant stabilization, increases separation time but improves quality",
            rich_help_panel="Processing",
        ),
    ] = 1,
    split: Annotated[
        bool,
        typer.Option(
            help="Split audio in chunks to save memory",
            rich_help_panel="Processing",
        ),
    ] = True,
    split_size: Annotated[
        int | None,
        typer.Option(
            "--split-size",
            min=1,
            help="Size of each chunk in seconds, smaller values use less GPU memory but process slower",
            rich_help_panel="Processing",
        ),
    ] = None,
    split_overlap: Annotated[
        float,
        typer.Option(
            "--split-overlap",
            min=0.0,
            max=1.0,
            help="Overlap between split chunks, higher values improve quality at chunk boundaries",
            rich_help_panel="Processing",
        ),
    ] = 0.25,
    # Output
    output: Annotated[
        str,
        typer.Option(
            "-o",
            "--output",
            help="Output path template. Variables: {model}, {track}, {stem}, {ext}, {date}, {time}, {timestamp}",
            rich_help_panel="Output",
        ),
    ] = "separated/{model}/{track}/{stem}.{ext}",
    isolate_stem: Annotated[
        StemName | None,
        typer.Option(
            help="Only creates a {stem} and no_{stem} stem/file",
            rich_help_panel="Output",
        ),
    ] = None,
    clip_mode: Annotated[
        ClipMode,
        typer.Option(
            help="Strategy for avoiding clipping",
            rich_help_panel="Output",
        ),
    ] = ClipMode.rescale,
    format: Annotated[
        str,
        typer.Option(
            "-f",
            "--format",
            help="Output audio format, anything supported by FFmpeg",
            rich_help_panel="Output",
        ),
    ] = "wav",
):
    """
    Separate the sources for the given tracks
    """
    if tracks is None or not tracks:
        # Just show the help for the separate command using the current context
        import click

        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()
        return

    # Expand directory paths to audio files
    audio_files = _expand_paths_to_audio_files(tracks)

    if not audio_files:
        console.print("[red]No audio files found to process.[/red]")
        return

    if model.value == "auto":
        selected_model_name, only_load_stem = select_model(
            audio=audio_files,
            isolate_stem=isolate_stem.value if isolate_stem else None,
        )
        console.print(
            f"[cyan]Auto-selected model:[/cyan] [bold]{selected_model_name}[/bold]"
        )
    else:
        selected_model_name = model.value
        only_load_stem = isolate_stem.value if isolate_stem else None

    if not _ensure_model_available(selected_model_name):
        return

    # Create separator (with automatic optimization if isolate_stem specified)
    separator = Separator(
        model=selected_model_name,
        device=device.value,
        only_load=only_load_stem,
    )

    # Validate that the requested stem exists in the model
    if isolate_stem is not None and isolate_stem.value not in separator.model.sources:
        console.print(
            f'[red]✗[/red] [bold]{selected_model_name}[/bold]: error: stem "{isolate_stem.value}" is not in selected model. STEM must be one of {", ".join(separator.model.sources)}.'
        )
        return

    # Show where tracks will be stored
    if audio_files:
        # Resolve variables that are constant: model, ext, timestamp vars
        resolved_template = output
        resolved_template = resolved_template.replace("{model}", selected_model_name)
        resolved_template = resolved_template.replace("{ext}", format)

        now = datetime.now()
        resolved_template = resolved_template.replace(
            "{date}", now.strftime("%Y-%m-%d")
        )
        resolved_template = resolved_template.replace(
            "{time}", now.strftime("%H-%M-%S")
        )
        resolved_template = resolved_template.replace(
            "{timestamp}", str(int(now.timestamp()))
        )

        # If single file, also resolve {track}
        if len(audio_files) == 1:
            track_name = audio_files[0].name.rsplit(".", 1)[0]
            resolved_template = resolved_template.replace("{track}", track_name)
            console.print(
                f"Separated track will be stored using template '{resolved_template}'"
            )
        else:
            console.print(
                f"Separated tracks will be stored using template '{resolved_template}'"
            )

    # Use improved progress tracking for multiple files
    with FileProgressTracker(len(audio_files)) as progress_tracker:
        for track in audio_files:
            # Get just the filename for cleaner display
            filename = track.name

            # Start tracking this file
            progress_tracker.start_file(filename)

            try:
                # Create callback for audio processing progress
                audio_callback = progress_tracker.create_audio_callback(filename)

                # Use the new API with SeparatedSources and progress callback
                separated = separator.separate(
                    audio=track,
                    shifts=shifts,
                    split=split,
                    split_size=split_size,
                    split_overlap=split_overlap,
                    progress_callback=audio_callback,
                )

                # If adding complement, create the "no_{stem}" stem
                if isolate_stem is not None:
                    stem_name = isolate_stem.value
                    separated = separated.isolate_stem(stem_name)

                # Save each stem using export_stem
                for stem_name in separated.sources:
                    stem_path = format_output_path(
                        output, selected_model_name, track, stem_name, format
                    )
                    separated.export_stem(
                        stem_name,
                        stem_path,
                        format=format,
                        clip=None if clip_mode == ClipMode.none else clip_mode.value,
                    )

            except Exception as e:
                error_msg = str(e)
                progress_tracker.error_file(filename, error_msg)
                console.print(f"[red]✗[/red] Error processing {filename}: {error_msg}")


def main():
    """
    Load the checkpoints file and run the command.
    """
    app = typer.Typer(
        add_completion=False,
        no_args_is_help=True,
        rich_markup_mode="legacy",
        pretty_exceptions_show_locals=False,
    )

    models_app = typer.Typer(
        help="Download, list and manage models", no_args_is_help=True
    )
    models_app.command(name="list")(list_models_command)
    models_app.command(name="download")(download_models_command)
    models_app.command(name="remove")(remove_models_command)

    # Main commands
    app.command(name="separate")(main_command)
    app.add_typer(models_app, name="models")
    app.command(name="version")(version_command)

    # Run the app
    app()


if __name__ == "__main__":
    main()
