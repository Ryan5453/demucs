# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from enum import Enum
from datetime import datetime

import torch
import time
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from typing_extensions import Annotated

from . import __version__
from .api import Separator
from .repo import ModelRepository
from .exceptions import ModelLoadingError

METADATA_PATH = Path(__file__).parent / "metadata.json"


class DeviceType(str, Enum):
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


class ModelName(str, Enum):
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


def _create_audio_progress_callback(progress_bar, task):
    """Create a progress callback that updates a Rich progress bar for audio processing."""

    def callback(event_type: str, data: dict):
        if event_type == "processing_start":
            progress_bar.update(
                task,
                description=f"Processing audio ({data['total_chunks']} chunks)",
                total=data["total_chunks"],
                completed=0,
            )
        elif event_type == "chunk_complete":
            progress_bar.update(
                task,
                completed=data["completed_chunks"],
                description=f"Processing audio ({data['completed_chunks']}/{data['total_chunks']} chunks)",
            )
        elif event_type == "processing_complete":
            progress_bar.update(
                task,
                completed=data["total_chunks"],
                description="Audio processing complete",
            )

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

    # Extract track name and extension
    track_name = track.name.rsplit(".", 1)[0]
    track_ext = track.name.rsplit(".", 1)[-1] if "." in track.name else ""

    # Template variables
    variables = {
        "model": model,
        "track": track_name,
        "trackext": track_ext,
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
        list[Path] | None, typer.Argument(help="Path to tracks", show_default=False)
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
            help="Output path template. Variables: {model}, {track}, {trackext}, {stem}, {ext}, {date}, {time}, {timestamp}",
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

    # Ensure model is available (download if necessary)
    if not _ensure_model_available(model.value):
        return

    separator = Separator(
        model=model.value,
        device=device.value,
    )

    if isolate_stem is not None and isolate_stem.value not in separator.model.sources:
        console.print(
            f'[red]✗[/red] [bold]{model.value}[/bold]: error: stem "{isolate_stem.value}" is not in selected model. STEM must be one of {", ".join(separator.model.sources)}.'
        )
        return

    # Show where tracks will be stored (using first track as example)
    if tracks:
        example_path = format_output_path(
            output, model.value, tracks[0], "vocals", format
        )
        console.print(
            f"Separated tracks will be stored in {example_path.parent.resolve()}/"
        )

    for track in tracks:
        if not track.exists():
            console.print(
                f"File {track} does not exist. If the path contains spaces, "
                'please try again after surrounding the entire path with quotes "".'
            )
            continue
        console.print(f"Separating track {track}")

        try:
            # Create progress bar for audio processing
            with _create_model_progress_bar() as audio_progress:
                audio_task = audio_progress.add_task(
                    "Processing audio...",
                    total=100,
                    completed=0,
                )

                # Create callback for audio processing progress
                audio_callback = _create_audio_progress_callback(
                    audio_progress, audio_task
                )

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
                    output, model.value, track, stem_name, format
                )
                separated.export_stem(
                    stem_name, stem_path, format=format, clip=None if clip_mode == ClipMode.none else clip_mode.value
                )

        except Exception as e:
            console.print(f"[red]✗[/red] Error processing {track}: {str(e)}")


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
