# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import torch
import typer
from rich.console import Console
from rich.progress import (
    TextColumn,
)
from rich.table import Table
from typing_extensions import Annotated

from . import __version__
from .api import Separator, save_audio, OtherMethod, ClipMode, SegmentValidationError
from .apply import BagOfModels
from .pretrained import DEFAULT_MODEL, METADATA_PATH, get_model
from .repo import ModelLoadingError, ModelRepository

console = Console()


class OutputFormat(str, Enum):
    wav16 = "wav16"  # 16-bit integer WAV (most common)
    wav24 = "wav24"  # 24-bit integer WAV (higher quality)
    wav32f = "wav32f"  # 32-bit float WAV (professional/highest quality)
    mp3 = "mp3"
    flac = "flac"


def version_command():
    """
    Show the installed version of Demucs.
    """
    typer.echo(f"Demucs version: {__version__}")


def list_models_command():
    """
    List all available models and show which ones are downloaded.
    """
    # Create a ModelRepository to manage models
    model_repo = ModelRepository(METADATA_PATH, None)

    # Get collections from metadata.json
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    collections = metadata.get("collections", {})

    # Get cache info to determine which models are downloaded
    cache_info = model_repo.get_cache_info()

    # Create a table with detailed model information
    table = Table(title="Available Demucs Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Components", style="blue")
    table.add_column("Segment", style="yellow")
    table.add_column("Size", style="magenta")
    table.add_column("Status", style="bright_green")

    # Use the order from metadata.json
    for name in collections.keys():
        info = collections[name]

        # Get model details
        model_count = len(info.get("models", []))
        segment = info.get("segment", "N/A")

        # Determine if model is downloaded and its size
        is_downloaded = name in cache_info
        model_size = "N/A"
        if is_downloaded:
            model_size = format_file_size(cache_info[name]["size_bytes"])

        table.add_row(
            name,
            "Collection",
            str(model_count) + (" model" if model_count == 1 else " models"),
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


def download_models_command(
    names: Annotated[
        List[str],
        typer.Argument(help="Pretrained model names or signatures to download."),
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
    import time

    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )

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
        # Get all collection names
        collections = get_collections()
        model_names = list(collections.keys())
    else:
        model_names = names

    # Create a ModelRepository to check if models are already downloaded
    model_repo = ModelRepository(METADATA_PATH, None)
    cache_info = model_repo.get_cache_info()

    # Get collections info to identify which models are collections
    collections = get_collections()

    # Filter out already downloaded models and count total models to download
    to_download = []
    total_models = 0  # Count of actual model files to download
    collection_count = 0  # Count of collections
    single_model_count = 0  # Count of individual models

    for name in model_names:
        if name in cache_info:
            # Check if it's a collection and show appropriate message
            if name in collections and "models" in collections[name]:
                model_count = len(collections[name]["models"])
                model_word = "model" if model_count == 1 else "models"
                console.print(
                    f"[green]✓[/green] [bold]{name}[/bold]: Already downloaded collection ({model_count} {model_word}, {format_file_size(cache_info[name]['size_bytes'])})"
                )
            else:
                console.print(
                    f"[green]✓[/green] [bold]{name}[/bold]: Already downloaded ({format_file_size(cache_info[name]['size_bytes'])})"
                )
        else:
            to_download.append(name)
            if name in collections and "models" in collections[name]:
                total_models += len(collections[name]["models"])
                collection_count += 1
            else:
                total_models += 1
                single_model_count += 1

    if not to_download:
        console.print("[green]All specified models are already downloaded.[/green]")
        return

    # Show appropriate message based on what we're downloading
    if len(to_download) > 1:
        parts = []
        if collection_count > 0:
            parts.append(
                f"{collection_count} collection{'s' if collection_count > 1 else ''}"
            )
        if single_model_count > 0:
            parts.append(
                f"{single_model_count} model{'s' if single_model_count > 1 else ''}"
            )
        what = " and ".join(parts)
        console.print(
            f"[bold]Downloading {what} ({total_models} total model files)...[/bold]"
        )
    else:
        # For a single item, show a simpler message
        name = to_download[0]
        if name in collections and "models" in collections[name]:
            model_count = len(collections[name]["models"])
            model_word = "model" if model_count == 1 else "models"
            console.print(
                f"[bold]Downloading collection {name} ({model_count} {model_word})...[/bold]"
            )
        else:
            console.print(f"[bold]Downloading model {name}...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,  # Make sure this is False to keep bars visible
        refresh_per_second=4,  # Even lower refresh rate to reduce flickering
        expand=True,  # Allow the progress bars to use the full width
    ) as progress_bar:
        for name in to_download:
            try:
                # Record start time for speed calculation
                start_time = time.time()

                if name in collections and "models" in collections[name]:
                    # For collections, create a parent task to track overall progress
                    collection = collections[name]
                    model_count = len(collection["models"])
                    model_word = "model" if model_count == 1 else "models"
                    parent_task = progress_bar.add_task(
                        f"[cyan bold]Collection Progress: 0/{model_count} {model_word} downloaded[/cyan bold]",
                        total=model_count,
                        completed=0,
                    )

                    # Download each model in the collection
                    for i, model_name in enumerate(collection["models"], 1):
                        # Update parent task description
                        progress_bar.update(
                            parent_task,
                            description=f"[cyan bold]Collection Progress: {i - 1}/{model_count} {model_word} downloaded[/cyan bold]",
                            completed=i - 1,
                        )

                        # Create a task for this specific model
                        model_task = progress_bar.add_task(
                            f"[cyan]Downloading {model_name} ({i}/{model_count})[/cyan]",
                            total=100,
                            completed=0,
                        )

                        # Download the model
                        model = get_model(
                            name=model_name,
                            repo=None,
                            progress_bar=progress_bar,
                            task_id=model_task,
                        )

                        # Remove the model task once complete
                        progress_bar.remove_task(model_task)

                    # Complete and remove the parent task
                    progress_bar.update(parent_task, completed=model_count)
                    progress_bar.remove_task(parent_task)

                else:
                    # For single models
                    console.print(f"[bold]Downloading model {name}...[/bold]")

                    # Create a new progress bar just for this model
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[bold blue]Downloading..."),
                        BarColumn(complete_style="green"),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        console=console,
                        transient=True,  # Use transient for clean display
                        refresh_per_second=10,
                    ) as download_progress:
                        # Create a single progress task for this model
                        task = download_progress.add_task("Downloading...", total=100)

                        # Download the model
                        model = get_model(
                            name=name,
                            repo=None,
                            progress_bar=download_progress,
                            task_id=task,
                        )

                # Calculate download time
                download_time = time.time() - start_time

                # Update cache info to get the size of the downloaded model
                new_cache_info = model_repo.get_cache_info()

                # Show success message with size and speed info
                num_sources = len(model.sources)

                # Determine model type based on metadata
                if name in collections and "models" in collections[name]:
                    model_count = len(collections[name]["models"])
                    model_word = "model" if model_count == 1 else "models"
                    model_type = f"Collection of {model_count} {model_word}"
                else:
                    model_type = "Single Model"

                size_str = ""
                speed_str = ""
                if name in new_cache_info:
                    size_bytes = new_cache_info[name]["size_bytes"]
                    size_str = f" ({format_file_size(size_bytes)})"

                    # Calculate download speed if download time is significant
                    if download_time > 0.1:
                        speed = size_bytes / download_time
                        speed_str = f" at {format_file_size(speed)}/s"

                console.print(
                    f"[green]✓[/green] [bold]{name}[/bold]: {model_type} with {num_sources} sources{size_str}{speed_str}"
                )

            except ModelLoadingError as error:
                console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")

            except Exception as e:
                console.print(
                    f"[red]✗[/red] [bold]{name}[/bold]: Unexpected error: {str(e)}"
                )

    console.print("[bold green]Download complete![/bold green]")


def remove_models_command(
    names: Annotated[
        List[str],
        typer.Argument(help="Pretrained model names or signatures to remove."),
    ] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Remove all downloaded models"),
    ] = False,
):
    """
    Remove models from the cache to free up space.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    # Create a ModelRepository to manage models
    model_repo = ModelRepository(METADATA_PATH, None)

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


# Add a function to get collections directly from metadata.json
def get_collections() -> Dict[str, Dict]:
    """Get collections from metadata.json"""
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return metadata.get("collections", {})


def main_command(
    # Input/Output
    tracks: Annotated[
        Optional[List[Path]], typer.Argument(help="Path to tracks", show_default=False)
    ] = None,
    # Model Selection
    name: Annotated[
        str,
        typer.Option(
            "-n",
            "--name",
            help="Model name or signature. Can be a pretrained model, a local model from --repo, or a model collection.",
            rich_help_panel="Model Selection",
        ),
    ] = DEFAULT_MODEL,
    repo: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to local model repository. Models in this folder will be available by their signature.",
            rich_help_panel="Model Selection",
        ),
    ] = None,
    # Processing Options
    device: Annotated[
        str,
        typer.Option(
            "-d",
            "--device",
            help="Device to process on, default is cuda if available, mps if available, else cpu",
            rich_help_panel="Processing",
        ),
    ] = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ),
    shifts: Annotated[
        int,
        typer.Option(
            help="Number of random shifts for equivariant stabilization. Increase separation time but improves quality.",
            rich_help_panel="Processing",
        ),
    ] = 1,
    jobs: Annotated[
        int,
        typer.Option(
            "-j",
            "--jobs",
            help="Number of jobs. Increases memory usage but will be much faster when multiple cores are available.",
            rich_help_panel="Processing",
        ),
    ] = 0,
    split: Annotated[
        bool,
        typer.Option(
            help="Split audio in chunks to save memory.",
            rich_help_panel="Processing",
        ),
    ] = True,
    segment: Annotated[
        Optional[int],
        typer.Option(
            help="Set split size of each chunk. This can help save memory of graphic card.",
            rich_help_panel="Processing",
        ),
    ] = None,
    overlap: Annotated[
        float,
        typer.Option(
            help="Overlap between the splits.", rich_help_panel="Processing"
        ),
    ] = 0.25,
    # Stem Selection
    stem: Annotated[
        Optional[str],
        typer.Option(
            "--two-stems",
            metavar="STEM",
            help="Only separate audio into {STEM} and no_{STEM}.",
            rich_help_panel="Stem Selection",
        ),
    ] = None,
    other_method: Annotated[
        OtherMethod,
        typer.Option(
            help='Decide how to get "no_{STEM}". "none" will not save "no_{STEM}". '
            '"add" will add all the other stems. "minus" will use the original track minus the selected stem.',
            rich_help_panel="Stem Selection",
        ),
    ] = OtherMethod.add,
    # Output Format
    out: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Folder where to put extracted tracks. A subfolder with the model name will be created.",
            rich_help_panel="Output",
        ),
    ] = Path("separated"),
    filename: Annotated[
        str,
        typer.Option(
            help="Set the name of output file. Use {track}, {trackext}, {stem}, {ext} "
            "to use variables of track name without extension, track extension, "
            "stem name and default output file extension.",
            rich_help_panel="Output",
        ),
    ] = "{track}/{stem}.{ext}",
    clip_mode: Annotated[
        ClipMode,
        typer.Option(
            help="Strategy for avoiding clipping: rescaling entire signal "
            "if necessary (rescale) or hard clipping (clamp).",
            rich_help_panel="Output",
        ),
    ] = ClipMode.rescale,
    format: Annotated[
        OutputFormat,
        typer.Option(
            "-f",
            "--format",
            help="Output audio format. wav16=16-bit integer WAV (standard), wav24=24-bit integer WAV (higher quality), wav32f=32-bit float WAV (professional/highest quality), mp3=320kbps MP3, flac=FLAC",
            rich_help_panel="Output",
        ),
    ] = OutputFormat.wav16,
):
    """
    Separate the sources for the given tracks.
    """
    if tracks is None or not tracks:
        typer.echo("No tracks provided.")
        typer.echo("Usage: demucs separate [options] tracks... \nHelp: demucs --help")
        return

    if name == DEFAULT_MODEL:
        console.print(f"[bold]Using default model: [cyan]{name}[/cyan][/bold]")

    try:
        separator = Separator(
            model=name,
            repo=repo,
            device=device,
            shifts=shifts,
            split=split,
            overlap=overlap,
            jobs=jobs,
            segment=segment,
            verbose=True
        )
    except Exception as error:
        console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")
        return

    if isinstance(separator.model, BagOfModels):
        console.print(
            f"Selected model is a bag of {len(separator.model.models)} models. "
            "You will see that many progress bars per track."
        )

    if stem is not None and stem not in separator.sources:
        console.print(
            f'[red]✗[/red] [bold]{name}[/bold]: error: stem "{stem}" is not in selected model. STEM must be one of {", ".join(separator.sources)}.'
        )
        return

    out_dir = out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Separated tracks will be stored in {out_dir.resolve()}")

    for track in tracks:
        if not track.exists():
            console.print(
                f"File {track} does not exist. If the path contains spaces, "
                'please try again after surrounding the entire path with quotes "".',
                err=True,
            )
            continue
        console.print(f"Separating track {track}")

        try:
            # Use the new API with SeparatedSources
            if stem is None:
                # Separate all stems
                separated = separator.separate_audio_file(track)
                sources_to_save = separated.sources
            else:
                # Isolate the requested stem
                isolated = separator.isolate_stem(track, stem)
                sources_to_save = {}
                sources_to_save[stem] = isolated[stem]
                if other_method != OtherMethod.none:
                    sources_to_save[f"no_{stem}"] = isolated[f"no_{stem}"]

            # Set up output format
            if format == OutputFormat.wav16:
                ext = "wav"
                kwargs = {
                    "samplerate": separator.samplerate,
                    "clip": clip_mode,
                    "bits_per_sample": 16,
                }
            elif format == OutputFormat.wav24:
                ext = "wav"
                kwargs = {
                    "samplerate": separator.samplerate,
                    "clip": clip_mode,
                    "bits_per_sample": 24,
                }
            elif format == OutputFormat.wav32f:
                ext = "wav"
                kwargs = {
                    "samplerate": separator.samplerate,
                    "clip": clip_mode,
                    "as_float": True,
                }
            elif format == OutputFormat.mp3:
                ext = "mp3"
                kwargs = {
                    "samplerate": separator.samplerate,
                    "clip": clip_mode,
                    "bitrate": 320,
                    "preset": 2,
                }
            elif format == OutputFormat.flac:
                ext = "flac"
                kwargs = {
                    "samplerate": separator.samplerate,
                    "clip": clip_mode,
                }

            # Save each stem
            for stem_name, source in sources_to_save.items():
                stem_path = out_dir / filename.format(
                    track=track.name.rsplit(".", 1)[0],
                    trackext=track.name.rsplit(".", 1)[-1],
                    stem=stem_name,
                    ext=ext,
                )
                stem_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(source, str(stem_path), **kwargs)

        except Exception as e:
            console.print(f"[red]✗[/red] Error processing {track}: {str(e)}")


def main():
    """
    Load the checkpoints file and run the command.
    """
    app = typer.Typer(
        help="Demucs: Audio Source Separation",
        add_completion=False,
        no_args_is_help=True,
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
