# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
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
from .api import Separator, save_audio, OtherMethod, ClipMode
from .apply import BagOfModels
from .pretrained import DEFAULT_MODEL, METADATA_PATH, get_model
from .repo import ModelRepository
from .errors import ModelLoadingError

console = Console()


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


def download_models_command(
    names: Annotated[
        List[str],
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
        # Get all model names
        models = get_models()
        model_names = list(models.keys())
    else:
        model_names = names

    # Create a ModelRepository to check if models are already downloaded
    model_repo = ModelRepository()
    cache_info = model_repo.get_cache_info()

    # Get models info
    models = get_models()

    # Filter out already downloaded models and count total layers to download
    to_download = []
    total_layers = 0  # Count of actual layer files to download

    for name in model_names:
        if name in cache_info:
            # Show already downloaded message
            layer_count = len(models[name]["models"])
            layer_word = "layer" if layer_count == 1 else "layers"
            console.print(
                f"[green]✓[/green] [bold]{name}[/bold]: Already downloaded ({layer_count} {layer_word}, {format_file_size(cache_info[name]['size_bytes'])})"
            )
        else:
            to_download.append(name)
            if name in models:
                total_layers += len(models[name]["models"])

    if not to_download:
        console.print("[green]All specified models are already downloaded.[/green]")
        return

    # Show appropriate message based on what we're downloading
    if len(to_download) > 1:
        console.print(
            f"[bold]Downloading {len(to_download)} models ({total_layers} total layers)...[/bold]"
        )
    else:
        # For a single item, show a simpler message
        name = to_download[0]
        if name in models:
            layer_count = len(models[name]["models"])
            layer_word = "layer" if layer_count == 1 else "layers"
            console.print(
                f"[bold]Downloading model {name} ({layer_count} {layer_word})...[/bold]"
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

                if name in models:
                    # Create a progress task for this model
                    layer_count = len(models[name]["models"])
                    layer_word = "layer" if layer_count == 1 else "layers"
                    task = progress_bar.add_task(
                        f"[cyan]Downloading {name} ({layer_count} {layer_word})[/cyan]",
                        total=100,
                        completed=0,
                    )

                    # Download the model
                    model = get_model(
                        name=name,
                        progress_bar=progress_bar,
                        task_id=task,
                    )

                    # Remove the task once complete
                    progress_bar.remove_task(task)
                else:
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
                            progress_bar=download_progress,
                            task_id=task,
                        )

                # Calculate download time
                download_time = time.time() - start_time

                # Update cache info to get the size of the downloaded model
                new_cache_info = model_repo.get_cache_info()

                # Show success message with size and speed info
                num_sources = len(model.sources)

                # Determine model info
                if name in models:
                    layer_count = len(models[name]["models"])
                    layer_word = "layer" if layer_count == 1 else "layers"
                    model_type = f"{layer_count} {layer_word}"
                else:
                    model_type = "Model"

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
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

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
def get_models() -> Dict[str, Dict]:
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
        Optional[List[Path]], typer.Argument(help="Path to tracks", show_default=False)
    ] = None,
    # Model Selection
    name: Annotated[
        str,
        typer.Option(
            "-n",
            "--name",
            help="Model name. Use 'demucs models list' to see available models.",
            rich_help_panel="Model Selection",
        ),
    ] = DEFAULT_MODEL,
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
        typer.Option(help="Overlap between the splits.", rich_help_panel="Processing"),
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
            device=device,
            shifts=shifts,
            split=split,
            overlap=overlap,
            jobs=jobs,
            segment=segment,
            verbose=True,
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
                # First separate all stems, then isolate the requested stem
                separated = separator.separate_audio_file(track)

            # Save each stem
            for stem_name, source in sources_to_save.items():
                stem_path = out_dir / filename.format(
                    track=track.name.rsplit(".", 1)[0],
                    trackext=track.name.rsplit(".", 1)[-1],
                    stem=stem_name,
                    ext="wav",
                )
                stem_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(
                    source,
                    str(stem_path),
                    samplerate=separator.samplerate,
                    clip=clip_mode,
                )

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
