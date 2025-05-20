# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import torch as th
import typer
from dora.log import fatal
from rich.console import Console
from typing_extensions import Annotated

from . import __version__
from .api import Separator, save_audio
from .apply import BagOfModels
from .htdemucs import HTDemucs
from .pretrained import DEFAULT_MODEL, METADATA_PATH, ModelLoadingError, get_model
from .repo import ModelLoadingError

console = Console()


class ClipMode(str, Enum):
    rescale = "rescale"
    clamp = "clamp"
    none = "none"


class OtherMethod(str, Enum):
    none = "none"
    add = "add"
    minus = "minus"


def version_command():
    """
    Show the installed version of Demucs.
    """
    typer.echo(f"Demucs version: {__version__}")


def list_models_command():
    """
    List all available models.
    """
    # Get collections directly from metadata.json
    collections = get_collections()
    console.print("[bold]Available models:[/bold]")

    # Display information about each collection
    for name in sorted(collections.keys()):
        info = collections[name]
        if "models" in info:
            model_count = len(info["models"])
            segment = info.get("segment", "default")
            description = f"Bag of {model_count} models"
            if "segment" in info:
                description += f", segment={segment}"
        else:
            description = "Model collection"
        console.print(f"  - [cyan]{name}[/cyan]: {description}")


def download_models_command(
    names: Annotated[
        List[str],
        typer.Argument(help="Pretrained model names or signatures to download."),
    ] = None,
    repo: Annotated[
        Optional[Path],
        typer.Option(help="Folder containing all pre-trained models for use with -n."),
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
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    if all_models:
        # Get all collection names
        collections = get_collections()
        model_names = list(collections.keys())
    else:
        if names is None or not names:
            console.print("[yellow]No models specified. Using default model.[/yellow]")
            model_names = [DEFAULT_MODEL]
        else:
            model_names = names

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        refresh_per_second=10,
    ) as progress_bar:
        task = progress_bar.add_task(
            "[yellow]Downloading models...", total=len(model_names)
        )

        for name in model_names:
            progress_bar.update(task, description=f"[cyan]Downloading {name}...[/cyan]")
            try:
                model = get_model(name=name, repo=repo)
                num_sources = len(model.sources)
                model_type = (
                    f"Bag of {len(model.models)} models"
                    if isinstance(model, BagOfModels)
                    else "Single Model"
                )
                progress_bar.update(task, advance=1)
                console.print(
                    f"[green]✓[/green] [bold]{name}[/bold]: {model_type} with {num_sources} sources"
                )
            except ModelLoadingError as error:
                progress_bar.update(task, advance=1)
                console.print(f"[red]✗[/red] [bold]{name}[/bold]: {error}")

    console.print("[bold green]Download complete![/bold green]")


# Add a function to get collections directly from metadata.json
def get_collections() -> Dict[str, Dict]:
    """Get collections from metadata.json"""
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return metadata.get("collections", {})


# Use plain typer.run() to create the simplest possible CLI
# This creates a direct CLI without any command structure
def main_command(
    tracks: Annotated[
        Optional[List[Path]], typer.Argument(help="Path to tracks")
    ] = None,
    sig: Annotated[
        Optional[str], typer.Option("-s", "--sig", help="Locally trained XP signature.")
    ] = None,
    name: Annotated[
        str,
        typer.Option(
            "-n",
            "--name",
            help="Pretrained model name or signature. Default is htdemucs.",
        ),
    ] = DEFAULT_MODEL,
    repo: Annotated[
        Optional[Path],
        typer.Option(help="Folder containing all pre-trained models for use with -n."),
    ] = None,
    list_models_flag: Annotated[
        bool,
        typer.Option(
            "--list-models", help="List available models from current repo and exit"
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Enable verbose output")
    ] = False,
    out: Annotated[
        Path,
        typer.Option(
            "-o",
            "--out",
            help="Folder where to put extracted tracks. A subfolder with the model name will be created.",
        ),
    ] = Path("separated"),
    filename: Annotated[
        str,
        typer.Option(
            help=(
                "Set the name of output file. Use {track}, {trackext}, {stem}, {ext} "
                "to use variables of track name without extension, track extension, "
                "stem name and default output file extension."
            )
        ),
    ] = "{track}/{stem}.{ext}",
    device: Annotated[
        str,
        typer.Option(
            "-d",
            "--device",
            help="Device to use, default is cuda if available else cpu",
        ),
    ] = (
        "cuda"
        if th.cuda.is_available()
        else "mps"
        if th.backends.mps.is_available()
        else "cpu"
    ),
    shifts: Annotated[
        int,
        typer.Option(
            help=(
                "Number of random shifts for equivariant stabilization. "
                "Increase separation time but improves quality for Demucs. "
                "10 was used in the original paper."
            )
        ),
    ] = 1,
    overlap: Annotated[float, typer.Option(help="Overlap between the splits.")] = 0.25,
    no_split: Annotated[
        bool,
        typer.Option(
            help="Doesn't split audio in chunks. This can use large amounts of memory."
        ),
    ] = False,
    segment: Annotated[
        Optional[int],
        typer.Option(
            help="Set split size of each chunk. This can help save memory of graphic card."
        ),
    ] = None,
    stem: Annotated[
        Optional[str],
        typer.Option(
            "--two-stems",
            metavar="STEM",
            help="Only separate audio into {STEM} and no_{STEM}.",
        ),
    ] = None,
    other_method: Annotated[
        OtherMethod,
        typer.Option(
            help=(
                'Decide how to get "no_{STEM}". "none" will not save '
                '"no_{STEM}". "add" will add all the other stems. "minus" will use the '
                "original track minus the selected stem."
            )
        ),
    ] = OtherMethod.add,
    int24: Annotated[
        bool, typer.Option(help="Save wav output as 24 bits wav.")
    ] = False,
    float32: Annotated[
        bool, typer.Option(help="Save wav output as float32 (2x bigger).")
    ] = False,
    clip_mode: Annotated[
        ClipMode,
        typer.Option(
            help=(
                "Strategy for avoiding clipping: rescaling entire signal "
                "if necessary (rescale) or hard clipping (clamp)."
            )
        ),
    ] = ClipMode.rescale,
    flac: Annotated[
        bool, typer.Option(help="Convert the output wavs to flac.")
    ] = False,
    mp3: Annotated[bool, typer.Option(help="Convert the output wavs to mp3.")] = False,
    mp3_bitrate: Annotated[int, typer.Option(help="Bitrate of converted mp3.")] = 320,
    mp3_preset: Annotated[
        int,
        typer.Option(
            help=(
                "Encoder preset of MP3, 2 for highest quality, 7 for "
                "fastest speed. Default is 2"
            ),
            min=2,
            max=7,
        ),
    ] = 2,
    jobs: Annotated[
        int,
        typer.Option(
            "-j",
            "--jobs",
            help=(
                "Number of jobs. This can increase memory usage but will "
                "be much faster when multiple cores are available."
            ),
        ),
    ] = 0,
):
    """
    Separate the sources for the given tracks.
    """
    if list_models_flag:
        collections = get_collections()
        typer.echo("Available models:")
        for name in sorted(collections.keys()):
            typer.echo(f"  {name}")
        return

    # Display a helpful message about downloading models
    if name != DEFAULT_MODEL:
        console.print(f"[bold]Using model: [cyan]{name}[/cyan][/bold]")
        console.print(
            f"[dim]To pre-download this model, run: demucs models download {name}[/dim]"
        )

    if tracks is None or not tracks:
        typer.echo("No tracks provided.")
        typer.echo("Usage: demucs separate [options] tracks... \nHelp: demucs --help")
        return

    split = not no_split

    try:
        separator = Separator(
            model=name if sig is None else sig,
            repo=repo,
            device=device,
            shifts=shifts,
            split=split,
            overlap=overlap,
            progress=True,
            jobs=jobs,
            segment=segment,
        )
    except ModelLoadingError as error:
        fatal(error.args[0])

    max_allowed_segment = float("inf")
    if isinstance(separator.model, HTDemucs):
        max_allowed_segment = float(separator.model.segment)
    elif isinstance(separator.model, BagOfModels):
        max_allowed_segment = separator.model.max_allowed_segment
    if segment is not None and segment > max_allowed_segment:
        fatal(
            "Cannot use a Transformer model with a longer segment "
            f"than it was trained for. Maximum segment is: {max_allowed_segment}"
        )

    if isinstance(separator.model, BagOfModels):
        typer.echo(
            f"Selected model is a bag of {len(separator.model.models)} models. "
            "You will see that many progress bars per track."
        )

    if stem is not None and stem not in separator.model.sources:
        fatal(
            'error: stem "{stem}" is not in selected model. '
            "STEM must be one of {sources}.".format(
                stem=stem, sources=", ".join(separator.model.sources)
            )
        )
    out_dir = out / (name if sig is None else sig)
    out_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Separated tracks will be stored in {out_dir.resolve()}")
    for track in tracks:
        if not track.exists():
            typer.echo(
                f"File {track} does not exist. If the path contains spaces, "
                'please try again after surrounding the entire path with quotes "".',
                err=True,
            )
            continue
        typer.echo(f"Separating track {track}")

        origin, res = separator.separate_audio_file(track)

        if mp3:
            ext = "mp3"
        elif flac:
            ext = "flac"
        else:
            ext = "wav"
        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": mp3_bitrate,
            "preset": mp3_preset,
            "clip": clip_mode,
            "as_float": float32,
            "bits_per_sample": 24 if int24 else 16,
        }
        if stem is None:
            for name, source in res.items():
                stem_path = out_dir / filename.format(
                    track=track.name.rsplit(".", 1)[0],
                    trackext=track.name.rsplit(".", 1)[-1],
                    stem=name,
                    ext=ext,
                )
                stem_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(source, str(stem_path), **kwargs)
        else:
            stem_path = out_dir / filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem="minus_" + stem,
                ext=ext,
            )
            if other_method == OtherMethod.minus:
                stem_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(origin - res[stem], str(stem_path), **kwargs)
            stem_path = out_dir / filename.format(
                track=track.name.rsplit(".", 1)[0],
                trackext=track.name.rsplit(".", 1)[-1],
                stem=stem,
                ext=ext,
            )
            stem_path.parent.mkdir(parents=True, exist_ok=True)
            save_audio(res.pop(stem), str(stem_path), **kwargs)
            # Warning : after poping the stem, selected stem is no longer in the dict 'res'
            if other_method == OtherMethod.add:
                other_stem = th.zeros_like(next(iter(res.values())))
                for i in res.values():
                    other_stem += i
                stem_path = out_dir / filename.format(
                    track=track.name.rsplit(".", 1)[0],
                    trackext=track.name.rsplit(".", 1)[-1],
                    stem="no_" + stem,
                    ext=ext,
                )
                stem_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(other_stem, str(stem_path), **kwargs)


def main():
    """
    Load the checkpoints file and run the command.
    """
    app = typer.Typer(
        help="Demucs: Music Source Separation",
        add_completion=False,
        no_args_is_help=True,  # Show help when no arguments are provided
    )
    # Create models command group
    models_app = typer.Typer(
        help="Download, list and manage models", no_args_is_help=True
    )
    models_app.command(name="list")(list_models_command)
    models_app.command(name="download")(download_models_command)

    # Main commands
    app.command(name="separate")(main_command)
    app.add_typer(models_app, name="models")
    app.command(name="version")(version_command)

    # Create a callback for the main command to show helpful info
    @app.callback()
    def callback():
        """
        Demucs: Music Source Separation tool

        USAGE:
          demucs separate [OPTIONS] TRACKS...    - Separate audio tracks
          demucs models list                     - List available models
          demucs models download [OPTIONS] [NAMES]... - Download model(s)
          demucs version                         - Show version information
        """
        pass

    # Run the app
    app()


if __name__ == "__main__":
    main()
