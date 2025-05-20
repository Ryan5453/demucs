# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch as th
import typer
from dora.log import fatal
from typing_extensions import Annotated

from . import __version__
from .api import Separator, list_models, save_audio
from .apply import BagOfModels
from .htdemucs import HTDemucs
from .pretrained import ModelLoadingError, DEFAULT_MODEL

class ClipMode(str, Enum):
    rescale = "rescale"
    clamp = "clamp"
    none = "none"


class OtherMethod(str, Enum):
    none = "none"
    add = "add"
    minus = "minus"


def version_callback(value: bool):
    if value:
        typer.echo(f"Demucs version: {__version__}")
        raise typer.Exit()


# Use plain typer.run() to create the simplest possible CLI
# This creates a direct CLI without any command structure
def main_command(
    version: Annotated[
        bool, typer.Option("--version", callback=version_callback, help="Show the version and exit.")
    ] = False,
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
    overlap: Annotated[
        float, typer.Option(help="Overlap between the splits.")
    ] = 0.25,
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
    mp3: Annotated[
        bool, typer.Option(help="Convert the output wavs to mp3.")
    ] = False,
    mp3_bitrate: Annotated[
        int, typer.Option(help="Bitrate of converted mp3.")
    ] = 320,
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
            )
        ),
    ] = 0,
):
    """
    Separate the sources for the given tracks.
    """
    if list_models_flag:
        models = list_models(repo)
        typer.echo("Bag of models:", nl=False)
        typer.echo("\n    " + "\n    ".join(models["bag"]))
        typer.echo("Single models:", nl=False)
        typer.echo("\n    " + "\n    ".join(models["single"]))
        raise typer.Exit()

    if tracks is None:
        tracks = []
    if len(tracks) == 0:
        # Instead of error, show help
        typer.echo("Please provide one or more audio tracks to separate.")
        typer.echo("Run 'demucs --help' for usage information.")
        raise typer.Exit(0)

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
    Entry point for the CLI.
    """
    # Use environment variable to set the correct program name
    import os
    os.environ["TYPER_CLI_NAME"] = "demucs"
    typer.run(main_command)


if __name__ == "__main__":
    main()
