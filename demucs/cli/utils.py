# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

console = Console()

METADATA_PATH = Path(__file__).parent.parent / "metadata.json"


def format_file_size(size_bytes: int) -> str:
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


def expand_paths_to_audio_files(paths: list[Path]) -> list[Path]:
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
