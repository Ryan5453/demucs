# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cog import BasePredictor, Input, Path
from typing import Dict, Optional
import tempfile
import torch
from pathlib import Path as PathlibPath

from demucs.api import Separator, list_models
from demucs.audio import ClipMode


class Predictor(BasePredictor):
    separators: Dict[str, Separator] = {}

    def setup(self) -> None:
        available_models = list_models()

        for model_name in available_models.keys():
            separator = Separator(
                model=model_name,
                device="cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu",
            )
            self.separators[model_name] = separator

    def predict(
        self,
        audio: Path = Input(description="The audio file to separate"),
        model: str = Input(
            description="Model to use for separation",
            default="htdemucs",
            choices=[
                "mdx",
                "mdx_extra",
                "mdx_q",
                "mdx_extra_q",
                "hdemucs_mmi",
                "htdemucs",
                "htdemucs_ft",
                "htdemucs_6s",
            ],
        ),
        output_format: str = Input(
            description="Output audio format",
            default="wav",
            choices=["wav", "flac", "mp3"],
        ),
        # Core separation parameters
        shifts: int = Input(
            description="Number of random shifts for equivariant stabilization (higher = better quality, slower)",
            default=1,
            ge=1,
            le=10,
        ),
        overlap: float = Input(
            description="Overlap between processing chunks (0.0 to 1.0). Higher values improve quality but increase processing time",
            default=0.25,
            ge=0.0,
            le=1.0,
        ),
        split: bool = Input(
            description="Whether to split the input into chunks for processing. Helps with memory usage for long files",
            default=True,
        ),
        segment: Optional[int] = Input(
            description="Length (in seconds) of each chunk when split=True. Leave empty for model default. Some models have max limits",
            default=None,
            ge=1,
            le=3600,  # Max 1 hour per segment
        ),
        jobs: int = Input(
            description="Number of parallel jobs for CPU processing (0 = automatic). Only affects CPU portions of processing",
            default=0,
            ge=0,
            le=16,
        ),
        # Audio processing parameters
        clip_mode: str = Input(
            description="Method to prevent audio clipping in output",
            default="rescale",
            choices=["rescale", "clamp", "tanh", "none"],
        ),
        # Stem selection
        stems: str = Input(
            description="Which stems to return. Examples: 'all' (all stems), 'vocals,drums' (specific stems), 'no_vocals' (instrumental), 'vocals,no_vocals' (both vocal and instrumental)",
            default="all",
        ),
        verbose: bool = Input(
            description="Show detailed processing information and progress",
            default=False,
        ),
    ) -> Dict[str, PathlibPath]:
        """
        Separate audio into stems using the selected Demucs model.

        This endpoint provides full control over all separation parameters available in the Demucs API.

        Stem Selection Examples:
        - 'all': Returns all available stems (drums, bass, vocals, other)
        - 'vocals': Returns only the vocals stem
        - 'vocals,drums': Returns vocals and drums stems
        - 'no_vocals': Returns instrumental track (everything except vocals)
        - 'vocals,no_vocals': Returns both vocals and instrumental
        - 'no_drums,no_bass': Returns track without drums and without bass
        """

        # Validate model selection
        if model not in self.separators:
            raise ValueError(
                f"Model {model} not available. Available models: {list(self.separators.keys())}"
            )

        # Get the pre-loaded separator
        separator = self.separators[model]

        # Validate segment parameter against model limits
        if segment is not None:
            max_segment = separator.model.max_allowed_segment
            if segment > max_segment:
                print(
                    f"⚠️  Warning: Segment {segment}s exceeds model limit {max_segment}s, using model maximum"
                )
                segment = int(max_segment) if max_segment != float("inf") else segment

        print(f"🎵 Separating audio using {model}...")
        if verbose:
            print(
                f"Parameters: shifts={shifts}, overlap={overlap}, split={split}, segment={segment}, jobs={jobs}"
            )
            print(f"🖥️  {self._get_gpu_memory_info()}")

        # Perform separation using the new unified API
        separated = separator.separate(
            audio=audio,
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment,
            jobs=jobs,
            progress_callback=None,
        )

        # Parse stems parameter
        stems_requested = []
        if stems.lower() == "all":
            # Return all available stems from the model
            stems_requested = list(separated.keys())
        else:
            # Parse comma-separated list
            stems_requested = [s.strip() for s in stems.split(",")]

        # Validate and process requested stems
        available_stems = list(separated.keys())
        complement_stems_needed = []

        for stem in stems_requested:
            if stem.startswith("no_"):
                # This is a complement stem request
                base_stem = stem[3:]  # Remove "no_" prefix
                if base_stem in available_stems:
                    complement_stems_needed.append(base_stem)
                else:
                    print(
                        f"⚠️  Warning: Cannot create '{stem}' - base stem '{base_stem}' not available in model"
                    )
            elif stem not in available_stems:
                print(
                    f"⚠️  Warning: Stem '{stem}' not available in model. Available: {', '.join(available_stems)}"
                )

        # Add complement stems to the separated sources
        for base_stem in complement_stems_needed:
            separated.add_complement_stem(base_stem)

        # Update stems_requested to only include valid stems
        valid_stems = []
        for stem in stems_requested:
            if stem in separated.keys():
                valid_stems.append(stem)

        if verbose:
            print(f"Requested stems: {stems_requested}")
            print(f"Valid stems to output: {valid_stems}")

        # Create temporary directory for outputs
        temp_dir = PathlibPath(tempfile.mkdtemp())

        # Convert clip_mode string to enum
        clip_mode_enum = ClipMode(clip_mode)

        # Save requested stems
        output_paths = {}

        for stem_name in valid_stems:
            output_filename = f"{stem_name}.{output_format}"
            output_path = temp_dir / output_filename

            if output_format == "wav":
                separated.save_stem(stem_name, output_path, clip=clip_mode_enum)
            else:
                audio_bytes = separated.export_stem(
                    stem_name, format=output_format, clip=clip_mode_enum
                )
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

            output_paths[stem_name] = output_path

        if verbose:
            print(f"🖥️  {self._get_gpu_memory_info()}")

        print(f"✅ Separation complete! Generated {len(output_paths)} files")
        return output_paths

    def _get_gpu_memory_info(self) -> str:
        """Get GPU memory usage information."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # Convert to GB
            return f"GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved"
        elif torch.backends.mps.is_available():
            return "Using MPS (Metal Performance Shaders)"
        else:
            return "Using CPU"
