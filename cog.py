# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cog import BasePredictor, File, Input, Path
from demucs import ModelRepository, Separator


class Predictor(BasePredictor):
    separators: dict[str, Separator] = {}

    def setup(self) -> None:
        repo = ModelRepository()

        for model_name in repo.list_models().keys():
            separator = Separator(
                model=model_name,
            )
            self.separators[model_name] = separator

    def predict(
        self,
        audio: Path = Input(description="The audio file to separate"),
        model: str = Input(
            description="Model to use for separation",
            default="htdemucs",
            choices=[
                "hdemucs_mmi",
                "htdemucs",
                "htdemucs_ft",
                "htdemucs_6s",
            ],
        ),
        format: str = Input(
            description="Output audio format, anything supported by FFmpeg",
            default="wav",
        ),
        isolate_stem: str = Input(
            description="Only creates a {stem} and no_{stem} stem/file",
            default=None,
            choices=[
                "drums",
                "bass",
                "other",
                "vocals",
                "guitar",
                "piano",
            ],
        ),
        shifts: int = Input(
            description="Number of random shifts for equivariant stabilization, more increases quality but increases processing time linearly",
            default=1,
            ge=1,
            le=20,
        ),
        split: bool = Input(
            description="Split audio into chunks to save memory",
            default=True,
        ),
        split_size: int | None = Input(
            description="Size of each chunk in seconds, smaller values use less GPU memory but process slower",
            default=None,
            ge=1,
        ),
        split_overlap: float = Input(
            description="Overlap between split chunks, higher values improve quality at chunk boundaries",
            default=0.25,
            ge=0.0,
            le=1.0,
        ),
        clip_mode: str | None = Input(
            description="Method to prevent audio clipping in output, or None for no clipping prevention",
            default="rescale",
            choices=["rescale", "clamp", "tanh"],
        ),
    ) -> dict[str, File]:
        separator = self.separators[model]

        if split_size is not None:
            max_segment = separator.model.max_allowed_segment
            if split_size > max_segment:
                split_size = (
                    int(max_segment) if max_segment != float("inf") else split_size
                )

        if isolate_stem is not None:
            separated = separator.separate(
                audio=audio,
                shifts=shifts,
                split=split,
                split_size=split_size,
                split_overlap=split_overlap,
                use_only_stem=isolate_stem,
            )
            separated = separated.isolate_stem(isolate_stem)
        else:
            separated = separator.separate(
                audio=audio,
                shifts=shifts,
                split=split,
                split_size=split_size,
                split_overlap=split_overlap,
            )

        return {
            stem: File(separated.export_stem(stem, format=format, clip=clip_mode))
            for stem in separated.sources
        }
