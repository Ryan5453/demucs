import tempfile
from io import BytesIO
from typing import Optional

import torch
from cog import BasePredictor, Input, Path
from torch.cuda import is_available as is_cuda_available

from demucs.api import Separator
from demucs.apply import BagOfModels
from demucs.audio import save_audio
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model

# The demucs API does have a method to get all models but it
# returns models we don't want so it's easier to manually curate
DEMUCS_MODELS = [
    # Demucs v4
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    # Demucs v3
    "hdemucs_mmi",
    # Demucs v2
    # I'm not including the non-quantized versions because
    # according to the author, there is no quality degradation
    # so this should just help speed up boot times
    "mdx_q",
    "mdx_extra_q",
]


class PreloadedSeparator(Separator):
    """
    For efficiency, this cog keeps the models in memory
    so that they don't need to be loaded for every single request.

    The Separator API only supports loading models by name, so
    we have to subclass it and load the model manually.
    """

    def __init__(
        self,
        model: BagOfModels,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
    ):
        self._model = model
        self._audio_channels = model.audio_channels
        self._samplerate = model.samplerate

        self.update_parameter(
            device="cuda" if is_cuda_available() else "cpu",
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment,
            jobs=jobs,
            progress=True,
            callback=None,
            callback_arg=None,
        )


class Predictor(BasePredictor):
    """
    This cog implements the Cog API to inference Demucs models.
    """

    def setup(self):
        """
        Loading the models into memory will provide faster prediction
        when multiple requests are made in succession.
        """
        self.models = {model: get_model(model) for model in DEMUCS_MODELS}

    def predict(
        self,
        audio: Path = Input(description="Upload the file to be processed here."),
        model: str = Input(
            default="htdemucs",
            description="Choose the demucs audio that proccesses your audio. The readme has more information on what to choose.",
            choices=DEMUCS_MODELS,
        ),
        stem: str = Input(
            default="none",
            description="If you just want to isolate one stem, you can choose it here.",
            choices=["none", "drums", "bass", "other", "vocals", "guitar", "piano"],
        ),
        # Audio Options
        output_format: str = Input(
            default="mp3",
            description="Choose the audio format you would like the result to be returned in.",
            choices=["mp3", "flac", "wav"],
        ),
        mp3_bitrate: int = Input(
            default=320,
            description="Choose the bitrate for the MP3 output. Higher is better quality but larger file size. If MP3 is not selected as the output type, this has no effect.",
        ),
        mp3_preset: int = Input(
            default=2,
            choices=range(2, 8),
            description="Choose the preset for the MP3 output. Higher is faster but worse quality. If MP3 is not selected as the output type, this has no effect.",
        ),
        wav_format: str = Input(
            default="int24",
            choices=["int16", "int24", "float32"],
            description="Choose format for the WAV output. If WAV is not selected as the output type, this has no effect.",
        ),
        clip_mode: str = Input(
            default="rescale",
            choices=["rescale", "clamp", "none"],
            description="Choose the strategy for avoiding clipping. Rescale will rescale entire signal if necessary or clamp will allow hard clipping.",
        ),
        # Separator Options
        shifts: int = Input(
            default=1,
            description="Choose the amount random shifts for equivariant stabilization. This performs multiple predictions with random shifts of the input and averages them, which makes it x times slower.",
        ),
        overlap: float = Input(
            default=0.25,
            description="Choose the amount of overlap between prediction windows.",
        ),
        split: bool = Input(
            default=True,
            description="Choose whether or not the audio should be split into chunks.",
        ),
        segment: int = Input(
            default=None,
            description="Choose the segment length to use for separation.",
        ),
        jobs: int = Input(
            default=0,
            description="Choose the number of parallel jobs to use for separation.",
        ),
    ) -> dict:
        # Use preloaded model
        model = self.models[model]

        if stem != "none" and stem not in model.sources:
            raise ValueError(
                f"Selected stem '{stem}' is not supported by chosen model."
            )

        max_allowed_segment = float("inf")
        if isinstance(model, HTDemucs):
            max_allowed_segment = float(model.segment)
        elif isinstance(model, BagOfModels):
            max_allowed_segment = model.max_allowed_segment

        if segment is not None and segment > max_allowed_segment:
            raise ValueError(
                f"Cannot use a Transformer model with a longer segment than it was trained for. Maximum allowed segment is {max_allowed_segment}."
            )

        separator = PreloadedSeparator(
            model=model,
            shifts=shifts,
            overlap=overlap,
            segment=segment,
            split=split,
            jobs=jobs,
        )

        _, outputs = separator.separate_audio_file(audio)

        kwargs = {
            "samplerate": separator.samplerate,
            "bitrate": mp3_bitrate,
            "preset": mp3_preset,
            "clip": clip_mode,
            "as_float": wav_format == "float32",
            "bits_per_sample": 24 if wav_format == "int24" else 16,
        }

        output_stems = {}

        if stem == "none":
            for name, source in outputs.items():
                with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                    save_audio(source.cpu(), f.name, **kwargs)
                    output_stems[name] = BytesIO(open(f.name, "rb").read())
        else:
            with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                save_audio(outputs[stem].cpu(), f.name, **kwargs)
                output_stems[stem] = BytesIO(open(f.name, "rb").read())

            other_stem = torch.zeros_like(outputs[stem])
            for source, audio in outputs.items():
                if source != stem:
                    other_stem += audio

            with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                save_audio(other_stem.cpu(), f.name, **kwargs)
                output_stems["no_" + stem] = BytesIO(open(f.name, "rb").read())

        return output_stems
