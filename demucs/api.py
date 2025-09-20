# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable
from io import BytesIO

import torch
from torch import Tensor
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

from enum import Enum
from .apply import apply_model
from .audio import ClipMode, convert_audio, save_audio, prevent_clip
from .pretrained import METADATA_PATH, get_model
from .repo import AnyModel, ModelRepository
from .errors import (
    LoadAudioError,
    ModelLoadingError,
    SegmentValidationError,
    InvalidStemError,
    InvalidComplementMethodError,
)
from . import __version__



class OtherMethod(str, Enum):
    add = "add"
    minus = "minus"


class SeparatedSources:
    """
    Container for storing and processing separated audio sources.
    """

    def __init__(
        self,
        sources: dict[str, Tensor],
        sample_rate: int,
        original: Tensor,
    ):
        """
        Initialize a SeparatedSources object.

        :param sources: Mapping of stem names to audio tensors
        :param sample_rate: Sample rate of the audio - comes from the model's sample rate
        :param original: Original unseparated audio
        """
        self.sources = sources
        self.sample_rate = sample_rate
        self.original = original

    def add_complement_stem(
        self, name: str, method: OtherMethod = OtherMethod.minus
    ) -> "SeparatedSources":
        """
        Add the complement of a stem to this SeparatedSources object.
        This modifies the current object by adding a "no_{name}" stem.

        :param name: Name of the stem to create complement for
        :param method: Method to use for creating the complement ("add" or "minus")
        :return: Self for method chaining
        :raises ValueError: If the requested stem isn't found in the sources or method is invalid
        """
        if name not in self.sources:
            raise InvalidStemError(
                f"Stem '{name}' not found in sources. Available: {list(self.sources.keys())}"
            )

        complement_name = f"no_{name}"

        if method == OtherMethod.add:
            complement = torch.zeros_like(self.sources[name])
            for source, audio in self.sources.items():
                if source != name and not source.startswith(
                    "no_"
                ):  # Don't include other "no_" stems
                    complement += audio
        elif method == OtherMethod.minus:
            complement = self.original - self.sources[name]
        else:
            raise InvalidComplementMethodError(
                f"Invalid method: {method}. Use 'add' or 'minus'."
            )

        self.sources[complement_name] = complement
        return self

    def export_stem(
        self,
        stem_name: str,
        path: Path | str | None = None,
        format: str = "wav",
        clip: ClipMode = ClipMode.rescale,
        encoding: str | None = None,
        bits_per_sample: int | None = None,
    ) -> Path | bytes:
        """
        Exports a stem to either a file path or a bytes object.

        :param stem_name: Name of the stem to get
        :param path: Path to save the stem to, if saving to disk
        :param format: Format to export the stem to
        :param clip: Clipping mode to prevent audio distortion (rescale or clamp)
        :param encoding: Encoding to use for the audio, only used for WAV and FLAC
        :param bits_per_sample: Bits per sample for audio encoding, only used for WAV and FLAC
        :return: Path to saved file if path provided, otherwise raw audio bytes
        :raises InvalidStemError: If the stem name is not found
        """
        if stem_name not in self.sources:
            raise InvalidStemError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        tensor = self.sources[stem_name]

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        tensor = prevent_clip(tensor, mode=clip)

        if path is not None:
            path = Path(path)

            if not path.suffix:
                file_path = path.with_suffix(f".{format}")
            else:
                file_path = path

            file_path.parent.mkdir(exist_ok=True, parents=True)

            save_audio(tensor, file_path, self.sample_rate)
            return file_path
        else:
            try:
                # Use native torchcodec AudioEncoder for BytesIO exports
                encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
                encoded_tensor = encoder.to_tensor(format=format)
                return encoded_tensor.numpy().tobytes()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to export stem '{stem_name}' as {format}: {e}"
                )


class Separator:
    """
    Audio source separation using Demucs models.
    """

    def __init__(
        self,
        model: str | AnyModel = "htdemucs",
        device: str = "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu",
    ):
        """
        Initialize a Separator with the specified model and device.

        :param model: Model to use for separation
        :param device: Device to use for processing
        """
        self.device = device
        self.model = get_model(name=model)
        if self.model is None:
            raise ModelLoadingError("Failed to load model")
        self.audio_channels = self.model.audio_channels
        self.sample_rate = self.model.samplerate

    def _to_tensor(
        self, audio: Tensor | Path | str | bytes, sample_rate: int | None = None
    ) -> Tensor:
        """
        Convert various input types (Tensor, path, bytes) to a 2D float32 tensor
        on the configured device, matching the model's sample rate and channels
        when possible.
        """
        wav: Tensor
        input_sr: int | None = None

        if isinstance(audio, Tensor):
            wav = audio
            input_sr = sample_rate
        elif isinstance(audio, (str, Path)):
            try:
                # Use native torchcodec AudioDecoder for better performance
                decoder = AudioDecoder(str(Path(audio)))
                audio_samples = decoder.get_all_samples()
                wav = audio_samples.data
                input_sr = audio_samples.sample_rate
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load file {audio} using torchcodec: {e}. "
                    "Make sure the file format is supported."
                )
        elif isinstance(audio, bytes):
            audio_buffer = BytesIO(audio)
            try:
                # Use native torchcodec AudioDecoder for better performance
                decoder = AudioDecoder(audio_buffer)
                audio_samples = decoder.get_all_samples()
                wav = audio_samples.data
                input_sr = audio_samples.sample_rate
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load audio from bytes using torchcodec: {e}. "
                    "Make sure the audio format is supported."
                )
            finally:
                audio_buffer.close()
        else:
            raise ValueError(
                f"Unsupported audio input type: {type(audio)}. "
                "Expected Tensor, file path (str/Path), or bytes."
            )

        # Minimal shape/dtype normalization
        if wav.dim() == 1:
            wav = wav[None]
        if wav.dtype != torch.float32:
            wav = wav.float()

        # Try to match expected sample rate/channels when we know input_sr, or channels mismatch
        if input_sr is not None and input_sr != self.sample_rate:
            wav = convert_audio(wav, input_sr, self.sample_rate, self.audio_channels)
        elif wav.shape[0] != self.audio_channels:
            # Adjust channels without resampling
            wav = convert_audio(
                wav, self.sample_rate, self.sample_rate, self.audio_channels
            )

        return wav.to(self.device)

    def separate(
        self,
        audio: Tensor | Path | str | bytes,
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: int | None = None,
        jobs: int = 0,
        sample_rate: int | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> SeparatedSources:
        """
        Separate audio into stems. Accepts tensor, file path, or raw bytes.

        :param audio: Audio input - can be:
                     - A Tensor of shape [channels, samples]
                     - A file path (str or Path)
                     - Raw audio bytes
        :param shifts: Number of random shifts for equivariant stabilization
                      Higher values improve quality but increase processing time
        :param overlap: Overlap between processing chunks (0.0 to 1.0)
        :param split: Whether to split the input into chunks for processing
        :param segment: Length (in seconds) of each chunk (only used if split=True)
        :param jobs: Number of parallel jobs (0 means automatic)
        :param sample_rate: Sample rate of input audio (only used with tensor input)
        :param progress_callback: Optional callback for progress updates during audio processing
        :return: SeparatedSources object containing the separated stems
        """
        # Validate segment parameter inline to reduce helpers
        if segment is not None:
            max_allowed = self.model.max_allowed_segment
            if segment > max_allowed:
                model_name = getattr(self.model, "name", type(self.model).__name__)
                raise SegmentValidationError(
                    f"Cannot use segment={segment} with model '{model_name}'. "
                    f"Maximum allowed segment for this model is {max_allowed} seconds. "
                    f"Transformer models cannot process segments longer than they were trained for."
                )

        # Normalize input to tensor
        wav = self._to_tensor(audio, sample_rate)

        # Separation logic (inlined)
        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std()
        ref = (ref - mean) / (1e-5 + std)

        sources_tensor = apply_model(
            self.model,
            wav[None],
            device=self.device,
            shifts=shifts,
            split=split,
            overlap=overlap,
            segment=segment,
            num_workers=jobs,
            progress_callback=progress_callback,
        )[0]

        sources = {}
        for source_idx, source_name in enumerate(self.model.sources):
            sources[source_name] = sources_tensor[source_idx] * std + mean

        return SeparatedSources(sources, self.sample_rate, original=wav)


def list_models() -> dict[str, dict[str, Any]]:
    """
    List all available models.

    :return: Dictionary with model names as keys and metadata as values
    """
    model_repo = ModelRepository(METADATA_PATH)
    return model_repo.list_models()


def get_version() -> str:
    """
    Get the version of Demucs you have installed.

    :return: Version string
    """
    return __version__
