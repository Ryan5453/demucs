# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, Optional, Union
from io import BytesIO

import torch
import torchaudio
from torch import Tensor

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

PathLike = Path | str


class OtherMethod(str, Enum):
    add = "add"
    minus = "minus"


class SeparatedSources:
    """
    Container for storing and processing separated audio sources.
    """

    def __init__(
        self,
        sources: Dict[str, Tensor],
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
        path: Optional[PathLike] = None,
        format: str = "wav",
        clip: ClipMode = ClipMode.rescale,
        encoding: Optional[str] = None,
        bits_per_sample: Optional[int] = None,
    ) -> Union[Path, bytes]:
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
            buffer = BytesIO()
            try:
                # For WAV format, use consistent high-quality settings
                if format.lower() == "wav":
                    torchaudio.save(
                        buffer,
                        tensor,
                        sample_rate=self.sample_rate,
                        format=format,
                        encoding="PCM_F",
                        bits_per_sample=32,
                    )
                # For other formats, use provided settings or let torchaudio use defaults
                else:
                    args = [buffer, tensor, self.sample_rate, format]
                    if encoding is not None:
                        args.append(encoding)
                    if bits_per_sample is not None:
                        args.append(bits_per_sample)
                    torchaudio.save(*args)

                return buffer.getvalue()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to export stem '{stem_name}' as {format}: {e}"
                )
            finally:
                buffer.close()

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
        self, audio: Union[Tensor, PathLike, bytes], sample_rate: Optional[int] = None
    ) -> Tensor:
        """
        Convert various input types (Tensor, path, bytes) to a 2D float32 tensor
        on the configured device, matching the model's sample rate and channels
        when possible.
        """
        wav: Tensor
        input_sr: Optional[int] = None

        if isinstance(audio, Tensor):
            wav = audio
            input_sr = sample_rate
        elif isinstance(audio, (str, Path)):
            try:
                wav, sr = torchaudio.load(str(Path(audio)), backend="ffmpeg")
                input_sr = sr
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load file {audio} using FFmpeg backend: {e}. "
                    "Make sure FFmpeg is installed and the file format is supported."
                )
        elif isinstance(audio, bytes):
            audio_buffer = BytesIO(audio)
            try:
                wav, sr = torchaudio.load(audio_buffer, backend="ffmpeg")
                input_sr = sr
            except Exception as e:
                raise LoadAudioError(
                    f"Could not load audio from bytes using FFmpeg backend: {e}. "
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
        audio: Union[Tensor, PathLike, bytes],
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
        verbose: bool = False,
        sample_rate: Optional[int] = None,
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
        :param verbose: Whether to show progress bars during processing
        :param sample_rate: Sample rate of input audio (only used with tensor input)
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
            progress=verbose,
        )[0]

        sources = {}
        for source_idx, source_name in enumerate(self.model.sources):
            sources[source_name] = sources_tensor[source_idx] * std + mean

        return SeparatedSources(sources, self.sample_rate, original=wav)


def list_models() -> Dict[str, Dict[str, Any]]:
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
