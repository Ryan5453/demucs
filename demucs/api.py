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

from .apply import apply_model
from .audio import convert_audio, prevent_clip
from .repo import AnyModel, ModelRepository
from .exceptions import (
    LoadAudioError,
    ModelLoadingError,
    ValidationError,
)
from . import __version__


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

    def isolate_stem(self, name: str) -> "SeparatedSources":
        """
        Isolate a stem from the separated sources.
        This creates a new SeparatedSources object with the isolated stem and the accompanying complement stem (no_{STEM})

        :param name: Name of the stem to isolate
        :return: New SeparatedSources object with the isolated stem and the accompanying complement stem
        :raises ValidationError: If the requested stem isn't found in the sources
        """
        if name not in self.sources:
            raise ValidationError(
                f"Stem '{name}' not found in sources. Available stems: {list(self.sources.keys())}"
            )

        complement = torch.zeros_like(self.sources[name])
        for source, audio in self.sources.items():
            if source != name: 
                complement += audio

        return SeparatedSources(
            sources={name: self.sources[name], f"no_{name}": complement},
            sample_rate=self.sample_rate,
            original=self.original,
        )

    def export_stem(
        self,
        stem_name: str,
        path: Path | str | None = None,
        format: str = "wav",
        clip: str | None = "rescale",
    ) -> Path | bytes:
        """
        Export a stem to either a file path or return as bytes.

        :param stem_name: Name of the stem to export
        :param path: Path to save the stem to. If None, returns raw audio bytes
        :param format: Format to export the stem to, anything supported by FFmpeg
        :param clip: Clipping mode to prevent audio distortion ("rescale", "clamp", "tanh", or None)
        :return: Path to saved file if path provided, otherwise raw audio bytes
        :raises ValidationError: If the stem name is not found
        """
        if stem_name not in self.sources:
            raise ValidationError(
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

            encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
            encoder.to_file(file_path)

            return file_path
        else:
            encoder = AudioEncoder(samples=tensor, sample_rate=self.sample_rate)
            encoded_tensor = encoder.to_tensor(format=format)
            return encoded_tensor.numpy().tobytes()


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
        only_load: str | None = None,
    ):
        """
        Initialize a Separator with the specified model and device.

        :param model: Model to use for separation (name or model instance)
        :param device: Device to use for processing (must be "cpu", "cuda", or "mps")
        :param only_load: If specified, load only the specialized model for this stem
                         (only applicable to bag-of-models like htdemucs_ft)
        :raises ValidationError: If device is not valid or only_load stem doesn't exist
        :raises ModelLoadingError: If model fails to load
        """
        # Validate device
        valid_devices = {"cpu", "cuda", "mps"}
        if device not in valid_devices:
            raise ValidationError(
                f"Invalid device '{device}'. Must be one of: {', '.join(sorted(valid_devices))}"
            )
        
        self.device = device
        
        # Handle both string model names and model instances
        if isinstance(model, str):
            model_repo = ModelRepository()
            self.model = model_repo.get_model(name=model, only_load=only_load)
        else:
            # model is already a Model instance
            self.model = model
        
        self.model.eval()
        if self.model is None:
            raise ModelLoadingError("Failed to load model")
        
        # Validate only_load stem exists in the loaded model
        if only_load and only_load not in self.model.sources:
            raise ValidationError(
                f"Stem '{only_load}' not found in model. "
                f"Available stems: {', '.join(self.model.sources)}"
            )
        
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
        split: bool = True,
        split_size: int | None = None,
        split_overlap: float = 0.25,
        sample_rate: int | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
        use_only_stem: str | None = None,
    ) -> SeparatedSources:
        """
        Separate audio into stems. Accepts tensor, file path, or raw bytes.

        :param audio: Audio input - can be:
                     - A Tensor of shape [channels, samples]
                     - A file path (str or Path)
                     - Raw audio bytes
        :param shifts: Number of random shifts for equivariant stabilization (1-20)
                      Higher values improve quality but increase processing time
        :param split_overlap: Overlap between split chunks (0.0 to 1.0)
                             Higher values improve quality at chunk boundaries
        :param split: Whether to split the input into chunks for processing
        :param split_size: Length (in seconds) of each chunk (only used if split=True)
        :param sample_rate: Sample rate of input audio (only used with tensor input)
        :param progress_callback: Optional callback for progress updates during audio processing
        :param use_only_stem: If specified and model is a ModelEnsemble, only use the model
                             specialized for this stem (performance optimization for Cog)
        :return: SeparatedSources object containing the separated stems
        :raises ValidationError: If any parameter value is invalid
        """
        # Validate shifts parameter
        if not isinstance(shifts, int) or shifts < 1 or shifts > 20:
            raise ValidationError(
                f"shifts must be an integer between 1 and 20 (inclusive), got {shifts}"
            )
        
        # Validate split_overlap parameter
        if not isinstance(split_overlap, (int, float)) or split_overlap < 0.0 or split_overlap > 1.0:
            raise ValidationError(
                f"split_overlap must be a float between 0.0 and 1.0 (inclusive), got {split_overlap}"
            )
        
        # Validate split_size parameter
        if split_size is not None:
            if not isinstance(split_size, (int, float)) or split_size < 1:
                raise ValidationError(
                    f"split_size must be a number >= 1 if provided, got {split_size}"
                )
            max_allowed = self.model.max_allowed_segment
            if split_size > max_allowed:
                model_name = getattr(self.model, "name", type(self.model).__name__)
                raise ValidationError(
                    f"Cannot use split_size={split_size} with model '{model_name}'. "
                    f"Maximum allowed split size for this model is {max_allowed} seconds. "
                    f"Transformer models cannot process segments longer than they were trained for."
                )
        
        # Validate progress_callback parameter
        if progress_callback is not None and not callable(progress_callback):
            raise ValidationError(
                f"progress_callback must be callable if provided, got {type(progress_callback)}"
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
            overlap=split_overlap,
            segment=split_size,
            progress_callback=progress_callback,
            use_only_stem=use_only_stem,
        )[0]

        sources = {}
        for source_idx, source_name in enumerate(self.model.sources):
            sources[source_name] = sources_tensor[source_idx] * std + mean

        return SeparatedSources(sources, self.sample_rate, original=wav)


def get_version() -> str:
    """
    Get the version of Demucs you have installed.

    :return: Version string
    """
    return __version__
