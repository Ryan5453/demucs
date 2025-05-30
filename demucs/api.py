# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torchaudio
from torch import Tensor

from enum import Enum
from .apply import apply_model
from .audio import AudioFile, convert_audio, save_audio
from .pretrained import DEFAULT_MODEL, METADATA_PATH, get_model
from .repo import AnyModel, ModelRepository


class ClipMode(str, Enum):
    rescale = "rescale"
    clamp = "clamp"
    none = "none"


class OtherMethod(str, Enum):
    none = "none"
    add = "add"
    minus = "minus"


class LoadAudioError(Exception):
    """
    Exception raised when audio loading fails.
    """

    pass


class LoadModelError(Exception):
    """
    Exception raised when model loading fails.
    """

    pass


class SegmentValidationError(Exception):
    """
    Exception raised when segment parameter is invalid for the model.
    """

    pass


class _NotProvided:
    """
    A class to indicate that a parameter is not provided.
    Used to differentiate between a parameter being explicitly set to None and
    a parameter being not provided at all.
    """

    pass


NotProvided = _NotProvided()


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
        :param sample_rate: Sample rate of the audio
        :param original: Original mixed audio
        """
        self.sources = sources
        self.sample_rate = sample_rate
        self.original = original

    def __getitem__(self, key: str) -> Tensor:
        """
        Access individual stems by name.

        :param key: Name of the stem to access
        :return: Audio tensor for the stem
        """
        return self.sources[key]

    def __contains__(self, key: str) -> bool:
        """
        Check if a stem exists.

        :param key: Name of the stem to check
        :return: True if the stem exists, False otherwise
        """
        return key in self.sources

    def __iter__(self):
        """
        Iterate over stem names.

        :return: Iterator over stem names
        """
        return iter(self.sources)

    def items(self):
        """
        Iterate over (stem_name, audio_tensor) pairs.

        :return: Iterator over (stem_name, audio_tensor) pairs
        """
        return self.sources.items()

    def keys(self):
        """
        Get all stem names.

        :return: Iterator over stem names
        """
        return self.sources.keys()

    def values(self):
        """
        Get all audio tensors.

        :return: Iterator over audio tensors
        """
        return self.sources.values()

    def isolate_stem(self, name: str, method: OtherMethod) -> Dict[str, Tensor]:
        """
        Isolates a single stem and its complement.

        :param name: Name of the stem to isolate
        :param method: Method to use for isolating the stem
        :return: Dictionary containing the isolated stem and its complement (no_name and name)
        :raises ValueError: If the requested stem isn't found in the sources or method is not add or minus
        """
        if name not in self.sources:
            raise ValueError(f"Stem {name} not found in sources")

        other = None
        if method == OtherMethod.add:
            other = torch.zeros_like(self.sources[name])
            for source, audio in self.sources.items():
                if source != name:
                    other += audio
        elif method == OtherMethod.minus:
            other = self.original - self.sources[name]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'add' or 'minus'.")

        return {name: self.sources[name], f"no_{name}": other}

    def save_stem(
        self, stem_name: str, path: Union[str, Path], format: str = "wav", **kwargs
    ) -> Path:
        """
        Save a specific stem to disk. 

        :param stem_name: Name of the stem to save
        :param path: Path where the stem will be saved (without extension)
        :param format: Audio format to use (wav, mp3, flac)
        :param kwargs: Additional arguments to pass to save_audio
        :return: Path to the saved file
        :raises ValueError: If the stem name is not found
        """
        if stem_name not in self.sources:
            raise ValueError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        path = Path(path)

        # If path doesn't have an extension, add the format extension
        if not path.suffix:
            file_path = path.with_suffix(f".{format}")
        else:
            file_path = path

        # Create parent directories if they don't exist
        file_path.parent.mkdir(exist_ok=True, parents=True)

        save_audio(self.sources[stem_name], file_path, self.sample_rate, **kwargs)
        return file_path


class Separator:
    """
    Audio source separation using Demucs models.
    """

    def __init__(
        self,
        model: Union[str, AnyModel] = DEFAULT_MODEL,
        repo: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None,
        jobs: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize a Separator with the specified model and parameters.

        :param model: Model to use for separation. Can be:
                     - A string with a model name or signature (e.g., "htdemucs", "mdx_q")
                     - A pre-loaded model instance (from demucs.pretrained.get_model)
        :param repo: Folder containing pre-trained models (used only when model is a string)
        :param device: Device to use for processing ("cuda", "cpu", etc.)
        :param shifts: Number of random shifts for equivariant stabilization
                      Higher values improve quality but increase processing time
        :param overlap: Overlap between processing chunks (0.0 to 1.0)
        :param split: Whether to split the input into chunks for processing
        :param segment: Length (in seconds) of each chunk (only used if split=True)
        :param jobs: Number of parallel jobs (0 means automatic)
        :param verbose: Whether to show progress bars during processing (default False for API usage)
        """
        self._repo = repo
        self._verbose = verbose

        # Handle different model input types
        if isinstance(model, str):
            self._name = model
            self._load_model()
        else:
            self._name = getattr(model, "name", "custom_model")
            self._model = model
            self._audio_channels = model.audio_channels
            self._samplerate = model.samplerate

        # Validate segment parameter before setting other parameters
        self._validate_segment(segment)

        self.update_parameter(
            device=device,
            shifts=shifts,
            overlap=overlap,
            split=split,
            segment=segment,
            jobs=jobs,
        )

    def update_parameter(
        self,
        device: Union[str, _NotProvided] = NotProvided,
        shifts: Union[int, _NotProvided] = NotProvided,
        overlap: Union[float, _NotProvided] = NotProvided,
        split: Union[bool, _NotProvided] = NotProvided,
        segment: Optional[Union[int, _NotProvided]] = NotProvided,
        jobs: Union[int, _NotProvided] = NotProvided,
        verbose: Union[bool, _NotProvided] = NotProvided,
    ):
        """
        Update separation parameters.

        :param device: Device to use for processing
        :param shifts: Number of random shifts for equivariant stabilization
        :param overlap: Overlap between processing chunks
        :param split: Whether to split the input into chunks for processing
        :param segment: Length (in seconds) of each chunk
        :param jobs: Number of parallel jobs
        :param verbose: Whether to show progress bars during processing
        """
        if not isinstance(device, _NotProvided):
            self._device = device
        if not isinstance(shifts, _NotProvided):
            self._shifts = shifts
        if not isinstance(overlap, _NotProvided):
            self._overlap = overlap
        if not isinstance(split, _NotProvided):
            self._split = split
        if not isinstance(segment, _NotProvided):
            # Validate segment before setting it
            self._validate_segment(segment)
            self._segment = segment
        if not isinstance(jobs, _NotProvided):
            self._jobs = jobs
        if not isinstance(verbose, _NotProvided):
            self._verbose = verbose

    def _load_model(self):
        """
        Load model by name.
        """
        self._model = get_model(name=self._name, repo=self._repo)
        if self._model is None:
            raise LoadModelError("Failed to load model")
        self._audio_channels = self._model.audio_channels
        self._samplerate = self._model.samplerate

    def _get_max_allowed_segment(self) -> float:
        """
        Get the maximum allowed segment length for the current model.
        
        :return: Maximum allowed segment length in seconds
        """
        from .htdemucs import HTDemucs
        from .apply import BagOfModels
        
        if isinstance(self._model, HTDemucs):
            return float(self._model.segment)
        elif isinstance(self._model, BagOfModels):
            return self._model.max_allowed_segment
        else:
            # For other model types, no segment restriction
            return float("inf")

    def _validate_segment(self, segment: Optional[int]) -> None:
        """
        Validate that the segment parameter is compatible with the model.
        
        :param segment: Segment length in seconds to validate
        :raises SegmentValidationError: If segment is too large for the model
        """
        if segment is None:
            return
            
        max_allowed = self._get_max_allowed_segment()
        if segment > max_allowed:
            model_name = getattr(self._model, 'name', self._name)
            raise SegmentValidationError(
                f"Cannot use segment={segment} with model '{model_name}'. "
                f"Maximum allowed segment for this model is {max_allowed} seconds. "
                f"Transformer models cannot process segments longer than they were trained for."
            )

    def _load_audio(self, track: Path):
        """
        Load audio file using either ffmpeg or torchaudio.

        :param track: Path to the audio file to load
        :return: Audio tensor
        :raises LoadAudioError: If loading fails with both ffmpeg and torchaudio
        """
        errors = {}
        wav = None

        try:
            wav = AudioFile(track).read(
                streams=0,
                samplerate=self._samplerate,
                channels=self._audio_channels,
                dtype=torch.float32,
            )
            wav = wav.t()  # channels first
            return wav
        except Exception as e:
            errors["ffmpeg"] = str(e)

        try:
            wav, sr = torchaudio.load(str(track))
        except Exception as e:
            errors["torchaudio"] = str(e)
            raise LoadAudioError(
                f"Could not load file {track}. "
                "Errors from various backends:\n"
                f"ffmpeg: {errors['ffmpeg']}\n"
                f"torchaudio: {errors['torchaudio']}"
            )

        if wav.dim() == 1:
            wav = wav[None]
        if wav.dim() != 2:
            raise LoadAudioError(
                f"Expected audio tensor with 2 dimensions, got {wav.dim()}"
            )
        if sr != self._samplerate:
            wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
        return wav

    def separate_tensor(
        self, wav: Tensor, sr: Optional[int] = None
    ) -> SeparatedSources:
        """
        Separate a loaded audio tensor into stems.

        :param wav: Audio tensor of shape [channels, samples]
        :param sr: Sample rate of the input audio (if different from model's sample rate)
        :return: SeparatedSources object containing the separated stems
        :raises ValueError: If input sample rate doesn't match model's expected rate
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(self._device)

        if sr is not None and sr != self._samplerate:
            raise ValueError(
                f"Input sample rate ({sr}) doesn't match model's expected rate ({self._samplerate})"
            )

        ref = wav.mean(0)
        mean = ref.mean()
        std = ref.std()
        ref = (ref - mean) / (1e-5 + std)

        sources_tensor = apply_model(
            self._model,
            wav[None],
            device=self._device,
            shifts=self._shifts,
            split=self._split,
            overlap=self._overlap,
            segment=self._segment,
            num_workers=self._jobs,
            progress=self._verbose,
        )[0]

        # Convert tensor output to dictionary of sources
        sources = {}
        for source_idx, source_name in enumerate(self._model.sources):
            sources[source_name] = sources_tensor[source_idx] * std + mean

        return SeparatedSources(sources, self._samplerate, original=wav)

    def separate_audio_file(self, file: Union[str, Path]) -> SeparatedSources:
        """
        Separate an audio file into stems.

        :param file: Path to the audio file
        :return: SeparatedSources object containing the separated stems
        """
        if isinstance(file, str):
            file = Path(file)

        wav = self._load_audio(file)
        return self.separate_tensor(wav)

    @property
    def samplerate(self):
        """
        Get the model's sample rate.

        :return: Sample rate in Hz
        """
        return self._samplerate

    @property
    def audio_channels(self):
        """
        Get the model's audio channels.

        :return: Number of audio channels
        """
        return self._audio_channels

    @property
    def model(self):
        """
        Get the underlying model.

        :return: The model instance
        """
        return self._model

    @property
    def sources(self) -> List[str]:
        """
        Get the list of sources (stems) available in this model.

        :return: List of source names
        """
        return self._model.sources

    @property
    def max_allowed_segment(self) -> float:
        """
        Get the maximum allowed segment length for the current model.
        
        :return: Maximum allowed segment length in seconds
        """
        return self._get_max_allowed_segment()


def list_models(repo: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    List all available models and collections.

    :param repo: Optional path to a local repository
    :return: Dictionary with model signatures/names as keys and metadata as values
    """
    model_repo = ModelRepository(METADATA_PATH, repo)
    return model_repo.list_models()
