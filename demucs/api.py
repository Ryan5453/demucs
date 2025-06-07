# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from io import BytesIO

import torch
import torchaudio
from torch import Tensor

from enum import Enum
from .apply import apply_model
from .audio import ClipMode, convert_audio, save_audio, prevent_clip
from .pretrained import DEFAULT_MODEL, METADATA_PATH, get_model
from .repo import AnyModel, ModelRepository, get_cache_dir, ModelLoadingError
from . import __version__


__all__ = [
    'Separator',
    'SeparatedSources',
    'OtherMethod',
    'ClipMode',
    'ModelRepository',
    'list_models',
    'get_version',
    'get_cache_dir',
    'ModelLoadingError',
    'LoadAudioError',
    'LoadModelError',
    'SegmentValidationError',
]


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

    def add_complement_stem(self, name: str, method: OtherMethod = OtherMethod.minus) -> 'SeparatedSources':
        """
        Add the complement of a stem to this SeparatedSources object.
        This modifies the current object by adding a "no_{name}" stem.

        :param name: Name of the stem to create complement for
        :param method: Method to use for creating the complement ("add" or "minus")
        :return: Self for method chaining
        :raises ValueError: If the requested stem isn't found in the sources or method is invalid
        
        Example:
            # Add complement stem in-place
            separated.add_complement_stem("vocals")  # Adds "no_vocals" to sources
            separated.save_stem("no_vocals", "backing_track.wav")
        """
        if name not in self.sources:
            raise ValueError(f"Stem '{name}' not found in sources. Available: {list(self.sources.keys())}")

        complement_name = f"no_{name}"
        
        if method == OtherMethod.add:
            complement = torch.zeros_like(self.sources[name])
            for source, audio in self.sources.items():
                if source != name and not source.startswith("no_"):  # Don't include other "no_" stems
                    complement += audio
        elif method == OtherMethod.minus:
            complement = self.original - self.sources[name]
        else:
            raise ValueError(f"Invalid method: {method}. Use 'add' or 'minus'.")

        self.sources[complement_name] = complement
        return self

    def isolate_stem(self, name: str, method: OtherMethod = OtherMethod.minus) -> 'SeparatedSources':
        """
        Create a new SeparatedSources object containing only the specified stem and its complement.

        :param name: Name of the stem to isolate
        :param method: Method to use for creating the complement ("add" or "minus")
        :return: New SeparatedSources object with just the stem and its complement
        :raises ValueError: If the requested stem isn't found in the sources or method is invalid
        
        Example:
            # Create isolated version
            vocals_only = separated.isolate_stem("vocals")
            vocals_only.save_stem("vocals", "vocals.wav")
            vocals_only.save_stem("no_vocals", "backing.wav")
        """
        if name not in self.sources:
            raise ValueError(f"Stem '{name}' not found in sources. Available: {list(self.sources.keys())}")

        complement_name = f"no_{name}"
        
        if method == OtherMethod.add:
            complement = torch.zeros_like(self.sources[name])
            for source, audio in self.sources.items():
                if source != name and not source.startswith("no_"):  # Don't include other "no_" stems
                    complement += audio
        elif method == OtherMethod.minus:
            complement = self.original - self.sources[name]
        else:
            raise ValueError(f"Invalid method: {method}. Use 'add' or 'minus'.")

        isolated_sources = {
            name: self.sources[name],
            complement_name: complement
        }

        return SeparatedSources(isolated_sources, self.sample_rate, self.original)

    def save_stem(
        self, stem_name: str, path: Union[str, Path], **kwargs
    ) -> Path:
        """
        Save a specific stem to disk as 32-bit float WAV format (native model output).

        :param stem_name: Name of the stem to save
        :param path: Path where the stem will be saved (with or without .wav extension)
        :param kwargs: Additional arguments to pass to save_audio (e.g., clip=ClipMode.clamp)
        :return: Path to the saved file
        :raises ValueError: If the stem name is not found
        """
        if stem_name not in self.sources:
            raise ValueError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        path = Path(path)

        # If path doesn't have an extension, add .wav
        if not path.suffix:
            file_path = path.with_suffix(".wav")
        else:
            file_path = path

        # Create parent directories if they don't exist
        file_path.parent.mkdir(exist_ok=True, parents=True)

        save_audio(
            self.sources[stem_name], 
            file_path, 
            self.sample_rate, 
            **kwargs
        )
        return file_path

    def export_stem(
        self, stem_name: str, format: str = "wav", clip: ClipMode = ClipMode.rescale
    ) -> bytes:
        """
        Export a specific stem as raw audio bytes in memory.

        :param stem_name: Name of the stem to export
        :param format: Audio format ("wav", "flac", etc.)
        :param clip: Clipping mode to prevent audio distortion
        :return: Raw audio bytes ready for streaming, web APIs, etc.
        :raises ValueError: If the stem name is not found
        
        Example:
            # For web APIs
            audio_bytes = separated.export_stem("vocals", format="mp3")
            return Response(audio_bytes, mimetype="audio/mpeg")
            
            # For streaming
            wav_data = separated.export_stem("drums", format="wav")
            audio_stream.write(wav_data)
        """
        if stem_name not in self.sources:
            raise ValueError(
                f"Stem '{stem_name}' not found. Available stems: {list(self.sources.keys())}"
            )

        # Get the audio tensor and prepare it
        wav = self.sources[stem_name]
        
        # Ensure tensor is on CPU and apply clipping
        if wav.device.type != "cpu":
            wav = wav.cpu()
        
        # Apply clipping prevention
        wav = prevent_clip(wav, mode=clip)
        
        # Export to bytes using BytesIO
        buffer = BytesIO()
        try:
            # Set encoding parameters based on format
            save_kwargs = {
                "sample_rate": self.sample_rate,
                "format": format,
            }
            
            # WAV supports custom encoding and bits per sample
            if format.lower() == "wav":
                save_kwargs.update({
                    "encoding": "PCM_F",
                    "bits_per_sample": 32,
                })
            
            torchaudio.save(buffer, wav, **save_kwargs)
            return buffer.getvalue()
        except Exception as e:
            raise RuntimeError(f"Failed to export stem '{stem_name}' as {format}: {e}")
        finally:
            buffer.close()

    def export_all_stems(
        self, format: str = "wav", clip: ClipMode = ClipMode.rescale
    ) -> Dict[str, bytes]:
        """
        Export all stems as raw audio bytes in memory.

        :param format: Audio format ("wav", "flac", etc.)
        :param clip: Clipping mode to prevent audio distortion
        :return: Dictionary mapping stem names to raw audio bytes
        """
        return {
            stem_name: self.export_stem(stem_name, format=format, clip=clip)
            for stem_name in self.sources.keys()
        }

    def save_all_stems(
        self, 
        output_dir: Union[str, Path], 
        filename_template: str = "{stem_name}", 
        **kwargs
    ) -> Dict[str, Path]:
        """
        Save all stems to disk.

        :param output_dir: Directory where stems will be saved
        :param filename_template: Template for naming files. Use {stem_name} as placeholder.
        :param kwargs: Additional arguments to pass to save_audio (e.g., clip=ClipMode.clamp)
        :return: Dictionary mapping stem names to absolute file paths
        
        Example:
            # Save all stems with default naming
            paths = separated.save_all_stems("output/")
            # Result: {"vocals": "/full/path/to/output/vocals.wav", "drums": "/full/path/to/output/drums.wav", ...}
            
            # Save with custom naming
            paths = separated.save_all_stems("output/", "{stem_name}_separated")
            # Result: {"vocals": "/full/path/to/output/vocals_separated.wav", ...}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        saved_paths = {}
        for stem_name in self.sources.keys():
            filename = filename_template.format(stem_name=stem_name)
            file_path = output_dir / filename
            saved_paths[stem_name] = self.save_stem(stem_name, file_path, **kwargs).resolve()
            
        return saved_paths


class Separator:
    """
    Audio source separation using Demucs models.
    
    Note: Requires FFmpeg to be installed for audio file loading.
    Install with: conda install -c conda-forge 'ffmpeg<7'
    """

    def __init__(
        self,
        model: Union[str, AnyModel] = DEFAULT_MODEL,
        device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
        shifts: int = 1,
        overlap: float = 0.25,
        split: bool = True,
        segment: Optional[int] = None, # Default is different for each model
        jobs: int = 0,
        verbose: bool = False,
    ):
        """
        Initialize a Separator with the specified model and parameters.

        :param model: Model to use for separation. Can be:
                     - A string with a model name (e.g., "htdemucs", "mdx_q")
                     - A pre-loaded model instance (from demucs.pretrained.get_model)
        :param device: Device to use for processing ("cuda", "cpu", etc.)
        :param shifts: Number of random shifts for equivariant stabilization
                      Higher values improve quality but increase processing time
        :param overlap: Overlap between processing chunks (0.0 to 1.0)
        :param split: Whether to split the input into chunks for processing
        :param segment: Length (in seconds) of each chunk (only used if split=True)
        :param jobs: Number of parallel jobs (0 means automatic)
        :param verbose: Whether to show progress bars during processing (default False for API usage)
        """
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
        self._model = get_model(name=self._name)
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
        Load audio file using torchaudio with FFmpeg backend.

        :param track: Path to the audio file to load
        :return: Audio tensor
        :raises LoadAudioError: If loading fails
        """
        try:
            wav, sr = torchaudio.load(str(track), backend="ffmpeg")
            
            # Ensure tensor has correct dimensions
            if wav.dim() == 1:
                wav = wav[None]
            if wav.dim() != 2:
                raise LoadAudioError(
                    f"Expected audio tensor with 2 dimensions, got {wav.dim()}"
                )
            
            # Convert to target sample rate and channels if needed
            if sr != self._samplerate:
                wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
                
            return wav
            
        except Exception as e:
            raise LoadAudioError(
                f"Could not load file {track} using FFmpeg backend: {e}. "
                "Make sure FFmpeg is installed and the file format is supported."
            )

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

    def separate_audio_bytes(self, audio_bytes: bytes) -> SeparatedSources:
        """
        Separate audio from raw bytes into stems.

        :param audio_bytes: Raw audio bytes (e.g., from uploaded file, API request, etc.)
        :return: SeparatedSources object containing the separated stems
        :raises LoadAudioError: If loading fails
        
        Example:
            # For web APIs
            audio_bytes = request.files['audio'].read()
            separated = separator.separate_audio_bytes(audio_bytes)
            
            # For streaming/in-memory processing
            with open('song.mp3', 'rb') as f:
                audio_bytes = f.read()
            separated = separator.separate_audio_bytes(audio_bytes)
        """
        try:
            # Create a BytesIO object from the bytes
            audio_buffer = BytesIO(audio_bytes)
            
            # Load audio from the buffer using torchaudio with FFmpeg backend
            wav, sr = torchaudio.load(audio_buffer, backend="ffmpeg")
            
            # Ensure tensor has correct dimensions
            if wav.dim() == 1:
                wav = wav[None]
            if wav.dim() != 2:
                raise LoadAudioError(
                    f"Expected audio tensor with 2 dimensions, got {wav.dim()}"
                )
            
            # Convert to target sample rate and channels if needed
            if sr != self._samplerate:
                wav = convert_audio(wav, sr, self._samplerate, self._audio_channels)
                
            return self.separate_tensor(wav)
            
        except Exception as e:
            raise LoadAudioError(
                f"Could not load audio from bytes using FFmpeg backend: {e}. "
                "Make sure the audio format is supported."
            )
        finally:
            # Clean up the buffer
            if 'audio_buffer' in locals():
                audio_buffer.close()

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


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models.

    :return: Dictionary with model names as keys and metadata as values
    """
    model_repo = ModelRepository(METADATA_PATH)
    return model_repo.list_models()


def get_version() -> str:
    """Return the installed version of Demucs."""
    return __version__
