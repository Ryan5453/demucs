# Demucs API

The Demucs Python API is primarily comprised of two classes: `Separator` and `SeparatedSources`.

## Separator

The `Separator` class is a higher level representation of a Demucs audio source separation model. When you want to separate an audio file into its constituent stems, you will first need to create an instance of the `Separator` class which will load the model into memory for use.

```python
separator = Separator(
    model: str | AnyModel = "htdemucs", 
    device: str = "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu",
)
```

A `Separator` only takes in two parameters, the model to use for separation, which backend to use for loading and running the model. Demucs can usually auto-detect the best backend to use based on the availability of the hardware.

Once you have a `Separator` instance, you can use the `separate` method to separate an audio file into its constituent stems.

```python
def separate(
    audio: Union[Tensor, PathLike, bytes],
    shifts: int = 1,
    overlap: float = 0.25,
    split: bool = True,
    segment: Optional[int] = None,
    jobs: int = 0,
    verbose: bool = False,
    sample_rate: Optional[int] = None,
) -> SeparatedSources:
```

When separating audio, you primarily only need to care about the `audio` parameter. The `audio` parameter can be a `Tensor` matching the model expectations, a file path, or raw audio bytes.

## SeparatedSources

After running the `Separator`'s `separate` method, you will be returned a `SeparatedSources` instance. This instance contains the separated audio sources, the sample rate of the audio, and the original audio.

If you're happy with the pure audio stems, you have the ability to export them to an audio container (rather than the Tensors that are stored in the `SeparatedSources` instance). 

```python
def export_stem(
    stem_name: str,
    path: Optional[PathLike] = None, # Format extension will be added if not provided
    format: str = "wav",
    clip: ClipMode = ClipMode.rescale,
    encoding: Optional[str] = None, # Only used for WAV and FLAC
    bits_per_sample: Optional[int] = None, # Only used for WAV and FLAC
) -> Union[Path, bytes]:
```
