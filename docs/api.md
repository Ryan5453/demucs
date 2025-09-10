# Demucs API

The Demucs Python API is primarily comprised of two classes `Separator` and `SeparatedSources`.

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
    sr: Optional[int] = None,
) -> SeparatedSources:
```

The `separate` method takes in the audio file to separate, and returns a `SeparatedSources` instance.

The `audio` parameter can be a `Tensor`, a file path, or raw audio bytes.

The `shifts` parameter is the number of random shifts for equivariant stabilization. Higher values improve quality but increase processing time.

The `overlap` parameter is the overlap between processing chunks.