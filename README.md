# Demucs

> [!NOTE] 
> This is a mantained fork of the [author's fork](https://github.com/adefossez/demucs) of the [original](https://github.com/facebookresearch/demucs) Demucs repository. It has been modified to be a inference-only package.

Demucs is a state-of-the-art music source separation model, currently capable of separating drums, bass, and vocals from the rest of the accompaniment. Demucs is based on a U-Net convolutional architecture inspired by [Wave-U-Net][waveunet]. 

Samples are available [online][samples] for both Hybrid Demucs and Hybrid Transformer Demucs. Checkout [the paper][htdemucs] for more information.

## Installation

### Prerequisites

Before installing Demucs, you need:

- Python 3.8 or later
- FFmpeg (required for audio processing):
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg` 
  - Windows: Download from [FFmpeg.org](https://ffmpeg.org/download.html)

### For Command Line Users

If you want to use Demucs primarily as a command-line tool to separate audio tracks:

#### Install UV (Recommended)

UV is a fast, modern Python package manager with isolated environments:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Installation Options

1. **UV Tool Install** (Recommended) - Creates an isolated environment:
   ```bash
   uv tool install demucs-inference
   ```

2. **Run Without Installing** - For one-time or occasional use:
   ```bash
   # Use uvx to run without permanent installation
   uvx --with demucs-inference demucs audio_file.mp3
   ```

3. **Standard pip Install**:
   ```bash
   pip install demucs-inference
   ```

### For Python Developers

If you want to use Demucs as a library in your Python applications:

```bash
# Using UV (recommended)
uv pip install demucs-inference
# or shorter syntax
uv add demucs-inference

# Using standard pip
pip install demucs-inference
```

### Upgrading

```bash
# If installed with uv tool
uv tool upgrade demucs-inference

# If installed with standard uv
uv pip install -U demucs-inference
# or shorter syntax
uv add -U demucs-inference

# If installed with pip
pip install -U demucs-inference
```

### GPU Support

For GPU acceleration (strongly recommended for faster processing):

1. Install PyTorch with CUDA support from the [PyTorch installation page](https://pytorch.org/get-started/locally/)
2. Ensure you have compatible NVIDIA drivers installed
3. At least 3GB of GPU VRAM is required (7GB recommended for default settings)

## Usage

After installing Demucs, you can use it like the following:

```bash
# View all options
demucs --help

# Separate one audio file
demucs audio_file.mp3

# Separate multiple audio files
demucs audio_file_1.mp3 audio_file_2.mp3

# Separate all audio files in the current directory
demucs *.mp3
```

### Notes

- If you have a GPU, but you run out of memory, please use the `--segment` option to reduce length of each split. It should be set to a integer describing the length of each segment in seconds.
- A segment length of at least 10 is recommended (the bigger the number is, the more memory is required, but quality may increase). Note that the Hybrid Transformer models only support a maximum segment length of 7.8 seconds.
- Creating an environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` is also helpful. If this still does not help, please add `-d cpu` to the command line. See the section hereafter for more details on the memory requirements for GPU acceleration.

Separated tracks are stored in the `separated/MODEL_NAME/TRACK_NAME` folder. There you will find four stereo wav files sampled at 44.1 kHz: `drums.wav`, `bass.wav`,
`other.wav`, `vocals.wav` (or `.mp3` if you used the `--mp3` option).

All audio formats supported by `torchaudio` can be processed (i.e. wav, mp3, flac, ogg/vorbis on Linux/macOS, etc.). On Windows, `torchaudio` has limited support, so we rely on `ffmpeg`, which should support pretty much anything.
Audio is resampled on the fly if necessary.
The output will be a wav file encoded as int16.
You can save as float32 wav files with `--float32`, or 24 bits integer wav with `--int24`.
You can pass `--mp3` to save as mp3 instead, and set the bitrate (in kbps) with `--mp3-bitrate` (default is 320).

It can happen that the output would need clipping, in particular due to some separation artifacts.
Demucs will automatically rescale each output stem so as to avoid clipping. This can however break
the relative volume between stems. If instead you prefer hard clipping, pass `--clip-mode clamp`.
You can also try to reduce the volume of the input mixture before feeding it to Demucs.


Other pre-trained models can be selected with the `-n` flag.
The list of pre-trained models is:
- `htdemucs`: first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
- `htdemucs_ft`: fine-tuned version of `htdemucs`, separation will take 4 times more time
    but might be a bit better. Same training set as `htdemucs`.
- `htdemucs_6s`: 6 sources version of `htdemucs`, with `piano` and `guitar` being added as sources.
    Note that the `piano` source is not working great at the moment.
- `hdemucs_mmi`: Hybrid Demucs v3, retrained on MusDB + 800 songs.
- `mdx`: trained only on MusDB HQ, winning model on track A at the [MDX][mdx] challenge.
- `mdx_extra`: trained with extra training data (**including MusDB test set**), ranked 2nd on the track B
    of the [MDX][mdx] challenge.
- `mdx_q`, `mdx_extra_q`: quantized version of the previous models. Smaller download and storage
    but quality can be slightly worse.

The `--two-stems=vocals` option allows separating vocals from the rest of the accompaniment (i.e., karaoke mode).
`vocals` can be changed to any source in the selected model.
This will mix the files after separating the mix fully, so this won't be faster or use less memory.

The `--shifts=SHIFTS` performs multiple predictions with random shifts (a.k.a the *shift trick*) of the input and average them. This makes prediction `SHIFTS` times
slower. Don't use it unless you have a GPU.

The `--overlap` option controls the amount of overlap between prediction windows. Default is 0.25 (i.e. 25%) which is probably fine.
It can probably be reduced to 0.1 to improve a bit speed.

The `-j` flag allow to specify a number of parallel jobs (e.g. `demucs -j 2 myfile.mp3`).
This will multiply by the same amount the RAM used so be careful!

### Memory requirements for GPU acceleration

If you want to use GPU acceleration, you will need at least 3GB of RAM on your GPU for `demucs`. However, about 7GB of RAM will be required if you use the default arguments. Add `--segment SEGMENT` to change size of each split. If you only have 3GB memory, set SEGMENT to 8 (though quality may be worse if this argument is too small). Creating an environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` can help users with even smaller RAM such as 2GB (I separated a track that is 4 minutes but only 1.5GB is used), but this would make the separation slower.

If you do not have enough memory on your GPU, simply add `-d cpu` to the command line to use the CPU. With Demucs, processing time should be roughly equal to 1.5 times the duration of the track.

## Demucs API

Demucs provides two APIs that can be used to separate audio files.

### Command Line Interface API

A simple API that provides a interface similar to the command line interface. 
```python
# Assume that your command is `demucs --mp3 --two-stems vocals -n mdx_extra "audio_file.mp3"`
import demucs.separate
demucs.separate.main(["--mp3", "--two-stems", "vocals", "-n", "mdx_extra", "audio_file.mp3"])
```

### Separator API

A more complicated API that provides a `Separator` class that can be used to separate audio files.

View the [API docs](docs/api.md) for more information on the `Separator` class.

## How to cite

```
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

## License

Demucs is released under the MIT license as found in the [LICENSE](LICENSE) file.

[waveunet]: https://github.com/f90/Wave-U-Net
[mdx]: https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
[kuielab]: https://github.com/kuielab/mdx-net-submission
[decouple]: https://arxiv.org/abs/2109.05418
[mdx_submission]: https://github.com/adefossez/mdx21_demucs
[bandsplit]: https://arxiv.org/abs/2209.15174
[htdemucs]: https://arxiv.org/abs/2211.08553
[samples]: https://ai.honu.io/papers/htdemucs/index.html