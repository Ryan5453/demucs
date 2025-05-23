# Demucs

> [!WARNING] 
> THIS IS IN PROGRESS DONT BELIEVE ANYTHING IN THIS README. DONT USE THE CODE EITHER FOR NOW.

Demucs is a state-of-the-art music source separation model, currently capable of separating drums, bass, and vocals from the rest of the accompaniment. Samples are available [online][samples] for both Hybrid Demucs and Hybrid Transformer Demucs. Checkout [the paper][htdemucs] for more information.

## Installation

### Prerequisites

Before installing Demucs, you need:

- Python 3.9 or later
- FFmpeg (required for audio processing):
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg` 
  - Windows: Download from [FFmpeg.org](https://ffmpeg.org/download.html)

### For Command Line Users

If you want to use Demucs as a command-line tool to separate audio tracks:

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
# or 
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
# or
uv add -U demucs-inference

# If installed with pip
pip install -U demucs-inference
```

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

### Pre-downloading Models

You can download and cache models before using them for offline use:

```bash
# Download the default model (htdemucs)
demucs download

# Download a specific model
demucs download mdx

# Download multiple models
demucs download mdx htdemucs_ft htdemucs_6s

# List all available models
demucs download --list-models

# Download all available models
demucs download --all
```

This is useful when:
- You want to prepare for offline use
- You want to avoid downloading during audio processing
- You're running batch jobs and want to ensure models are ready

### GPU Memory Requirements and Optimization

If you want to use GPU acceleration:

- Minimum requirement: 3GB of GPU RAM (default settings need about 7GB)
- For devices with limited memory:
  - Use `--segment SEGMENT` to reduce split length (set to integer seconds)
  - For 3GB GPU memory, try SEGMENT=8 (quality may be affected by smaller values)
  - Hybrid Transformer models only support a maximum segment length of 7.8 seconds
  - Set environment variable `PYTORCH_NO_CUDA_MEMORY_CACHING=1` to further reduce usage
  - For very limited memory (2GB or less), use `-d cpu` to run on CPU instead
- Processing time on CPU is roughly 1.5× the duration of the track

### Output Format

Separated tracks are stored in the `separated/MODEL_NAME/TRACK_NAME` folder. There you will find four stereo wav files sampled at 44.1 kHz: `drums.wav`, `bass.wav`,
`other.wav`, `vocals.wav` (or `.mp3` if you used the `--mp3` option).

All audio formats supported by `torchaudio` can be processed (i.e. wav, mp3, flac, ogg/vorbis on Linux/macOS, etc.). On Windows, `torchaudio` has limited support, so we rely on `ffmpeg`, which should support pretty much anything.
Audio is resampled on the fly if necessary.

#### Output File Types

- Default: WAV files encoded as int16
- `--float32`: Save as float32 WAV files
- `--int24`: Save as 24-bit integer WAV files
- `--mp3`: Save as MP3 files
- `--mp3-bitrate`: Set MP3 bitrate in kbps (default is 320)

#### Handling Clipping

Demucs will automatically rescale each output stem to avoid clipping, which may affect relative volume between stems. Options:
- `--clip-mode clamp`: Use hard clipping if you prefer preserving relative volumes
- Alternatively, try reducing the volume of the input mixture before processing

### Model Selection

Select pre-trained models with the `-n` flag:

- `htdemucs`: First version of Hybrid Transformer Demucs (default). Trained on MusDB + 800 songs.
- `htdemucs_ft`: Fine-tuned version of `htdemucs`. Better quality but 4× slower.
- `htdemucs_6s`: 6-source version adding `piano` and `guitar` (piano performance is limited).
- `hdemucs_mmi`: Hybrid Demucs v3, retrained on MusDB + 800 songs.
- `mdx`: Trained only on MusDB HQ. Winner on track A at the [MDX][mdx] challenge.
- `mdx_extra`: Trained with extra data (including MusDB test set). Ranked 2nd on track B.
- `mdx_q`, `mdx_extra_q`: Quantized versions. Smaller size but slightly lower quality.

### Processing Options

- `--two-stems=vocals`: Separate vocals from accompaniment (karaoke mode). Replace "vocals" with any source.
- `--shifts=SHIFTS`: Perform multiple predictions with random shifts and average them. Makes processing `SHIFTS` times slower (GPU recommended).
- `--overlap`: Control overlap between prediction windows (default: 0.25). Can be reduced to 0.1 for faster processing.
- `-j N`: Specify number of parallel jobs (e.g., `-j 2`). Multiplies RAM usage by the same amount.

## Demucs API

Demucs provides an API that can be used to separate audio files programmatically.

```python
from demucs.api import Separator, save_audio

# Initialize the separator with desired model and parameters
separator = Separator(model="htdemucs", device="cuda")

# Separate an audio file
original, separated_stems = separator.separate_audio_file("audio_file.mp3")

# Save the separated stems
for stem_name, stem_audio in separated_stems.items():
    save_audio(stem_audio, f"{stem_name}.wav", samplerate=separator.samplerate)
```

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