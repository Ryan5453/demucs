# 🎛️ Demucs

Demucs is a state-of-the-art music source separation model, currently capable of separating drums, bass, and vocals from the rest of the accompaniment. Samples are available [online](https://ai.honu.io/papers/htdemucs/index.html) for both Hybrid Demucs and Hybrid Transformer Demucs. Checkout [the paper](https://arxiv.org/abs/2211.08553) for more information.

## Installation

### Prerequisites

Before installing Demucs, you need to install FFmpeg on your system.

Use the following commands to install FFmpeg:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg` 
  - Windows: Download from [FFmpeg.org](https://ffmpeg.org/download.html)

### Install UV (optional, but recommended)

It is recommended to install Demucs using UV, a fast, modern Python package manager with isolated environments. 

Use the following commands to install UV:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Temporary Installation

With UV, you can use the `uvx` command to run Demucs without installing it permanently on your system. This sets up a temporary virtual enviornment for the duration of the command. Keep in mind Demucs does not come with a CUDA-enabled version of PyTorch which means it will only run on CPU or Apple Silicon GPUs.

```bash
uvx --with demucs-inference demucs audio_file.mp3
```

### Install using UV

Install Demucs using the following command:

```bash
uv pip install demucs-inference --torch-backend=auto
```
### Install without UV

Install Demucs using the following command:

```bash
pip install demucs-inference
```

**Note**: For (non-Apple Silicon) GPU support, either use `--extra-index-url`  with a PyTorch wheel or install PyTorch with support for your GPU yourself.

## Usage

After installing Demucs, you can use it like the following:

```bash
# View all options
demucs --help

# Separate one audio file
demucs separate audio_file.mp3

# Separate multiple audio files
demucs separate audio_file_1.mp3 audio_file_2.mp3

# Separate all audio files in the current directory
demucs separate *.mp3
```


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
