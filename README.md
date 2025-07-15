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
uvx --with demucs-inference demucs separate audio_file.mp3
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
## API Usage

Demucs provides a Python API for separating audio files. Please refer to the [API docs](docs/api.md) for more information.