# 🎛️ demucs-inference

> [!WARNING]
> `demucs-inference` is still in an alpha state and is not recommended for production use. A stable release will be released soon,

Demucs is a state-of-the-art music source separation model capable of separating drums, bass, and vocals from the rest of the accompaniment.
This is a fork of the [author](https://github.com/adefossez)'s [fork](https://github.com/adefossez/demucs) of the [original Demucs repository](https://github.com/facebookresearch/demucs). It has been updated to use modern versions of Python, PyTorch, and TorchCodec.

## Installation

### Prerequisites

#### FFmpeg

Demucs requires FFmpeg v4+ to be installed on your system. You can install it using the following command:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
```

#### UV

The recommended (but optional) way to install demucs-inference is to use UV, an alternative Python package manager. You can install it using the following command:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Temporary Installation using UV

With UV, you can use the `uvx` command to run Demucs without installing it permanently on your system. This sets up a temporary virtual enviornment for the duration of the command. 

```bash
uvx demucs-inference separate audio_file.mp3
```

**Note**: Demucs does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or PyTorch's default CUDA version (currently 12.8) on Linux when using uvx. Demucs will fall back to CPU if one of the above conditions are not met.

### Install using UV

Install Demucs using the following command:

```bash
uv pip install demucs-inference --torch-backend=auto
```

The `--torch-backend=auto` flag automatically detects your GPU and installs the appropriate version of PyTorch compatible with your system.

### Install without UV

Install Demucs using the following command:

```bash
pip install demucs-inference
```

**Note**: Demucs does not specify a specific PyTorch wheel. If you want to use a GPU, view the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and append the correct index URL.

## Usage

After installing Demucs, you can use it like the following:

```bash
# View separation options
demucs separate --help

# Separate one audio file
demucs separate audio_file.mp3

# Separate multiple audio files
demucs separate audio_file_1.mp3 audio_file_2.mp3

# Separate all audio files in a directory
demucs separate /path/to/music/folder
```

## Cog Usage

Demucs provides a [Cog](https://github.com/replicate/cog), which allows you to easily deploy a Demucs model as a REST API. You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

## API Usage

Demucs provides a Python API for separating audio files. Please refer to the [API docs](docs/api.md) for more information.

## Changelog

The [changelog](docs/changelog.md) contains information about the changes between versions of demucs-inference, including a migration guide from upstream demucs.