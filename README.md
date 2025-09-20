# 🎛️ demucs-inference

Demucs is a state-of-the-art music source separation model capable of separating drums, bass, and vocals from the rest of the accompaniment.
This is a fork of the [author](https://github.com/adefossez)'s [fork](https://github.com/adefossez/demucs) of the [original Demucs repository](https://github.com/facebookresearch/demucs). This fork is a infrence-only version, updated to use the latest versions of PyTorch and TorchAudio.

## Installation

The reccomended way to install demucs-inference is to use UV, an alternative Python package manager.

### Installing UV

Use the following commands to install UV:
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

**Note**: Demucs does not specify a specific PyTorch wheel. This means that GPUs will only work on Apple Silicon or the current version of CUDA (currently 12.8) on Linux. Demucs will fall back to CPU if one of the above conditions are not met.

### Install using UV

Install Demucs using the following command:

```bash
uv pip install demucs-inference --torch-backend=auto
```

The `--torch-backend=auto` flag automatically detects your GPU and installs the appropriate PyTorch version. If you are on Windows and have a Intel GPU (XPU), you will need to use `--torch-backend=xpu` as UV will not automatically detect it.

### Install without UV

Install Demucs using the following command:

```bash
pip install demucs-inference
```

**Note**: Demucs does not specify a specific PyTorch wheel. If you want to use a GPU, view the [PyTorch installation guide](https://pytorch.org/get-started/locally/) and append the correct index URL.

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

**Note:** Demucs also provides a `demucs-inference` command in your enviornment. This is identical to the `demucs` command, but is provided for compatibility with uvx.

## Cog Usage

Demucs provides a [Cog](https://github.com/replicate/cog), which allows you to easily deploy a Demucs model as a REST API. You can alternatively use the hosted version at [Replicate](https://replicate.com/ryan5453/demucs).

## API Usage

Demucs provides a Python API for separating audio files. Please refer to the [API docs](docs/api.md) for more information.

## Changelog

The [changelog](docs/changelog.md) contains information about the changes between versions of demucs-inference, including a migration guide from upstream demucs.