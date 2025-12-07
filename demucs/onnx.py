# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025-present Ryan Fahey
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from pathlib import Path

import torch
import torch.nn as nn
import typer
from rich.console import Console
from typing_extensions import Annotated

from .repo import ModelRepository
from .htdemucs import HTDemucs
from .blocks import spectro

console = Console()


class HTDemucsONNXWrapper(nn.Module):
    """
    Wrapper that makes HTDemucs compatible with ONNX export.
    
    This wrapper:
    1. Accepts pre-computed spectrogram magnitude (real/imag as channels)
    2. Handles the neural network processing
    3. Returns separated spectrograms and waveforms
    
    STFT/iSTFT are handled by the caller (JavaScript in browser).
    """
    
    def __init__(self, model: HTDemucs):
        super().__init__()
        self.model = model
        self.sources = model.sources
        self.samplerate = model.samplerate
        self.audio_channels = model.audio_channels
        self.nfft = model.nfft
        self.hop_length = model.hop_length
        
        # Remove references to training-specific features
        self.model.training = False
        
    def forward(self, spec_real: torch.Tensor, spec_imag: torch.Tensor, 
                mix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONNX export.
        
        Args:
            spec_real: Real part of spectrogram [B, C, Fq, T]
            spec_imag: Imaginary part of spectrogram [B, C, Fq, T]
            mix: Raw audio waveform [B, C, samples]
        
        Returns:
            Tuple of (out_spec_real, out_spec_imag, out_wave):
            - out_spec_real: Real part of separated spectrograms [B, S, C, Fq, T]
            - out_spec_imag: Imaginary part of separated spectrograms [B, S, C, Fq, T]
            - out_wave: Separated waveforms from time branch [B, S, C, samples]
        """
        B, C, Fq, T = spec_real.shape
        
        # Combine real/imag into channels (CaC format)
        # CaC format is [ch0_real, ch0_imag, ch1_real, ch1_imag, ...]
        # spec_real: [B, C, Fq, T], spec_imag: [B, C, Fq, T]
        # Stack to get [B, C, 2, Fq, T] then reshape to [B, C*2, Fq, T]
        x = torch.stack([spec_real, spec_imag], dim=2)  # [B, C, 2, Fq, T]
        x = x.reshape(B, C * 2, Fq, T)  # [B, C*2, Fq, T] with interleaved real/imag
        
        # Normalize frequency branch input
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # Normalize time branch input  
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Encoder
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []
        
        for idx, encode in enumerate(self.model.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.model.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.model.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.model.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.model.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.model.freq_emb_scale * emb
            saved.append(x)

        # Cross-transformer
        if self.model.crosstransformer:
            if self.model.bottom_channels:
                b, c, f, t = x.shape
                x = x.flatten(2)
                x = self.model.channel_upsampler(x)
                x = x.view(b, -1, f, t)
                xt = self.model.channel_upsampler_t(xt)

            x, xt = self.model.crosstransformer(x, xt)

            if self.model.bottom_channels:
                x = x.flatten(2)
                x = self.model.channel_downsampler(x)
                x = x.view(b, -1, f, t)
                xt = self.model.channel_downsampler_t(xt)

        # Decoder
        for idx, decode in enumerate(self.model.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.model.depth - len(self.model.tdecoder)
            if idx >= offset:
                tdec = self.model.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Reshape outputs
        S = len(self.sources)
        
        # Frequency branch: [B, S, C*2, Fq, T]
        # CaC format is [ch0_real, ch0_imag, ch1_real, ch1_imag, ...]
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]
        
        # Split CaC into real/imag properly
        # CaC layout: [ch0_real, ch0_imag, ch1_real, ch1_imag]
        # We need to extract [ch0_real, ch1_real] and [ch0_imag, ch1_imag]
        # x shape: [B, S, C*2, Fq, T]
        out_spec_real = x[:, :, 0::2, :, :]  # Every other channel starting at 0
        out_spec_imag = x[:, :, 1::2, :, :]  # Every other channel starting at 1

        # Time branch: [B, S, C, samples]
        samples = mix.shape[-1]
        xt = xt.view(B, S, -1, samples)
        xt = xt * stdt[:, None] + meant[:, None]

        return out_spec_real, out_spec_imag, xt


def compute_stft_for_export(audio: torch.Tensor, nfft: int, hop_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute STFT for model input, matching HTDemucs preprocessing.
    
    Args:
        audio: Input audio [B, C, samples]
        nfft: FFT size
        hop_length: Hop length
        
    Returns:
        Tuple of (real, imag) spectrograms [B, C, Fq, T]
    """
    # Padding to match HTDemucs._spec
    le = int(math.ceil(audio.shape[-1] / hop_length))
    pad = hop_length // 2 * 3
    
    # Pad the audio
    padded = torch.nn.functional.pad(audio, (pad, pad + le * hop_length - audio.shape[-1]), mode='reflect')
    
    # Compute STFT
    z = spectro(padded, nfft, hop_length)
    
    # Trim to expected size
    z = z[..., :-1, :]  # Remove last frequency bin
    z = z[..., 2:2+le]  # Trim time dimension
    
    # Split into real and imaginary
    real = z.real
    imag = z.imag
    
    return real, imag


def export_to_onnx(
    model_name: str = "htdemucs",
    output_path: str = "htdemucs.onnx",
    opset_version: int = 17,
    segment_seconds: float = 10.0,
) -> str:
    """
    Export HTDemucs model to ONNX format.
    
    Args:
        model_name: Name of the model to export
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version
        segment_seconds: Audio segment length in seconds
        
    Returns:
        Path to the exported ONNX model
    """
    console.print(f"[bold]Loading model:[/bold] {model_name}")
    repo = ModelRepository()
    model = repo.get_model(model_name)
    
    if not isinstance(model, HTDemucs):
        raise ValueError(f"Model {model_name} is not an HTDemucs model")
    
    model.eval()
    
    # Create the ONNX wrapper
    wrapper = HTDemucsONNXWrapper(model)
    wrapper.eval()
    
    # Create dummy inputs
    samplerate = model.samplerate
    segment_samples = int(segment_seconds * samplerate)
    nfft = model.nfft
    hop_length = model.hop_length
    
    # Calculate spectrogram dimensions
    le = int(math.ceil(segment_samples / hop_length))
    fq = nfft // 2
    
    console.print(f"[dim]Sample rate:[/dim] {samplerate}")
    console.print(f"[dim]Segment samples:[/dim] {segment_samples}")
    console.print(f"[dim]Spectrogram shape:[/dim] [B, C, {fq}, {le}]")
    console.print(f"[dim]Sources:[/dim] {model.sources}")
    
    # Create dummy tensors
    batch_size = 1
    audio_channels = model.audio_channels
    
    dummy_audio = torch.randn(batch_size, audio_channels, segment_samples)
    dummy_spec_real, dummy_spec_imag = compute_stft_for_export(dummy_audio, nfft, hop_length)
    
    console.print(f"[dim]Dummy audio shape:[/dim] {list(dummy_audio.shape)}")
    console.print(f"[dim]Dummy spec_real shape:[/dim] {list(dummy_spec_real.shape)}")
    console.print(f"[dim]Dummy spec_imag shape:[/dim] {list(dummy_spec_imag.shape)}")
    
    # Test forward pass
    console.print("[bold]Testing forward pass...[/bold]")
    with torch.no_grad():
        out_real, out_imag, out_wave = wrapper(dummy_spec_real, dummy_spec_imag, dummy_audio)
    console.print(f"[dim]Output spec_real shape:[/dim] {list(out_real.shape)}")
    console.print(f"[dim]Output spec_imag shape:[/dim] {list(out_imag.shape)}")
    console.print(f"[dim]Output wave shape:[/dim] {list(out_wave.shape)}")
    
    # Export to ONNX
    console.print(f"[bold]Exporting to ONNX:[/bold] {output_path}")
    
    # Use torch.onnx.export
    torch.onnx.export(
        wrapper,
        (dummy_spec_real, dummy_spec_imag, dummy_audio),
        output_path,
        input_names=["spec_real", "spec_imag", "audio"],
        output_names=["out_spec_real", "out_spec_imag", "out_wave"],
        dynamic_axes={
            "spec_real": {0: "batch", 3: "time"},
            "spec_imag": {0: "batch", 3: "time"},
            "audio": {0: "batch", 2: "samples"},
            "out_spec_real": {0: "batch", 4: "time"},
            "out_spec_imag": {0: "batch", 4: "time"},
            "out_wave": {0: "batch", 3: "samples"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    # Check file size
    path = Path(output_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓[/green] Successfully exported to [bold]{output_path}[/bold] ({size_mb:.2f} MB)")
    
    return output_path


def export_onnx_command(
    model: Annotated[
        str,
        typer.Option(
            "-m",
            "--model",
            help="Model name to export",
        ),
    ] = "htdemucs",
    output: Annotated[
        str,
        typer.Option(
            "-o",
            "--output",
            help="Output ONNX file path",
        ),
    ] = "htdemucs.onnx",
    opset: Annotated[
        int,
        typer.Option(
            help="ONNX opset version",
        ),
    ] = 17,
    segment: Annotated[
        float,
        typer.Option(
            help="Segment length in seconds",
        ),
    ] = 10.0,
):
    """
    Export HTDemucs model to ONNX format for browser inference.
    
    This is an internal developer tool for creating ONNX models
    that can be used with ONNX Runtime Web in the browser.
    """
    try:
        export_to_onnx(
            model_name=model,
            output_path=output,
            opset_version=opset,
            segment_seconds=segment,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error exporting model:[/red] {e}")
        raise typer.Exit(1)
