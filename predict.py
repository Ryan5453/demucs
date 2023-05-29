import tempfile
from io import BytesIO
from typing import List

import torch
from cog import BaseModel, BasePredictor, File, Input, Path
from demucs.apply import apply_model, BagOfModels
from demucs.audio import save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track
from demucs.htdemucs import HTDemucs


class DemucsStem(BaseModel):
    name: str
    audio: File


class DemucsResponse(BaseModel):
    stems: List[DemucsStem]


class Predictor(BasePredictor):

    def predict(
        self,
        audio: Path = Input(description="Upload the file to be processed here."),
        model: str = Input(
            default="htdemucs",
            description="Choose the demucs audio that proccesses your audio. Options: htdemucs (first version of hybrid transformer demucs), htdemucs_ft (fine-tuned version of htdemucs, separation will take 4 times longer but may be a bit better), htdemucs_6s (adds piano and guitar sources to htdemucs), hdemucs_mmi (hybrid demucs v3, this is what the cog previously used by default), mdx (trained on exclusively MusDB HQ), mdx_q (quantized version of mdx, slightly faster but worse quality), mdx_extra (adds extra training data to mdx), mdx_extra_q (quantized version of mdx_extra, slightly faster but worse quality)",
            choices=["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_q", "mdx_extra", "mdx_extra_q"]
        ),
        two_stems: str = Input(
            default=None,
            description="If you just want to isolate one stem, you can choose it here. This does not improve performance, as it just combines all of the stems that you did not choose.",
            choices=["drums", "bass", "other", "vocals", "guitar", "piano"],
        ),
        output_format: str = Input(
            default="mp3",
            description="Choose the audio format you would like the result to be returned in.",
            choices=["mp3", "aac", "flac", "wav"]
        ),
        no_split: bool = Input(
            default=True,
            description=""
        ),
        segment: int = Input(
            default=None,
            description=""
        ),
        clip_mode: str = Input(
            default="rescale",
            description="Choose the strategy for avoiding clipping. Rescaling adjusts the overall scale of a signal to prevent any clipping, while hard clipping limits the signal to a maximum range, distorting parts of the signal that exceed that range.",
            choices=["rescale", "clamp"],
        ),
        shifts: int = Input(
            default=1,
            description="Choose the amount random shifts for equivariant stabilization. This performs multiple predictions with random shifts of the input and averages them, which makes it x times slower.",
        ),
        overlap: float = Input(
            default=0.25, 
            description="Choose the amount of overlap between prediction windows."
        ),
    ) -> DemucsResponse:

        model = get_model(model)

        if isinstance(model, BagOfModels):
            if segment is not None:
                for sub in model.models:
                    sub.segment = segment
        else:
            if segment is not None:
                model.segment = segment

        if two_stems is not None and two_stems not in model.sources:
            raise Exception("Chosen stem is not supported by chosen model.")

        model.cpu()
        model.eval()

        wav = load_track(audio, model.audio_channels, model.samplerate)

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        sources = apply_model(
            model,
            wav[None],
            device="cuda" if torch.cuda.is_available() else "cpu",
            split=True,
            shifts=shifts,
            overlap=overlap,
            progress=True,
        )[0]
        sources = sources * ref.std() + ref.mean()

        kwargs = {
            "samplerate": model.samplerate,
            "bitrate": 320,
            "clip": clip_mode,
            "as_float": False,
            "bits_per_sample": 24,
        }

        output_stems = []

        if two_stems is None:
            for source, name in zip(sources, self.model.sources):
                with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                    save_audio(source.cpu(), f.name, **kwargs)
                    output_stems.append(
                        DemucsStem(name=name, audio=BytesIO(open(f.name, "rb").read()))
                    )
        else:
            sources = list(sources)

            with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                save_audio(
                    sources[self.model.sources.index(two_stems)].cpu(), f.name, **kwargs
                )
                output_stems.append(
                    DemucsStem(name=two_stems, audio=BytesIO(open(f.name, "rb").read()))
                )

            sources.pop(self.model.sources.index(two_stems))

            other_stem = torch.zeros_like(sources[0])
            for i in sources:
                other_stem += i

            with tempfile.NamedTemporaryFile(suffix=f".{output_format}") as f:
                save_audio(other_stem.cpu(), f.name, **kwargs)
                output_stems.append(
                    DemucsStem(
                        name="no_" + two_stems, audio=BytesIO(open(f.name, "rb").read())
                    )
                )

        return DemucsResponse(stems=output_stems)
