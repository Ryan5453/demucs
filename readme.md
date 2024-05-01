# About
Demucs is an audio source separator model created by Facebook Research. The model ingests an audio file, and returns stems (the separated parts such as the vocals, drums, and bass.)

The original author of Demucs left Meta and therefore the original repository is not maintained, so this Cog uses a fork by the original author. You can find the fork [here.](https://github.com/adefossez/demucs) 

## Model Versions

### Demucs v4 (Hybrid Transformers Demucs)
- htdemucs - base model
- htdemucs_ft - fine-tuned version (this has better results but will take 4x as long)
- htdemucs_6s - adds piano and guitar stems (piano does not work very well)
### Demucs v3 (Hybrid Demucs)
- hdemucs_mmi - base model
### Demucs v2
- mdx_q - quantized of the base model (trained on only MusDB)
- mdx_extra_q - quantized version of the base model with extra data