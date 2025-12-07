import FFT from 'fft.js';
import type { STFTResult } from '../types';
import { NFFT, HOP_LENGTH } from '../types';

const fftInstance = new FFT(NFFT);

const hannWindow = new Float32Array(NFFT);
for (let i = 0; i < NFFT; i++) {
    hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / NFFT));
}

function reflectIndex(i: number, len: number): number {
    while (i < 0 || i >= len) {
        if (i < 0) {
            i = -i;
        }
        if (i >= len) {
            i = 2 * (len - 1) - i;
        }
    }
    return i;
}

export function computeSTFT(audio: Float32Array): STFTResult {
    const numChannels = 2;
    const numSamples = audio.length / numChannels;

    const le = Math.ceil(numSamples / HOP_LENGTH);
    const demucs_pad = Math.floor(HOP_LENGTH / 2) * 3;
    const demucs_pad_right = demucs_pad + le * HOP_LENGTH - numSamples;
    const demucs_padded_length = demucs_pad + numSamples + demucs_pad_right;

    const demucs_padded = [
        new Float32Array(demucs_padded_length),
        new Float32Array(demucs_padded_length)
    ];

    for (let c = 0; c < numChannels; c++) {
        for (let i = 0; i < demucs_padded_length; i++) {
            const origIdx = i - demucs_pad;
            const srcIdx = reflectIndex(origIdx, numSamples);
            demucs_padded[c][i] = audio[srcIdx * numChannels + c];
        }
    }

    const centerPad = NFFT / 2;
    const paddedLength = demucs_padded_length + 2 * centerPad;

    const paddedChannels = [
        new Float32Array(paddedLength),
        new Float32Array(paddedLength)
    ];

    for (let c = 0; c < numChannels; c++) {
        for (let i = 0; i < paddedLength; i++) {
            const origIdx = i - centerPad;
            if (origIdx >= 0 && origIdx < demucs_padded_length) {
                paddedChannels[c][i] = demucs_padded[c][origIdx];
            } else {
                const srcIdx = reflectIndex(origIdx, demucs_padded_length);
                paddedChannels[c][i] = demucs_padded[c][srcIdx];
            }
        }
    }

    const rawFrames = Math.floor((paddedLength - NFFT) / HOP_LENGTH) + 1;
    const numBins = NFFT / 2 + 1;

    const real = new Float32Array(numChannels * numBins * rawFrames);
    const imag = new Float32Array(numChannels * numBins * rawFrames);

    const fftInput = fftInstance.createComplexArray();
    const fftOutput = fftInstance.createComplexArray();

    const norm = 1.0 / Math.sqrt(NFFT);

    for (let c = 0; c < numChannels; c++) {
        const channelData = paddedChannels[c];

        for (let f = 0; f < rawFrames; f++) {
            const frameStart = f * HOP_LENGTH;

            for (let i = 0; i < NFFT; i++) {
                const idx = frameStart + i;
                if (idx < paddedLength) {
                    fftInput[i * 2] = channelData[idx] * hannWindow[i];
                } else {
                    fftInput[i * 2] = 0;
                }
                fftInput[i * 2 + 1] = 0;
            }

            fftInstance.transform(fftOutput, fftInput);

            const binOffset = (c * rawFrames + f) * numBins;
            for (let k = 0; k < numBins; k++) {
                real[binOffset + k] = fftOutput[k * 2] * norm;
                imag[binOffset + k] = fftOutput[k * 2 + 1] * norm;
            }
        }
    }

    const outBins = numBins - 1;
    const outFrames = le;

    const outReal = new Float32Array(numChannels * outBins * outFrames);
    const outImag = new Float32Array(numChannels * outBins * outFrames);

    for (let c = 0; c < numChannels; c++) {
        for (let f = 0; f < outFrames; f++) {
            for (let b = 0; b < outBins; b++) {
                const srcIdx = (c * rawFrames + (f + 2)) * numBins + b;
                const dstIdx = c * outBins * outFrames + b * outFrames + f;
                outReal[dstIdx] = real[srcIdx];
                outImag[dstIdx] = imag[srcIdx];
            }
        }
    }

    return { real: outReal, imag: outImag, numBins: outBins, numFrames: outFrames };
}

export function computeISTFT(
    real: Float32Array,
    imag: Float32Array,
    numChannels: number,
    numBins: number,
    numFrames: number,
    targetLength: number
): Float32Array {
    const paddedBins = numBins + 1;
    const paddedFrames = numFrames + 4;

    const pad = Math.floor(HOP_LENGTH / 2) * 3;
    const le = HOP_LENGTH * Math.ceil(targetLength / HOP_LENGTH) + 2 * pad;

    const output = new Float32Array(numChannels * le);
    const windowSum = new Float32Array(le);

    const nfft = NFFT;
    const hopLength = HOP_LENGTH;

    const scale = Math.sqrt(nfft);

    const ifftInput = fftInstance.createComplexArray();
    const ifftOutput = fftInstance.createComplexArray();

    for (let c = 0; c < numChannels; c++) {
        for (let fp = 0; fp < paddedFrames; fp++) {
            const f = fp - 2;

            for (let i = 0; i < nfft * 2; i++) {
                ifftInput[i] = 0;
            }

            for (let b = 0; b < paddedBins; b++) {
                let realVal = 0, imagVal = 0;

                if (f >= 0 && f < numFrames && b < numBins) {
                    const srcIdx = c * numBins * numFrames + b * numFrames + f;
                    realVal = real[srcIdx];
                    imagVal = imag[srcIdx];
                }

                ifftInput[b * 2] = realVal * scale;
                ifftInput[b * 2 + 1] = imagVal * scale;
            }

            for (let b = 1; b < paddedBins - 1; b++) {
                const negIdx = nfft - b;
                ifftInput[negIdx * 2] = ifftInput[b * 2];
                ifftInput[negIdx * 2 + 1] = -ifftInput[b * 2 + 1];
            }

            fftInstance.inverseTransform(ifftOutput, ifftInput);

            const frameStart = fp * hopLength;
            for (let i = 0; i < nfft; i++) {
                const outIdx = frameStart + i - nfft / 2;
                if (outIdx >= 0 && outIdx < le) {
                    const sample = ifftOutput[i * 2] * hannWindow[i];
                    output[c * le + outIdx] += sample;
                    if (c === 0) {
                        windowSum[outIdx] += hannWindow[i] * hannWindow[i];
                    }
                }
            }
        }
    }

    for (let c = 0; c < numChannels; c++) {
        for (let i = 0; i < le; i++) {
            if (windowSum[i] > 1e-8) {
                output[c * le + i] /= windowSum[i];
            }
        }
    }

    const finalOutput = new Float32Array(numChannels * targetLength);
    for (let c = 0; c < numChannels; c++) {
        for (let i = 0; i < targetLength; i++) {
            finalOutput[c * targetLength + i] = output[c * le + pad + i];
        }
    }

    return finalOutput;
}
