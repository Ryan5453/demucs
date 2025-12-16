import FFT from 'fft.js';
import type { STFTResult } from '../types';
import { NFFT, HOP_LENGTH, SEGMENT_SAMPLES } from '../types';

const fftInstance = new FFT(NFFT);

const hannWindow = new Float32Array(NFFT);
for (let i = 0; i < NFFT; i++) {
    hannWindow[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / NFFT));
}

// Pre-calculate dimensions for buffer allocation
const NUM_CHANNELS = 2;
const LE = Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH);
const DEMUCS_PAD = Math.floor(HOP_LENGTH / 2) * 3;
const DEMUCS_PAD_RIGHT = DEMUCS_PAD + LE * HOP_LENGTH - SEGMENT_SAMPLES;
const DEMUCS_PADDED_LENGTH = DEMUCS_PAD + SEGMENT_SAMPLES + DEMUCS_PAD_RIGHT;
const CENTER_PAD = NFFT / 2;
const PADDED_LENGTH = DEMUCS_PADDED_LENGTH + 2 * CENTER_PAD;
const RAW_FRAMES = Math.floor((PADDED_LENGTH - NFFT) / HOP_LENGTH) + 1;
const NUM_BINS = NFFT / 2 + 1;
const OUT_BINS = NUM_BINS - 1;
const OUT_FRAMES = LE;

// ISTFT dimensions
const ISTFT_PAD = Math.floor(HOP_LENGTH / 2) * 3;
const ISTFT_LE = HOP_LENGTH * Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH) + 2 * ISTFT_PAD;

/**
 * Pre-allocated buffers for STFT computation to avoid repeated allocations
 */
export interface STFTBuffers {
    demucs_padded: [Float32Array, Float32Array];
    paddedChannels: [Float32Array, Float32Array];
    real: Float32Array;
    imag: Float32Array;
    outReal: Float32Array;
    outImag: Float32Array;
    fftInput: number[];
    fftOutput: number[];
}

/**
 * Pre-allocated buffers for ISTFT computation to avoid repeated allocations
 */
export interface ISTFTBuffers {
    output: Float32Array;
    windowSum: Float32Array;
    finalOutput: Float32Array;
    ifftInput: number[];
    ifftOutput: number[];
}

/**
 * Create reusable STFT buffers - call once before processing loop
 */
export function createSTFTBuffers(): STFTBuffers {
    return {
        demucs_padded: [
            new Float32Array(DEMUCS_PADDED_LENGTH),
            new Float32Array(DEMUCS_PADDED_LENGTH)
        ],
        paddedChannels: [
            new Float32Array(PADDED_LENGTH),
            new Float32Array(PADDED_LENGTH)
        ],
        real: new Float32Array(NUM_CHANNELS * NUM_BINS * RAW_FRAMES),
        imag: new Float32Array(NUM_CHANNELS * NUM_BINS * RAW_FRAMES),
        outReal: new Float32Array(NUM_CHANNELS * OUT_BINS * OUT_FRAMES),
        outImag: new Float32Array(NUM_CHANNELS * OUT_BINS * OUT_FRAMES),
        fftInput: fftInstance.createComplexArray(),
        fftOutput: fftInstance.createComplexArray(),
    };
}

/**
 * Create reusable ISTFT buffers - call once before processing loop
 */
export function createISTFTBuffers(): ISTFTBuffers {
    return {
        output: new Float32Array(NUM_CHANNELS * ISTFT_LE),
        windowSum: new Float32Array(ISTFT_LE),
        finalOutput: new Float32Array(NUM_CHANNELS * SEGMENT_SAMPLES),
        ifftInput: fftInstance.createComplexArray(),
        ifftOutput: fftInstance.createComplexArray(),
    };
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

/**
 * Compute STFT using pre-allocated buffers to avoid memory allocations
 */
export function computeSTFT(audio: Float32Array, buffers: STFTBuffers): STFTResult {
    const numSamples = audio.length / NUM_CHANNELS;
    const { demucs_padded, paddedChannels, real, imag, outReal, outImag, fftInput, fftOutput } = buffers;

    // Clear buffers
    demucs_padded[0].fill(0);
    demucs_padded[1].fill(0);
    paddedChannels[0].fill(0);
    paddedChannels[1].fill(0);
    real.fill(0);
    imag.fill(0);
    outReal.fill(0);
    outImag.fill(0);

    for (let c = 0; c < NUM_CHANNELS; c++) {
        for (let i = 0; i < DEMUCS_PADDED_LENGTH; i++) {
            const origIdx = i - DEMUCS_PAD;
            const srcIdx = reflectIndex(origIdx, numSamples);
            demucs_padded[c][i] = audio[srcIdx * NUM_CHANNELS + c];
        }
    }

    for (let c = 0; c < NUM_CHANNELS; c++) {
        for (let i = 0; i < PADDED_LENGTH; i++) {
            const origIdx = i - CENTER_PAD;
            if (origIdx >= 0 && origIdx < DEMUCS_PADDED_LENGTH) {
                paddedChannels[c][i] = demucs_padded[c][origIdx];
            } else {
                const srcIdx = reflectIndex(origIdx, DEMUCS_PADDED_LENGTH);
                paddedChannels[c][i] = demucs_padded[c][srcIdx];
            }
        }
    }

    const norm = 1.0 / Math.sqrt(NFFT);

    for (let c = 0; c < NUM_CHANNELS; c++) {
        const channelData = paddedChannels[c];

        for (let f = 0; f < RAW_FRAMES; f++) {
            const frameStart = f * HOP_LENGTH;

            for (let i = 0; i < NFFT; i++) {
                const idx = frameStart + i;
                if (idx < PADDED_LENGTH) {
                    fftInput[i * 2] = channelData[idx] * hannWindow[i];
                } else {
                    fftInput[i * 2] = 0;
                }
                fftInput[i * 2 + 1] = 0;
            }

            fftInstance.transform(fftOutput, fftInput);

            const binOffset = (c * RAW_FRAMES + f) * NUM_BINS;
            for (let k = 0; k < NUM_BINS; k++) {
                real[binOffset + k] = fftOutput[k * 2] * norm;
                imag[binOffset + k] = fftOutput[k * 2 + 1] * norm;
            }
        }
    }

    for (let c = 0; c < NUM_CHANNELS; c++) {
        for (let f = 0; f < OUT_FRAMES; f++) {
            for (let b = 0; b < OUT_BINS; b++) {
                const srcIdx = (c * RAW_FRAMES + (f + 2)) * NUM_BINS + b;
                const dstIdx = c * OUT_BINS * OUT_FRAMES + b * OUT_FRAMES + f;
                outReal[dstIdx] = real[srcIdx];
                outImag[dstIdx] = imag[srcIdx];
            }
        }
    }

    return { real: outReal, imag: outImag, numBins: OUT_BINS, numFrames: OUT_FRAMES };
}

/**
 * Compute ISTFT using pre-allocated buffers to avoid memory allocations
 */
export function computeISTFT(
    real: Float32Array,
    imag: Float32Array,
    numChannels: number,
    numBins: number,
    numFrames: number,
    targetLength: number,
    buffers: ISTFTBuffers
): Float32Array {
    const paddedBins = numBins + 1;
    const paddedFrames = numFrames + 4;
    const { output, windowSum, finalOutput, ifftInput, ifftOutput } = buffers;

    // Clear buffers
    output.fill(0);
    windowSum.fill(0);
    finalOutput.fill(0);

    const nfft = NFFT;
    const hopLength = HOP_LENGTH;
    const scale = Math.sqrt(nfft);

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
                if (outIdx >= 0 && outIdx < ISTFT_LE) {
                    const sample = ifftOutput[i * 2] * hannWindow[i];
                    output[c * ISTFT_LE + outIdx] += sample;
                    if (c === 0) {
                        windowSum[outIdx] += hannWindow[i] * hannWindow[i];
                    }
                }
            }
        }
    }

    for (let c = 0; c < numChannels; c++) {
        for (let i = 0; i < ISTFT_LE; i++) {
            if (windowSum[i] > 1e-8) {
                output[c * ISTFT_LE + i] /= windowSum[i];
            }
        }
    }

    for (let c = 0; c < numChannels; c++) {
        for (let i = 0; i < targetLength; i++) {
            finalOutput[c * targetLength + i] = output[c * ISTFT_LE + ISTFT_PAD + i];
        }
    }

    return finalOutput;
}
