import { useState, useCallback, useRef } from 'react';
import type { DemucsState, LogEntry } from '../types';
import { SOURCES, SAMPLE_RATE, SEGMENT_SAMPLES } from '../types';
import { loadModel as loadOnnxModel, getSession, ort } from '../utils/onnx-runtime';
import { computeSTFT, computeISTFT } from '../utils/audio-processor';
import { createWavBlob } from '../utils/wav-utils';

const initialState: DemucsState = {
    modelLoaded: false,
    modelLoading: false,
    audioLoaded: false,
    audioBuffer: null,
    audioFile: null,
    separating: false,
    progress: 0,
    status: 'Ready',
    logs: [],
};

export function useDemucs() {
    const [state, setState] = useState<DemucsState>(initialState);
    const audioContextRef = useRef<AudioContext | null>(null);

    // Store pre-created blob URLs
    const [stemUrls, setStemUrls] = useState<Record<string, string>>({});

    const addLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
        setState(prev => ({
            ...prev,
            logs: [...prev.logs, { timestamp: new Date(), message, type }]
        }));
    }, []);

    const setStatus = useCallback((status: string) => {
        setState(prev => ({ ...prev, status }));
    }, []);

    const setProgress = useCallback((progress: number) => {
        setState(prev => ({ ...prev, progress }));
    }, []);

    const getAudioContext = useCallback(() => {
        if (!audioContextRef.current) {
            audioContextRef.current = new AudioContext({ sampleRate: SAMPLE_RATE });
        }
        return audioContextRef.current;
    }, []);

    const loadModel = useCallback(async () => {
        setState(prev => ({ ...prev, modelLoading: true }));
        const success = await loadOnnxModel(addLog);
        setState(prev => ({
            ...prev,
            modelLoading: false,
            modelLoaded: success,
        }));
    }, [addLog]);

    const loadAudio = useCallback(async (file: File) => {
        try {
            addLog(`Loading audio: ${file.name}`, 'info');
            const arrayBuffer = await file.arrayBuffer();
            const ctx = getAudioContext();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);



            addLog('Audio loaded successfully.', 'success');

            setState(prev => ({
                ...prev,
                audioLoaded: true,
                audioBuffer,
                audioFile: file,
            }));
        } catch (error) {
            addLog(`Failed to load audio: ${(error as Error).message}`, 'error');
        }
    }, [addLog, getAudioContext]);

    const separateAudio = useCallback(async () => {
        const session = getSession();
        if (!session || !state.audioBuffer) {
            addLog('Model or audio not loaded', 'error');
            return;
        }

        try {
            setState(prev => ({ ...prev, separating: true }));
            setStemUrls({}); // Clear old URLs
            setStatus('Preparing audio...');
            setProgress(5);
            const startTime = performance.now();
            addLog('Starting separation...', 'info');

            const numChannels = 2;
            const numSamples = state.audioBuffer.length;
            const audio = new Float32Array(numSamples * numChannels);

            const left = state.audioBuffer.getChannelData(0);
            const right = state.audioBuffer.numberOfChannels > 1
                ? state.audioBuffer.getChannelData(1)
                : left;

            for (let i = 0; i < numSamples; i++) {
                audio[i * 2] = left[i];
                audio[i * 2 + 1] = right[i];
            }



            const OVERLAP = Math.floor(SEGMENT_SAMPLES * 0.5);
            const STEP = SEGMENT_SAMPLES - OVERLAP;
            const numSegments = Math.ceil((numSamples - OVERLAP) / STEP);


            const outputs: Record<string, Float32Array> = {};
            for (const source of SOURCES) {
                outputs[source] = new Float32Array(numSamples * numChannels);
            }

            const fadeIn = new Float32Array(OVERLAP);
            const fadeOut = new Float32Array(OVERLAP);
            for (let i = 0; i < OVERLAP; i++) {
                fadeIn[i] = i / OVERLAP;
                fadeOut[i] = 1 - i / OVERLAP;
            }

            for (let seg = 0; seg < numSegments; seg++) {
                const segStart = seg * STEP;
                const segEnd = Math.min(segStart + SEGMENT_SAMPLES, numSamples);
                const segLength = segEnd - segStart;

                setStatus(`Separating segment ${seg + 1} of ${numSegments}...`);
                setProgress(10 + (seg / numSegments) * 80);

                const segmentPlanar = new Float32Array(SEGMENT_SAMPLES * numChannels);
                for (let i = 0; i < segLength; i++) {
                    const srcIdx = (segStart + i) * numChannels;
                    segmentPlanar[i] = audio[srcIdx];
                    segmentPlanar[SEGMENT_SAMPLES + i] = audio[srcIdx + 1];
                }

                const segmentInterleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
                for (let i = 0; i < SEGMENT_SAMPLES; i++) {
                    segmentInterleaved[i * 2] = segmentPlanar[i];
                    segmentInterleaved[i * 2 + 1] = segmentPlanar[SEGMENT_SAMPLES + i];
                }

                // STFT computation
                const stft = computeSTFT(segmentInterleaved);

                const specShape = [1, numChannels, stft.numBins, stft.numFrames];
                const audioShape = [1, numChannels, SEGMENT_SAMPLES];

                const specRealTensor = new ort.Tensor('float32', stft.real, specShape);
                const specImagTensor = new ort.Tensor('float32', stft.imag, specShape);
                const audioTensor = new ort.Tensor('float32', segmentPlanar, audioShape);

                // Inference
                // const startTime = performance.now();

                const results = await session.run({
                    'spec_real': specRealTensor,
                    'spec_imag': specImagTensor,
                    'audio': audioTensor
                });

                // const inferenceTime = ((performance.now() - startTime) / 1000).toFixed(2);
                // Inference completed

                const outSpecReal = results['out_spec_real'];
                const outSpecImag = results['out_spec_imag'];
                const outWave = results['out_wave'];

                const specRealData = outSpecReal.data as Float32Array;
                const specImagData = outSpecImag.data as Float32Array;
                const waveData = outWave.data as Float32Array;

                for (let s = 0; s < SOURCES.length; s++) {
                    const specOffset = s * numChannels * stft.numBins * stft.numFrames;

                    const sourceReal = new Float32Array(numChannels * stft.numBins * stft.numFrames);
                    const sourceImag = new Float32Array(numChannels * stft.numBins * stft.numFrames);

                    for (let c = 0; c < numChannels; c++) {
                        const cOffset = c * stft.numBins * stft.numFrames;
                        for (let b = 0; b < stft.numBins; b++) {
                            for (let t = 0; t < stft.numFrames; t++) {
                                const idx = b * stft.numFrames + t;
                                const specIdx = specOffset + cOffset + idx;
                                sourceReal[cOffset + idx] = specRealData[specIdx];
                                sourceImag[cOffset + idx] = specImagData[specIdx];
                            }
                        }
                    }

                    // iSTFT computation
                    const freqAudio = computeISTFT(sourceReal, sourceImag, numChannels, stft.numBins, stft.numFrames, SEGMENT_SAMPLES);

                    const sourceWaveOffset = s * numChannels * SEGMENT_SAMPLES;

                    for (let i = 0; i < segLength; i++) {
                        const globalIdx = segStart + i;
                        if (globalIdx >= numSamples) continue;

                        const outIdx = globalIdx * numChannels;

                        const leftFreq = freqAudio[i];
                        const rightFreq = freqAudio[SEGMENT_SAMPLES + i];
                        const leftTime = waveData[sourceWaveOffset + i];
                        const rightTime = waveData[sourceWaveOffset + SEGMENT_SAMPLES + i];

                        const leftVal = leftFreq + leftTime;
                        const rightVal = rightFreq + rightTime;

                        let weight = 1.0;
                        if (seg > 0 && i < OVERLAP) {
                            weight = fadeIn[i];
                        }
                        if (seg < numSegments - 1 && i >= SEGMENT_SAMPLES - OVERLAP) {
                            const fadeIdx = i - (SEGMENT_SAMPLES - OVERLAP);
                            weight = fadeOut[fadeIdx];
                        }

                        outputs[SOURCES[s]][outIdx] += leftVal * weight;
                        outputs[SOURCES[s]][outIdx + 1] += rightVal * weight;
                    }
                }
            }

            // Create blob URLs IMMEDIATELY after separation (like original code)
            setStatus('Finalizing...');
            setProgress(98);
            // addLog('Creating audio files...', 'info');

            const urls: Record<string, string> = {};
            for (const source of SOURCES) {
                const blob = createWavBlob(outputs[source], numChannels, SAMPLE_RATE);
                urls[source] = URL.createObjectURL(blob);
            }

            setStemUrls(urls);

            const duration = ((performance.now() - startTime) / 1000).toFixed(2);
            setStatus('Complete!');
            setProgress(100);
            addLog(`Finished separation in ${duration}s.`, 'success');

            setState(prev => ({
                ...prev,
                separating: false,
            }));

        } catch (error) {
            addLog(`Separation failed: ${(error as Error).message}`, 'error');
            setStatus('Error during separation');
            setState(prev => ({ ...prev, separating: false }));
        }
    }, [state.audioBuffer, addLog, setStatus, setProgress]);

    return {
        ...state,
        stemUrls,
        loadModel,
        loadAudio,
        separateAudio,
    };
}
