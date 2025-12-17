import { useState, useCallback, useRef } from 'react';
import type { DemucsState, LogEntry, ModelType } from '../types';
import { SAMPLE_RATE, SEGMENT_SAMPLES, NFFT, HOP_LENGTH } from '../types';
import { loadModel as loadOnnxModel, unloadModel as unloadOnnxModel, getSession, getSources, ort } from '../utils/onnx-runtime';
import { computeSTFT, computeISTFT, createSTFTBuffers, createISTFTBuffers } from '../utils/audio-processor';
import { createWavBlob } from '../utils/wav-utils';
import { decodeAudioFile } from '../utils/audio-decoder';

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
    const [audioError, setAudioError] = useState<string | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);

    // Store pre-created blob URLs
    const [stemUrls, setStemUrls] = useState<Record<string, string>>({});
    // Store artwork URL (album art from audio file)
    const [artworkUrl, setArtworkUrl] = useState<string | null>(null);
    // Store waveform data for visualization (array of 0-100 values)
    const [stemWaveforms, setStemWaveforms] = useState<Record<string, number[]>>({});

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

    const loadModel = useCallback(async (model: ModelType) => {
        setState(prev => ({ ...prev, modelLoading: true }));
        const result = await loadOnnxModel(model, addLog);
        setState(prev => ({
            ...prev,
            modelLoading: false,
            modelLoaded: result.success,
        }));
    }, [addLog]);

    const unloadModel = useCallback(async () => {
        await unloadOnnxModel();
        setState(prev => ({ ...prev, modelLoaded: false }));
        addLog('Model unloaded', 'info');
    }, [addLog]);

    const clearAudioError = useCallback(() => {
        setAudioError(null);
    }, []);

    const loadAudio = useCallback(async (file: File) => {
        try {
            setAudioError(null);
            addLog(`Loading audio: ${file.name}`, 'info');
            const ctx = getAudioContext();

            const { buffer: audioBuffer, artwork, usedFallback } = await decodeAudioFile(file, ctx);

            if (usedFallback === 'ffmpeg') {
                addLog('Audio decoded using fallback decoder (ffmpeg.wasm)', 'info');
            } else {
                addLog('Audio decoded with Mediabunny', 'info');
            }

            // Store artwork if present
            if (artwork) {
                setArtworkUrl(artwork);
                addLog('Album artwork extracted', 'info');
            }

            addLog('Audio loaded successfully.', 'success');

            setState(prev => ({
                ...prev,
                audioLoaded: true,
                audioBuffer,
                audioFile: file,
            }));
        } catch (error) {
            const errorMessage = (error as Error).message;
            addLog(`Failed to load audio: ${errorMessage}`, 'error');
            setAudioError(errorMessage);
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
            setStemWaveforms({}); // Clear old waveforms
            setStatus('Preparing audio...');
            setProgress(0);

            // Yield to allow React to render the separating UI before heavy processing
            await new Promise(resolve => setTimeout(resolve, 0));

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

            // Get sources from the loaded model
            const sources = getSources();

            const outputs: Record<string, Float32Array> = {};
            for (const source of sources) {
                outputs[source] = new Float32Array(numSamples * numChannels);
            }

            const fadeIn = new Float32Array(OVERLAP);
            const fadeOut = new Float32Array(OVERLAP);
            for (let i = 0; i < OVERLAP; i++) {
                fadeIn[i] = i / OVERLAP;
                fadeOut[i] = 1 - i / OVERLAP;
            }

            // Pre-allocate reusable buffers to reduce GC pressure
            const segmentPlanar = new Float32Array(SEGMENT_SAMPLES * numChannels);
            const segmentInterleaved = new Float32Array(SEGMENT_SAMPLES * numChannels);
            const specBufferSize = numChannels * (NFFT / 2) * Math.ceil(SEGMENT_SAMPLES / HOP_LENGTH);
            const sourceReal = new Float32Array(specBufferSize);
            const sourceImag = new Float32Array(specBufferSize);
            const stftBuffers = createSTFTBuffers();
            const istftBuffers = createISTFTBuffers();

            for (let seg = 0; seg < numSegments; seg++) {
                const segStart = seg * STEP;
                const segEnd = Math.min(segStart + SEGMENT_SAMPLES, numSamples);
                const segLength = segEnd - segStart;

                setStatus(`Separating segment ${seg + 1} of ${numSegments}...`);
                setProgress(((seg + 1) / numSegments) * 95);

                segmentPlanar.fill(0);
                for (let i = 0; i < segLength; i++) {
                    const srcIdx = (segStart + i) * numChannels;
                    segmentPlanar[i] = audio[srcIdx];
                    segmentPlanar[SEGMENT_SAMPLES + i] = audio[srcIdx + 1];
                }

                segmentInterleaved.fill(0);
                for (let i = 0; i < SEGMENT_SAMPLES; i++) {
                    segmentInterleaved[i * 2] = segmentPlanar[i];
                    segmentInterleaved[i * 2 + 1] = segmentPlanar[SEGMENT_SAMPLES + i];
                }

                const stft = computeSTFT(segmentInterleaved, stftBuffers);

                const specShape = [1, numChannels, stft.numBins, stft.numFrames];
                const audioShape = [1, numChannels, SEGMENT_SAMPLES];

                const specRealTensor = new ort.Tensor('float32', stft.real, specShape);
                const specImagTensor = new ort.Tensor('float32', stft.imag, specShape);
                const audioTensor = new ort.Tensor('float32', segmentPlanar, audioShape);

                const results = await session.run({
                    'spec_real': specRealTensor,
                    'spec_imag': specImagTensor,
                    'audio': audioTensor
                });

                const outSpecReal = results['out_spec_real'];
                const outSpecImag = results['out_spec_imag'];
                const outWave = results['out_wave'];

                const specRealData = outSpecReal.data as Float32Array;
                const specImagData = outSpecImag.data as Float32Array;
                const waveData = outWave.data as Float32Array;

                specRealTensor.dispose();
                specImagTensor.dispose();
                audioTensor.dispose();

                for (let s = 0; s < sources.length; s++) {
                    const specOffset = s * numChannels * stft.numBins * stft.numFrames;

                    sourceReal.fill(0);
                    sourceImag.fill(0);

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
                    const freqAudio = computeISTFT(sourceReal, sourceImag, numChannels, stft.numBins, stft.numFrames, SEGMENT_SAMPLES, istftBuffers);

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

                        outputs[sources[s]][outIdx] += leftVal * weight;
                        outputs[sources[s]][outIdx + 1] += rightVal * weight;
                    }
                }

                outSpecReal.dispose();
                outSpecImag.dispose();
                outWave.dispose();
            }

            // Create blob URLs IMMEDIATELY after separation (like original code)
            setStatus('Finalizing...');
            setProgress(98);
            // addLog('Creating audio files...', 'info');

            const urls: Record<string, string> = {};
            const waveforms: Record<string, number[]> = {};
            const numBars = 60; // Number of waveform bars to display

            for (const source of sources) {
                const blob = createWavBlob(outputs[source], numChannels, SAMPLE_RATE);
                urls[source] = URL.createObjectURL(blob);

                // Compute waveform for visualization
                const audioData = outputs[source];
                const samplesPerBar = Math.floor(audioData.length / numBars);
                const bars: number[] = [];

                for (let i = 0; i < numBars; i++) {
                    const start = i * samplesPerBar;
                    const end = Math.min(start + samplesPerBar, audioData.length);

                    // Calculate RMS (root mean square) for this segment
                    let sumSquares = 0;
                    for (let j = start; j < end; j++) {
                        sumSquares += audioData[j] * audioData[j];
                    }
                    const rms = Math.sqrt(sumSquares / (end - start));

                    // Convert to percentage (0-100), with some scaling for visual appeal
                    // Audio RMS is typically 0-0.3 for normal audio, scale to 0-100
                    const barHeight = Math.min(100, Math.max(15, rms * 300));
                    bars.push(barHeight);
                }

                waveforms[source] = bars;
            }

            setStemUrls(urls);
            setStemWaveforms(waveforms);

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

    const resetForNewTrack = useCallback(() => {
        // Revoke old blob URLs to prevent memory leaks
        Object.values(stemUrls).forEach(url => URL.revokeObjectURL(url));
        if (artworkUrl) {
            URL.revokeObjectURL(artworkUrl);
        }

        setState(prev => ({
            ...prev,
            audioLoaded: false,
            audioBuffer: null,
            audioFile: null,
            separating: false,
            progress: 0,
            status: 'Ready',
        }));
        setStemUrls({});
        setStemWaveforms({});
        setArtworkUrl(null);
        setAudioError(null);
    }, [stemUrls, artworkUrl]);

    return {
        ...state,
        stemUrls,
        stemWaveforms,
        artworkUrl,
        audioError,
        loadModel,
        unloadModel,
        loadAudio,
        clearAudioError,
        separateAudio,
        resetForNewTrack,
    };
}
