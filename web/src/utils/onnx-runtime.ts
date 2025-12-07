import * as ort from 'onnxruntime-web/webgpu';
import type { LogEntry } from '../types';

const isDev = import.meta.env.DEV;
ort.env.wasm.wasmPaths = isDev ? '/node_modules/onnxruntime-web/dist/' : '/';
ort.env.wasm.numThreads = 4;
ort.env.logLevel = 'warning';

let session: ort.InferenceSession | null = null;

export async function loadModel(
    addLog: (message: string, type: LogEntry['type']) => void
): Promise<boolean> {
    try {
        addLog('Starting model load...', 'info');

        const startTime = performance.now();

        session = await ort.InferenceSession.create('https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs.onnx', {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
        });

        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
        addLog(`Model loaded in ${loadTime}s`, 'success');

        return true;
    } catch (error) {
        addLog(`Failed to load model: ${(error as Error).message}`, 'error');
        return false;
    }
}

export function getSession(): ort.InferenceSession | null {
    return session;
}

export { ort };
