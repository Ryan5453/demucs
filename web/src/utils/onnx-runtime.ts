// Use global 'ort' from CDN script instead of bundled package
// Import types only - actual runtime comes from CDN
import type { InferenceSession } from 'onnxruntime-web';
import type { LogEntry } from '../types';

// Access the global ort object loaded from CDN
const ort = (window as unknown as { ort: typeof import('onnxruntime-web') }).ort;

// Load WASM files from CDN to avoid bundling large files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/';
ort.env.wasm.numThreads = 4;
ort.env.logLevel = 'warning';

let session: InferenceSession | null = null;

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

export function getSession(): InferenceSession | null {
    return session;
}

export { ort };

