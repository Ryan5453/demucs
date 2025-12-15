// Use global 'ort' from CDN script instead of bundled package
// Import types only - actual runtime comes from CDN
import type { InferenceSession } from 'onnxruntime-web';
import type { LogEntry } from '../types';
import type { ModelType } from '../types';

// Access the global ort object loaded from CDN
const ort = (window as unknown as { ort: typeof import('onnxruntime-web') }).ort;

// Load WASM files from CDN to avoid bundling large files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/';
ort.env.wasm.numThreads = 4;
ort.env.logLevel = 'warning';

let session: InferenceSession | null = null;
let loadedSources: string[] = [];

// Model URLs on HuggingFace
const MODEL_URLS: Record<ModelType, string> = {
    'htdemucs': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs.onnx',
    'htdemucs_6s': 'https://huggingface.co/Ryan5453/demucs-next/resolve/main/htdemucs_6s.onnx',
    'hdemucs_mmi': 'https://huggingface.co/Ryan5453/demucs-onnx/resolve/main/htdemucs.onnx', // Fallback to htdemucs for now
};

// Sources for each model type - must match the order in the trained model
const MODEL_SOURCES: Record<ModelType, string[]> = {
    'htdemucs': ['drums', 'bass', 'other', 'vocals'],
    'htdemucs_6s': ['drums', 'bass', 'guitar', 'piano', 'other', 'vocals'],
    'hdemucs_mmi': ['drums', 'bass', 'other', 'vocals'], // Uses htdemucs fallback
};

export interface ModelLoadResult {
    success: boolean;
    sources: string[];
}

export async function loadModel(
    model: ModelType,
    addLog: (message: string, type: LogEntry['type']) => void
): Promise<ModelLoadResult> {
    try {
        const modelUrl = MODEL_URLS[model];
        const displayName = model === 'hdemucs_mmi' ? `${model} (using htdemucs fallback)` : model;
        addLog(`Starting model load: ${displayName}...`, 'info');

        const startTime = performance.now();

        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['webgpu'],
            graphOptimizationLevel: 'all',
        });

        // Set sources based on model type
        loadedSources = MODEL_SOURCES[model];
        addLog(`Model sources: ${loadedSources.join(', ')}`, 'info');

        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
        addLog(`Model loaded in ${loadTime}s`, 'success');

        return { success: true, sources: loadedSources };
    } catch (error) {
        addLog(`Failed to load model: ${(error as Error).message}`, 'error');
        return { success: false, sources: [] };
    }
}

export function getSession(): InferenceSession | null {
    return session;
}

export function getSources(): string[] {
    return loadedSources;
}

export { ort };
