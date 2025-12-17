// Use global 'ort' from CDN script instead of bundled package
// Import types only - actual runtime comes from CDN
import type { InferenceSession } from 'onnxruntime-web';
import type { LogEntry, ModelType } from '../types';

// Access the global ort object loaded from CDN
const ort = (window as unknown as { ort: typeof import('onnxruntime-web') }).ort;

// Load WASM files from CDN to avoid bundling large files
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.0-dev.20251116-b39e144322/dist/';
ort.env.wasm.numThreads = 4;
ort.env.logLevel = 'warning';

let session: InferenceSession | null = null;
let loadedSources: string[] = [];
let currentBackend: 'webgpu' | 'wasm' | null = null;

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

/**
 * Check if WebGPU is available in the current browser
 */
export async function isWebGPUAvailable(): Promise<boolean> {
    if (!navigator.gpu) {
        return false;
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter !== null;
    } catch {
        return false;
    }
}

/**
 * Get the current backend being used
 */
export function getBackend(): 'webgpu' | 'wasm' | null {
    return currentBackend;
}

export interface ModelLoadResult {
    success: boolean;
    sources: string[];
    backend?: 'webgpu' | 'wasm';
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

        // Check WebGPU availability
        const hasWebGPU = await isWebGPUAvailable();
        const executionProviders = hasWebGPU ? ['webgpu', 'wasm'] : ['wasm'];

        if (hasWebGPU) {
            addLog('Using WebGPU backend', 'info');
        } else {
            addLog('WebGPU not available, using WASM backend', 'info');
        }

        session = await ort.InferenceSession.create(modelUrl, {
            executionProviders,
            graphOptimizationLevel: 'all',
        });

        // Determine which backend was actually used
        currentBackend = hasWebGPU ? 'webgpu' : 'wasm';

        // Set sources based on model type
        loadedSources = MODEL_SOURCES[model];
        addLog(`Model sources: ${loadedSources.join(', ')}`, 'info');

        const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);
        addLog(`Model loaded in ${loadTime}s`, 'success');

        return { success: true, sources: loadedSources, backend: currentBackend };
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

export async function unloadModel(): Promise<void> {
    if (session) {
        await session.release();
        session = null;
    }
    loadedSources = [];
    currentBackend = null;
}

export { ort };
