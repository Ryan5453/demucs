import { useState, useRef, useEffect } from 'react';
import type { ModelType } from '../../types';

export interface ModelInfo {
    id: ModelType;
    name: string;
    stems: number;
    sizeMB: number;
}

export const MODELS: ModelInfo[] = [
    { id: 'htdemucs', name: 'Demucs v4', stems: 4, sizeMB: 169 },
    { id: 'htdemucs_6s', name: 'Demucs v4 (6-source)', stems: 6, sizeMB: 110 },
    { id: 'hdemucs_mmi', name: 'Demucs v3', stems: 4, sizeMB: 169 },
];

interface ModelPickerProps {
    selectedModel: ModelType;
    modelLoaded: boolean;
    modelLoading: boolean;
    onModelSelect: (model: ModelType) => void;
}

export function ModelPicker({
    selectedModel,
    modelLoaded,
    modelLoading,
    onModelSelect,
}: ModelPickerProps) {
    const [showMenu, setShowMenu] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const currentModel = MODELS.find(m => m.id === selectedModel)!;

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setShowMenu(false);
            }
        };
        if (showMenu) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showMenu]);

    const handleSelect = (model: ModelType) => {
        setShowMenu(false);
        onModelSelect(model);
    };

    return (
        <div className="model-dropdown" ref={dropdownRef}>
            <button
                onClick={() => !modelLoading && setShowMenu(!showMenu)}
                className={`model-dropdown-btn px-5 py-3 bg-slate-800/90 border-2 border-slate-600 rounded-2xl text-slate-100 font-semibold flex items-center gap-2 shadow-sm relative overflow-hidden ${modelLoading ? 'cursor-default' : ''}`}
            >
                {/* Loading progress bar */}
                {modelLoading && (
                    <div className="absolute inset-0 bg-sage-500/20">
                        <div
                            className="h-full bg-sage-500/40 transition-all duration-300 ease-out"
                            style={{
                                animation: 'model-load-sweep 2s ease-in-out infinite'
                            }}
                        />
                    </div>
                )}
                <svg className="w-4 h-4 relative z-10" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
                </svg>
                <span className="relative z-10">
                    {modelLoading ? 'Loading...' : modelLoaded ? currentModel.name : 'Select Model'}
                </span>
                {!modelLoading && (
                    <svg className={`w-4 h-4 ml-1 transition-transform relative z-10 ${showMenu ? 'rotate-180' : ''}`} viewBox="0 0 24 24" fill="currentColor">
                        <path d="M7 10l5 5 5-5H7z" />
                    </svg>
                )}
            </button>
            {showMenu && (
                <div className="model-dropdown-menu">
                    {MODELS.map((model) => {
                        const isLoaded = modelLoaded && selectedModel === model.id;
                        return (
                            <div
                                key={model.id}
                                className={`model-dropdown-item flex items-center gap-3 ${selectedModel === model.id ? 'selected' : ''} ${isLoaded ? 'cursor-default' : ''}`}
                                onClick={() => handleSelect(model.id)}
                            >
                                <div className="flex-1 min-w-0">
                                    <div className="font-semibold text-slate-100">{model.name}</div>
                                    <div className="text-xs text-slate-400">
                                        {model.stems} stems • {model.sizeMB} MB
                                    </div>
                                </div>
                                {isLoaded && (
                                    <svg className="w-5 h-5 text-cyan-400 flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
                                    </svg>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
