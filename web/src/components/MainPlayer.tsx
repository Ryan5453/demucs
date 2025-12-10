import { useRef, useState } from 'react';
import { AquaToolbarButton, AquaToolbarDivider } from './ui/AquaButton';
import { InfoPanel, InfoTime, InfoStatus, AquaProgressBar, StatusDot } from './ui/LCDPanel';
import { FileAudio, Scissors, Loader2, ChevronDown } from 'lucide-react';

export type ModelType = 'htdemucs' | 'htdemucs_6s' | 'hdemucs_mmi';

interface ModelInfo {
    id: ModelType;
    name: string;
    description: string;
}

const MODELS: ModelInfo[] = [
    { id: 'htdemucs', name: 'Demucs v4', description: '4 stems (drums, bass, other, vocals)' },
    { id: 'htdemucs_6s', name: 'Demucs v4 (6-source)', description: '6 stems (drums, bass, guitar, piano, other, vocals)' },
    { id: 'hdemucs_mmi', name: 'Demucs v3', description: '4 stems (legacy model)' },
];

interface MainPlayerProps {
    fileName: string | null;
    status: string;
    progress: number;
    duration: number;
    modelLoaded: boolean;
    modelLoading: boolean;
    audioLoaded: boolean;
    separating: boolean;
    onLoadModel: (model: ModelType) => void;
    onLoadAudio: (file: File) => void;
    onSeparate: () => void;
}

export function MainPlayer({
    fileName,
    status,
    progress,
    duration,
    modelLoaded,
    modelLoading,
    audioLoaded,
    separating,
    onLoadModel,
    onLoadAudio,
    onSeparate,
}: MainPlayerProps) {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [selectedModel, setSelectedModel] = useState<ModelType>('htdemucs');
    const [showModelMenu, setShowModelMenu] = useState(false);

    const handleFileClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            onLoadAudio(file);
        }
    };

    const handleModelSelect = (model: ModelType) => {
        setSelectedModel(model);
        setShowModelMenu(false);
        onLoadModel(model);
    };

    const currentModel = MODELS.find(m => m.id === selectedModel)!;
    const displayText = fileName || 'No audio loaded';

    return (
        <div className="flex flex-col gap-4">
            {/* Toolbar */}
            <div className="aqua-toolbar mx-[-12px] mt-[-12px] justify-center">
                {/* Model Selection */}
                <div className="relative">
                    <button
                        className="aqua-toolbar-btn"
                        onClick={() => setShowModelMenu(!showModelMenu)}
                        disabled={modelLoading || modelLoaded}
                        style={{ minWidth: '100px', maxWidth: '160px', flex: '1 1 auto' }}
                    >
                        <span className="aqua-toolbar-btn-icon">
                            {modelLoading ? <Loader2 className="animate-spin" size={20} /> : <ChevronDown size={16} />}
                        </span>
                        <span className="aqua-toolbar-btn-label">
                            {modelLoading ? 'Loading...' : modelLoaded ? currentModel.name : 'Load Model'}
                        </span>
                    </button>

                    {/* Dropdown Menu */}
                    {showModelMenu && (
                        <div
                            className="absolute top-full left-0 mt-1 z-50 bg-white border border-gray-500 shadow-md"
                            style={{ minWidth: '180px' }}
                        >
                            {MODELS.map((model) => (
                                <button
                                    key={model.id}
                                    className={`w-full text-left px-3 py-1.5 text-[11px] hover:bg-[#3875d7] hover:text-white border-b border-gray-200 last:border-b-0 ${selectedModel === model.id ? 'bg-[#3875d7] text-white' : 'text-gray-800'
                                        }`}
                                    onClick={() => handleModelSelect(model.id)}
                                >
                                    {model.name}
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                <AquaToolbarDivider />

                <AquaToolbarButton
                    icon={<FileAudio size={20} />}
                    label="Load Audio"
                    onClick={handleFileClick}
                />

                <AquaToolbarDivider />

                <AquaToolbarButton
                    icon={separating ? <Loader2 className="animate-spin" size={20} /> : <Scissors size={20} />}
                    label={separating ? 'Working...' : 'Separate'}
                    onClick={onSeparate}
                    disabled={!modelLoaded || !audioLoaded || separating}
                />

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                />
            </div>

            {/* Info Display */}
            <InfoPanel className="flex flex-wrap items-center gap-3 sm:gap-4">
                <InfoTime seconds={duration} />
                <div className="flex-1 min-w-0 flex flex-col gap-0.5 overflow-hidden" style={{ minWidth: '120px' }}>
                    <span className="info-title truncate">{displayText}</span>
                    <InfoStatus text={status} />
                </div>
                <div className="flex gap-2 sm:gap-3">
                    <StatusDot
                        status={modelLoading ? 'loading' : modelLoaded ? 'on' : 'off'}
                        label="Model"
                    />
                    <StatusDot
                        status={audioLoaded ? 'on' : 'off'}
                        label="Audio"
                    />
                </div>
            </InfoPanel>

            {/* Progress Bar */}
            <AquaProgressBar progress={progress} />
        </div>
    );
}
