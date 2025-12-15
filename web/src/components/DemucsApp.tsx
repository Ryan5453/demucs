import { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { useDemucs } from '../hooks/useDemucs';
import type { ModelType } from '../types';

interface ModelInfo {
    id: ModelType;
    name: string;
    stems: number;
}

const MODELS: ModelInfo[] = [
    { id: 'htdemucs', name: 'Demucs v4', stems: 4 },
    { id: 'htdemucs_6s', name: 'Demucs v4 (6-source)', stems: 6 },
    { id: 'hdemucs_mmi', name: 'Demucs v3', stems: 4 },
];

interface StemInfo {
    name: string;
    bg: string;
    accent: string;
    btnBg: string;
    hoverGlow: string;
}

const STEM_STYLES: Record<string, StemInfo> = {
    drums: { name: 'Drums', bg: '#2D2A1F', accent: '#FBBF24', btnBg: '#F59E0B', hoverGlow: 'rgba(251, 191, 36, 0.3)' },
    bass: { name: 'Bass', bg: '#1F2D20', accent: '#4ADE80', btnBg: '#22C55E', hoverGlow: 'rgba(74, 222, 128, 0.3)' },
    guitar: { name: 'Guitar', bg: '#2D1F1F', accent: '#FB7185', btnBg: '#F43F5E', hoverGlow: 'rgba(251, 113, 133, 0.3)' },
    piano: { name: 'Piano', bg: '#251F2D', accent: '#C084FC', btnBg: '#A855F7', hoverGlow: 'rgba(192, 132, 252, 0.3)' },
    other: { name: 'Other', bg: '#252528', accent: '#A1A1AA', btnBg: '#71717A', hoverGlow: 'rgba(161, 161, 170, 0.3)' },
    vocals: { name: 'Vocals', bg: '#2D1F25', accent: '#F472B6', btnBg: '#EC4899', hoverGlow: 'rgba(244, 114, 182, 0.3)' },
};

// Generate organic waveform heights
const generateWaveform = (seed: number) => {
    return Array.from({ length: 60 }).map((_, i) => {
        const base = Math.sin((i + seed) * 0.4) * 30;
        const noise = Math.sin((i + seed) * 1.3) * 15;
        const peak = i > 8 && i < 20 ? 10 : 0;
        return Math.max(20, Math.min(95, 50 + base + noise + peak));
    });
};

const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${String(secs).padStart(2, '0')}`;
};

export function DemucsApp() {
    const {
        modelLoaded,
        modelLoading,
        audioLoaded,
        audioBuffer,
        audioFile,
        separating,
        progress,
        status,
        stemUrls,
        stemWaveforms,
        loadModel,
        loadAudio,
        separateAudio,
    } = useDemucs();

    const fileInputRef = useRef<HTMLInputElement>(null);
    const [selectedModel, setSelectedModel] = useState<ModelType>('htdemucs');
    const [showModelMenu, setShowModelMenu] = useState(false);
    const [volumes, setVolumes] = useState<Record<string, number>>({});
    const [playingStems, setPlayingStems] = useState<Record<string, boolean>>({});
    const [currentTimes, setCurrentTimes] = useState<Record<string, number>>({});
    const [hoverInfo, setHoverInfo] = useState<{ stemKey: string; x: number; time: number } | null>(null);
    const audioRefs = useRef<Record<string, HTMLAudioElement>>({});
    const waveformRefs = useRef<Record<string, HTMLDivElement>>({});

    const currentModel = MODELS.find(m => m.id === selectedModel)!;
    const duration = audioBuffer?.duration ?? 0;
    const stems = Object.keys(stemUrls);
    const hasStemsReady = stems.length > 0;

    const handleModelSelect = (model: ModelType) => {
        setSelectedModel(model);
        setShowModelMenu(false);
        loadModel(model);
    };

    const handleFileClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            loadAudio(file);
        }
    };

    const togglePlay = (stemName: string) => {
        const audio = audioRefs.current[stemName];
        if (!audio) return;

        if (playingStems[stemName]) {
            audio.pause();
            setPlayingStems(prev => ({ ...prev, [stemName]: false }));
        } else {
            audio.play();
            setPlayingStems(prev => ({ ...prev, [stemName]: true }));
        }
    };

    const handleDownload = (stemName: string) => {
        const url = stemUrls[stemName];
        if (!url) return;
        const a = document.createElement('a');
        a.href = url;
        a.download = `${stemName}.wav`;
        a.click();
    };

    const handleDownloadAll = () => {
        stems.forEach((source, index) => {
            const url = stemUrls[source];
            if (url) {
                setTimeout(() => {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${source}.wav`;
                    a.click();
                }, index * 200);
            }
        });
    };

    return (
        <>
            <div className="parchment-overlay" />

            <div className="min-h-screen flex flex-col">
                <div className="flex-1 flex flex-col">
                    <div className="w-full max-w-4xl mx-auto p-8 flex-1">

                        {/* Header */}
                        <header className="text-center mb-10">
                            <h1
                                className="text-5xl font-bold text-slate-100 tracking-wide"
                                style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}
                            >
                                demucs.app
                            </h1>
                        </header>

                        {/* Controls Row */}
                        <div className="flex flex-wrap items-center justify-center gap-3 mb-10">
                            {/* Model Selector with loading indicator */}
                            <div className="model-dropdown">
                                <button
                                    onClick={() => !modelLoaded && !modelLoading && setShowModelMenu(!showModelMenu)}
                                    className={`model-dropdown-btn px-5 py-3 bg-slate-800/90 border-2 border-slate-600 rounded-2xl text-slate-100 font-semibold flex items-center gap-2 shadow-sm relative overflow-hidden ${modelLoaded || modelLoading ? 'cursor-default' : ''}`}
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
                                    {modelLoaded && (
                                        <svg className="w-4 h-4 text-sage-600 relative z-10" viewBox="0 0 24 24" fill="currentColor">
                                            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
                                        </svg>
                                    )}
                                    {!modelLoaded && !modelLoading && (
                                        <svg className={`w-4 h-4 ml-1 transition-transform relative z-10 ${showModelMenu ? 'rotate-180' : ''}`} viewBox="0 0 24 24" fill="currentColor">
                                            <path d="M7 10l5 5 5-5H7z" />
                                        </svg>
                                    )}
                                </button>
                                {showModelMenu && (
                                    <div className="model-dropdown-menu">
                                        {MODELS.map((model) => (
                                            <div
                                                key={model.id}
                                                className={`model-dropdown-item ${selectedModel === model.id ? 'selected' : ''}`}
                                                onClick={() => handleModelSelect(model.id)}
                                            >
                                                <div className="font-semibold text-slate-100">{model.name}</div>
                                                <div className="text-xs text-slate-400">{model.stems} stems</div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            <button
                                onClick={separateAudio}
                                disabled={!modelLoaded || !audioLoaded || separating}
                                className="btn-primary px-6 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {separating ? 'Separating...' : 'Separate Audio'}
                            </button>

                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="audio/*"
                                onChange={handleFileChange}
                                className="hidden"
                            />
                        </div>

                        {/* Vinyl Progress - Show when separating */}
                        {separating && (
                            <div className="flex flex-col items-center mb-10">
                                <div className="relative w-52 h-52 mb-5">
                                    {/* Progress ring */}
                                    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 208 208">
                                        <circle cx="104" cy="104" r="96" fill="none" stroke="#334155" strokeWidth="10" opacity="0.5" />
                                        <circle
                                            cx="104" cy="104" r="96"
                                            fill="none" stroke="#06B6D4" strokeWidth="10"
                                            strokeLinecap="round"
                                            strokeDasharray="603"
                                            strokeDashoffset={603 - (603 * progress / 100)}
                                            transform="rotate(-90 104 104)"
                                            style={{ filter: 'drop-shadow(0 0 6px rgba(6, 182, 212, 0.5))' }}
                                        />
                                    </svg>

                                    {/* Vinyl record */}
                                    <div
                                        className="absolute inset-3 rounded-full vinyl vinyl-rim"
                                        style={{ animation: 'vinyl-spin 3s linear infinite' }}
                                    >
                                        <div className="absolute inset-0 rounded-full vinyl-shine" />
                                        <div className="absolute inset-[12%] border border-[#2a2a2a] rounded-full" />
                                        <div className="absolute inset-[20%] border border-[#333] rounded-full" />
                                        <div className="absolute inset-[28%] border border-[#2a2a2a] rounded-full" />
                                        <div className="absolute inset-[36%] border border-[#333] rounded-full" />
                                        <div
                                            className="absolute inset-[38%] bg-gradient-to-br from-terracotta-400 via-terracotta-500 to-terracotta-700 rounded-full flex items-center justify-center"
                                            style={{ boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.3), inset 0 -1px 4px rgba(255,255,255,0.1)' }}
                                        >
                                            <div className="w-3 h-3 bg-brown-900 rounded-full shadow-inner" />
                                        </div>
                                    </div>
                                </div>

                                <span
                                    className="text-4xl font-bold text-slate-100"
                                    style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}
                                >
                                    {Math.round(progress)}%
                                </span>
                                <span className="text-slate-400 text-sm mt-1 font-medium">{status}</span>
                            </div>
                        )}

                        {/* File info - Show when audio loaded and not separating */}
                        {audioFile && !separating && (
                            <div className="text-center mb-8">
                                <p className="text-xl font-bold text-slate-100">{audioFile.name}</p>
                                <p className="text-slate-400 text-sm font-medium">
                                    {formatTime(duration)} • 44.1 kHz • Stereo
                                </p>
                            </div>
                        )}

                        {/* Stems Section */}
                        {hasStemsReady && (
                            <div className="mb-8">
                                <div className="flex items-center justify-center gap-4 mb-5">
                                    <div className="h-px flex-1 max-w-[80px] bg-gradient-to-r from-transparent to-slate-600" />
                                    <h2 className="text-slate-400 text-xs font-bold tracking-[0.25em] uppercase">
                                        Audio Stems
                                    </h2>
                                    <div className="h-px flex-1 max-w-[80px] bg-gradient-to-l from-transparent to-slate-600" />
                                </div>

                                <div className="grid gap-3">
                                    {stems.map((stemKey, stemIndex) => {
                                        const stemStyle = STEM_STYLES[stemKey] || STEM_STYLES.other;
                                        // Use real waveform if available, fallback to generated
                                        const waveform = stemWaveforms[stemKey] || generateWaveform(stemIndex * 7);
                                        const volume = volumes[stemKey] ?? 80;

                                        return (
                                            <div
                                                key={stemKey}
                                                className="stem-card rounded-3xl card-shadow"
                                                style={{ backgroundColor: stemStyle.bg }}
                                                onMouseEnter={(e) => {
                                                    (e.currentTarget as HTMLElement).style.boxShadow = `
                                                        0 4px 16px rgba(0, 0, 0, 0.3),
                                                        0 8px 32px rgba(0, 0, 0, 0.2),
                                                        0 0 0 1px ${stemStyle.hoverGlow},
                                                        inset 0 1px 0 rgba(255, 255, 255, 0.1)
                                                    `;
                                                }}
                                                onMouseLeave={(e) => {
                                                    (e.currentTarget as HTMLElement).style.boxShadow = '';
                                                }}
                                            >
                                                <audio
                                                    ref={(el) => { if (el) audioRefs.current[stemKey] = el; }}
                                                    src={stemUrls[stemKey]}
                                                    onEnded={() => setPlayingStems(prev => ({ ...prev, [stemKey]: false }))}
                                                    onTimeUpdate={(e) => {
                                                        const audio = e.currentTarget;
                                                        if (audio) {
                                                            setCurrentTimes(prev => ({ ...prev, [stemKey]: audio.currentTime }));
                                                        }
                                                    }}
                                                />

                                                <div className="p-4">
                                                    {/* Top row: Icon, Name, Controls */}
                                                    <div className="flex items-center gap-3">
                                                        {/* Icon */}
                                                        <div
                                                            className="w-10 h-10 sm:w-12 sm:h-12 rounded-2xl flex items-center justify-center flex-shrink-0"
                                                            style={{
                                                                background: `linear-gradient(145deg, ${stemStyle.btnBg}, ${stemStyle.accent})`,
                                                                boxShadow: `0 4px 12px ${stemStyle.hoverGlow}, inset 0 1px 0 rgba(255,255,255,0.2)`
                                                            }}
                                                        >
                                                            <svg className="w-5 h-5 sm:w-6 sm:h-6 text-white drop-shadow-sm" viewBox="0 0 24 24" fill="currentColor">
                                                                <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
                                                            </svg>
                                                        </div>

                                                        {/* Name & Duration */}
                                                        <div className="w-20 flex-shrink-0">
                                                            <p className="text-base sm:text-lg font-bold text-slate-100 truncate">{stemStyle.name}</p>
                                                            <p className="text-slate-400 text-xs font-medium">{formatTime(duration)}</p>
                                                        </div>

                                                        {/* Desktop waveform - hidden on mobile */}
                                                        <div
                                                            ref={(el) => { if (el) waveformRefs.current[stemKey] = el; }}
                                                            className="hidden md:flex flex-1 relative items-center justify-center gap-[3px] h-10 cursor-pointer overflow-visible"
                                                            onClick={(e) => {
                                                                const rect = e.currentTarget.getBoundingClientRect();
                                                                const x = e.clientX - rect.left;
                                                                const percent = x / rect.width;
                                                                const seekTime = percent * duration;
                                                                const audio = audioRefs.current[stemKey];
                                                                if (audio) {
                                                                    audio.currentTime = seekTime;
                                                                }
                                                            }}
                                                            onMouseMove={(e) => {
                                                                const rect = e.currentTarget.getBoundingClientRect();
                                                                const x = e.clientX - rect.left;
                                                                const percent = Math.max(0, Math.min(1, x / rect.width));
                                                                const time = percent * duration;
                                                                setHoverInfo({ stemKey, x, time });
                                                            }}
                                                            onMouseLeave={() => setHoverInfo(null)}
                                                        >
                                                            {/* Tooltip */}
                                                            {hoverInfo && hoverInfo.stemKey === stemKey && (
                                                                <div
                                                                    className="absolute -top-8 px-2 py-1 bg-slate-700 text-white text-xs rounded shadow-lg whitespace-nowrap z-10 pointer-events-none"
                                                                    style={{ left: hoverInfo.x, transform: 'translateX(-50%)' }}
                                                                >
                                                                    {formatTime(hoverInfo.time)}
                                                                </div>
                                                            )}

                                                            {/* Playhead indicator */}
                                                            {duration > 0 && (
                                                                <div
                                                                    className="absolute top-0 bottom-0 w-0.5 bg-slate-100 z-10 pointer-events-none"
                                                                    style={{ left: `${((currentTimes[stemKey] || 0) / duration) * 100}%` }}
                                                                />
                                                            )}

                                                            {/* Waveform bars */}
                                                            {waveform.map((height, i) => (
                                                                <div
                                                                    key={i}
                                                                    className="waveform-bar flex-1 rounded-full"
                                                                    style={{
                                                                        height: `${height}%`,
                                                                        backgroundColor: stemStyle.accent,
                                                                        opacity: 0.5 + (height / 200)
                                                                    }}
                                                                />
                                                            ))}
                                                        </div>

                                                        {/* Volume slider - hidden on mobile */}
                                                        <div className="hidden sm:flex items-center gap-2 w-28 lg:w-36">
                                                            <svg className="w-4 h-4 text-slate-400 flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
                                                                <path d="M3 9v6h4l5 5V4L7 9H3z" />
                                                            </svg>
                                                            <input
                                                                type="range"
                                                                value={volume}
                                                                onChange={(e) => setVolumes(prev => ({ ...prev, [stemKey]: Number(e.target.value) }))}
                                                                className="flex-1"
                                                                style={{
                                                                    background: `linear-gradient(90deg, ${stemStyle.accent} ${volume}%, #334155 ${volume}%)`
                                                                }}
                                                            />
                                                        </div>

                                                        {/* Play Button */}
                                                        <button
                                                            onClick={() => togglePlay(stemKey)}
                                                            className="btn-play w-10 h-10 sm:w-11 sm:h-11 rounded-2xl flex items-center justify-center flex-shrink-0"
                                                            style={{
                                                                background: `linear-gradient(145deg, ${stemStyle.btnBg}, ${stemStyle.accent})`,
                                                                boxShadow: `0 3px 10px ${stemStyle.hoverGlow}`
                                                            }}
                                                        >
                                                            {playingStems[stemKey] ? (
                                                                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white drop-shadow-sm" fill="currentColor" viewBox="0 0 24 24">
                                                                    <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z" />
                                                                </svg>
                                                            ) : (
                                                                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-white ml-0.5 drop-shadow-sm" fill="currentColor" viewBox="0 0 24 24">
                                                                    <path d="M8 5v14l11-7z" />
                                                                </svg>
                                                            )}
                                                        </button>

                                                        {/* Download Button */}
                                                        <button
                                                            onClick={() => handleDownload(stemKey)}
                                                            className="btn-download w-10 h-10 sm:w-11 sm:h-11 bg-slate-700/80 hover:bg-slate-600 rounded-2xl flex items-center justify-center flex-shrink-0 border border-slate-600 shadow-sm"
                                                        >
                                                            <svg className="w-4 h-4 sm:w-5 sm:h-5 text-slate-200" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                                                                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                                                            </svg>
                                                        </button>
                                                    </div>

                                                    {/* Mobile waveform - shown only on mobile, full width */}
                                                    <div
                                                        className="md:hidden relative flex items-center gap-[2px] h-8 mt-3 cursor-pointer overflow-visible"
                                                        onClick={(e) => {
                                                            const rect = e.currentTarget.getBoundingClientRect();
                                                            const x = e.clientX - rect.left;
                                                            const percent = x / rect.width;
                                                            const seekTime = percent * duration;
                                                            const audio = audioRefs.current[stemKey];
                                                            if (audio) {
                                                                audio.currentTime = seekTime;
                                                            }
                                                        }}
                                                    >
                                                        {/* Playhead indicator */}
                                                        {duration > 0 && (
                                                            <div
                                                                className="absolute top-0 bottom-0 w-0.5 bg-slate-100 z-10 pointer-events-none"
                                                                style={{ left: `${((currentTimes[stemKey] || 0) / duration) * 100}%` }}
                                                            />
                                                        )}

                                                        {/* Waveform bars */}
                                                        {waveform.map((height, i) => (
                                                            <div
                                                                key={i}
                                                                className="waveform-bar flex-1 rounded-full"
                                                                style={{
                                                                    height: `${height}%`,
                                                                    backgroundColor: stemStyle.accent,
                                                                    opacity: 0.5 + (height / 200)
                                                                }}
                                                            />
                                                        ))}
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Clickable Drop Zone - No audio yet */}
                        {!hasStemsReady && !separating && !audioFile && (
                            <div
                                onClick={handleFileClick}
                                className="drop-zone cursor-pointer text-center py-20 mb-8 rounded-3xl border-2 border-dashed border-slate-600 bg-slate-800/30 hover:bg-slate-700/40 hover:border-slate-500 transition-all"
                            >
                                <div className="relative w-32 h-32 mx-auto mb-6">
                                    {/* Static vinyl for drop zone */}
                                    <div className="absolute inset-0 rounded-full vinyl vinyl-rim opacity-80">
                                        <div className="absolute inset-0 rounded-full vinyl-shine" />
                                        <div className="absolute inset-[12%] border border-[#2a2a2a] rounded-full" />
                                        <div className="absolute inset-[20%] border border-[#333] rounded-full" />
                                        <div className="absolute inset-[28%] border border-[#2a2a2a] rounded-full" />
                                        <div className="absolute inset-[36%] border border-[#333] rounded-full" />
                                        <div
                                            className="absolute inset-[38%] bg-gradient-to-br from-cyan-400 via-cyan-500 to-cyan-700 rounded-full flex items-center justify-center"
                                            style={{ boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.3), inset 0 -1px 4px rgba(255,255,255,0.1)' }}
                                        >
                                            <div className="w-2 h-2 bg-slate-900 rounded-full shadow-inner" />
                                        </div>
                                    </div>
                                </div>
                                <p className="text-slate-200 font-semibold text-lg">Click to load an audio file</p>
                                <p className="text-slate-400 text-sm mt-1">Supports MP3, WAV, FLAC, and more</p>
                            </div>
                        )}

                        {/* Audio loaded but not separated yet */}
                        {!hasStemsReady && !separating && audioFile && (
                            <div className="text-center py-8 mb-8">
                                {modelLoaded ? (
                                    <>
                                        <p className="text-slate-300 font-medium">Ready to separate</p>
                                        <p className="text-slate-500 text-sm mt-1">Click "Separate Audio" to extract stems</p>
                                    </>
                                ) : (
                                    <>
                                        <p className="text-slate-300 font-medium">Select a model first</p>
                                        <p className="text-slate-500 text-sm mt-1">Choose a model from the dropdown above</p>
                                    </>
                                )}
                            </div>
                        )}

                        {/* Download All */}
                        {hasStemsReady && (
                            <div className="text-center mb-12">
                                <button
                                    onClick={handleDownloadAll}
                                    className="px-6 py-2.5 bg-gradient-to-b from-terracotta-400 to-terracotta-600 hover:from-terracotta-400 hover:to-terracotta-500 text-white font-semibold rounded-xl shadow-md transition-all hover:shadow-lg hover:-translate-y-0.5"
                                    style={{ boxShadow: '0 4px 12px rgba(152, 80, 48, 0.3)' }}
                                >
                                    <span className="flex items-center gap-2">
                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
                                            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
                                        </svg>
                                        Download All
                                    </span>
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                <footer className="w-full mt-auto bg-slate-800/90 py-3">
                    <nav className="flex items-center justify-center gap-6 text-xs font-medium text-slate-400">
                        <Link to="/" className="footer-link hover:text-cyan-400">Home</Link>
                        <Link to="/about" className="footer-link hover:text-cyan-400">About</Link>
                        <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer" className="footer-link hover:text-cyan-400">GitHub</a>
                        <Link to="/privacy" className="footer-link hover:text-cyan-400">Privacy</Link>
                    </nav>
                </footer>
            </div>
        </>
    );
}
