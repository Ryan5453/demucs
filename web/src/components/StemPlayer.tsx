import { useState, useRef } from 'react';
import type { SourceName } from '../types';
import { Play, Pause, Save } from 'lucide-react';

interface StemPlayerProps {
    name: SourceName;
    color: string;
    icon: React.ReactNode;
    audioUrl?: string;
}

export function StemPlayer({ name, color, icon, audioUrl }: StemPlayerProps) {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);

    const togglePlay = async () => {
        if (!audioRef.current || !audioUrl) return;

        if (isPlaying) {
            audioRef.current.pause();
            setIsPlaying(false);
        } else {
            try {
                await audioRef.current.play();
                setIsPlaying(true);
            } catch (e) {
                console.error("Error playing audio:", e);
            }
        }
    };

    const handleTimeUpdate = () => {
        if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime);
        }
    };

    const handleLoadedMetadata = () => {
        if (audioRef.current) {
            setDuration(audioRef.current.duration);
        }
    };

    const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (audioRef.current) {
            const time = parseFloat(e.target.value);
            audioRef.current.currentTime = time;
            setCurrentTime(time);
        }
    };

    const handleDownload = () => {
        if (!audioUrl) return;
        const a = document.createElement('a');
        a.href = audioUrl;
        a.download = `${name}.wav`;
        a.click();
    };

    const formatTime = (time: number) => {
        const mins = Math.floor(time / 60);
        const secs = Math.floor(time % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="bg-[#1a1a1a] border border-[#555] p-1 mb-1 font-mono text-xs shadow-[inset_-1px_-1px_1px_rgba(255,255,255,0.2),inset_1px_1px_1px_rgba(0,0,0,0.8)]">
            <audio
                ref={audioRef}
                src={audioUrl}
                onEnded={() => setIsPlaying(false)}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
            />

            {/* Top Row: Controls + Info */}
            <div className="flex items-center gap-2 mb-1">
                {/* Play/Pause Button */}
                <button
                    onClick={togglePlay}
                    disabled={!audioUrl}
                    className={`w-6 h-6 flex items-center justify-center border border-[#555] active:border-[#333] ${isPlaying ? 'bg-[#0f0] text-black shadow-[inset_1px_1px_2px_rgba(255,255,255,0.5)]' : 'bg-[#333] text-[#0f0] shadow-[1px_1px_0px_#000]'}`}
                >
                    {isPlaying ? <Pause size={14} fill="currentColor" /> : <Play size={14} fill="currentColor" />}
                </button>

                {/* Title and Time - WINAMP GREEN */}
                <div className="flex-1 bg-black border border-[#555] px-2 py-0.5 text-[#0f0] flex justify-between items-center shadow-[inset_1px_1px_2px_rgba(0,0,0,0.8)]">
                    <div className="flex items-center gap-2">
                        {icon}
                        <span className="uppercase tracking-widest" style={{ textShadow: `0 0 2px ${color}` }}>
                            {name}
                        </span>
                    </div>
                    <span>{formatTime(currentTime)} / {formatTime(duration)}</span>
                </div>

                {/* Download Button */}
                <button
                    onClick={handleDownload}
                    disabled={!audioUrl}
                    className="w-6 h-6 bg-[#333] text-[#0f0] border border-[#555] flex items-center justify-center hover:brightness-110 active:translate-y-[1px] shadow-[1px_1px_0px_#000]"
                    title="Save WAV"
                >
                    <Save size={14} />
                </button>
            </div>

            {/* Bottom Row: Seek Bar */}
            <div className="px-0.5 pb-0.5">
                <input
                    type="range"
                    min={0}
                    max={duration || 100}
                    value={currentTime}
                    onChange={handleSeek}
                    className="winamp-slider h-3"
                    disabled={!audioUrl}
                />
            </div>
        </div>
    );
}

