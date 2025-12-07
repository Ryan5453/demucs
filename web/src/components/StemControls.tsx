import type { SourceName } from '../types';
import { SOURCES } from '../types';
import { StemPlayer } from './StemPlayer';
import { Music2, Guitar, Piano, Mic } from 'lucide-react';
import type { ReactNode } from 'react';

interface StemControlsProps {
    stemUrls: Record<string, string>;
}

export function StemControls({ stemUrls }: StemControlsProps) {
    if (Object.keys(stemUrls).length === 0) {
        return (
            <div className="inset-panel p-6 text-center">
                <span className="text-sm text-gray-500 uppercase">
                    Separate audio to see stems
                </span>
            </div>
        );
    }

    const stemColors: Record<SourceName, string> = {
        drums: '#ff6b6b',
        bass: '#4dabf7',
        other: '#69db7c',
        vocals: '#ffd43b',
    };

    const stemIcons: Record<SourceName, ReactNode> = {
        drums: <Music2 size={16} />,
        bass: <Guitar size={16} />,
        other: <Piano size={16} />,
        vocals: <Mic size={16} />,
    };

    return (
        <div className="flex flex-col gap-[2px]">
            {SOURCES.map((source) => (
                <StemPlayer
                    key={source}
                    name={source}
                    color={stemColors[source]}
                    icon={stemIcons[source]}
                    audioUrl={stemUrls[source]}
                />
            ))}
        </div>
    );
}
