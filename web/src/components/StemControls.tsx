import type { SourceName } from '../types';
import { SOURCES } from '../types';
import { StemPlayer } from './StemPlayer';
import { Download } from 'lucide-react';

interface StemControlsProps {
    stemUrls: Record<string, string>;
}

export function StemControls({ stemUrls }: StemControlsProps) {
    const hasStemsReady = Object.keys(stemUrls).length > 0;

    const handleDownloadAll = () => {
        SOURCES.forEach((source, index) => {
            const url = stemUrls[source];
            if (url) {
                // Stagger downloads slightly to avoid browser blocking
                setTimeout(() => {
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `${source}.wav`;
                    a.click();
                }, index * 200);
            }
        });
    };

    if (!hasStemsReady) {
        return (
            <div className="aqua-inset-panel text-center py-4">
                <span className="text-[12px] text-gray-500">
                    Separate audio to view stems
                </span>
            </div>
        );
    }

    const stemLabels: Record<SourceName, string> = {
        drums: 'Drums',
        bass: 'Bass',
        other: 'Other',
        vocals: 'Vocals',
    };

    return (
        <div className="flex flex-col gap-2">
            <table className="tiger-table">
                <thead>
                    <tr>
                        <th style={{ width: '30px' }}>#</th>
                        <th>Stem</th>
                        <th style={{ width: '50px' }}>Time</th>
                        <th style={{ width: '40px' }}></th>
                        <th style={{ width: '40px' }}></th>
                    </tr>
                </thead>
                <tbody>
                    {SOURCES.map((source, index) => (
                        <StemPlayer
                            key={source}
                            index={index + 1}
                            name={source}
                            label={stemLabels[source]}
                            audioUrl={stemUrls[source]}
                        />
                    ))}
                </tbody>
            </table>

            <div className="flex justify-end">
                <button
                    onClick={handleDownloadAll}
                    className="tiger-transport-btn"
                    style={{ width: 'auto', height: 'auto', padding: '4px 10px', gap: '4px' }}
                >
                    <Download size={12} />
                    <span className="text-[11px]">Download All</span>
                </button>
            </div>
        </div>
    );
}
