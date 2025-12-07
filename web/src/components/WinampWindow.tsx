import type { ReactNode } from 'react';

interface WinampWindowProps {
    title: string;
    children: ReactNode;
    width?: number;
}

import { Minus, X } from 'lucide-react';

export function WinampWindow({ title, children, width = 500 }: WinampWindowProps) {
    return (
        <div className="winamp-window" style={{ width: '100%', maxWidth: `${width}px` }}>
            <div className="winamp-titlebar">
                <span className="winamp-titlebar-text">{title}</span>
                <div className="winamp-titlebar-buttons">
                    <button className="winamp-titlebar-btn" aria-label="Minimize">
                        <Minus size={10} strokeWidth={4} />
                    </button>
                    <button className="winamp-titlebar-btn" aria-label="Close">
                        <X size={10} strokeWidth={4} />
                    </button>
                </div>
            </div>
            <div className="p-2">
                {children}
            </div>
        </div>
    );
}
