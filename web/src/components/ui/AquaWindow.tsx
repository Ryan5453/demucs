import type { ReactNode } from 'react';

interface AquaWindowProps {
    title: string;
    children: ReactNode;
    width?: number;
    showToolbar?: boolean;
    toolbar?: ReactNode;
}

export function AquaWindow({
    title,
    children,
    width = 550,
    showToolbar = false,
    toolbar
}: AquaWindowProps) {
    return (
        <div className="aqua-window" style={{ width: '100%', maxWidth: `${width}px` }}>
            <div className="aqua-titlebar">
                <div className="aqua-traffic-lights">
                    <span className="aqua-traffic-btn close" />
                    <span className="aqua-traffic-btn minimize" />
                    <span className="aqua-traffic-btn maximize" />
                </div>
                <span className="aqua-titlebar-text">{title}</span>
            </div>
            {showToolbar && toolbar && (
                <div className="aqua-toolbar">
                    {toolbar}
                </div>
            )}
            <div className="aqua-window-content">
                {children}
            </div>
        </div>
    );
}
