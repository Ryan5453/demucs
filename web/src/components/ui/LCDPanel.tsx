interface InfoPanelProps {
    children: React.ReactNode;
    className?: string;
}

export function InfoPanel({ children, className = '' }: InfoPanelProps) {
    return (
        <div className={`info-panel ${className}`}>
            {children}
        </div>
    );
}

interface InfoTimeProps {
    seconds: number;
    className?: string;
}

export function InfoTime({ seconds, className = '' }: InfoTimeProps) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    return (
        <span className={`info-time ${className}`}>
            {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
        </span>
    );
}

interface InfoTitleProps {
    text: string;
    className?: string;
}

export function InfoTitle({ text, className = '' }: InfoTitleProps) {
    return (
        <span className={`info-title ${className}`}>
            {text}
        </span>
    );
}

interface InfoStatusProps {
    text: string;
    className?: string;
}

export function InfoStatus({ text, className = '' }: InfoStatusProps) {
    return (
        <span className={`info-status ${className}`}>
            {text}
        </span>
    );
}

interface AquaProgressBarProps {
    progress: number;
    className?: string;
}

export function AquaProgressBar({ progress, className = '' }: AquaProgressBarProps) {
    return (
        <div className={`aqua-progress ${className}`}>
            <div
                className="aqua-progress-fill"
                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
        </div>
    );
}

interface StatusDotProps {
    status: 'off' | 'on' | 'loading' | 'error';
    label?: string;
}

export function StatusDot({ status, label }: StatusDotProps) {
    return (
        <div className="flex items-center gap-1.5">
            <div className={`aqua-status-dot ${status}`} />
            {label && (
                <span className="text-[11px] text-gray-600">{label}</span>
            )}
        </div>
    );
}
