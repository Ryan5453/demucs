// eslint-disable-next-line @typescript-eslint/no-unused-vars

interface LEDDisplayProps {
    text: string;
    scroll?: boolean;
    height?: number;
}

export function LEDDisplay({ text, scroll = false, height = 16 }: LEDDisplayProps) {
    return (
        <div className="led-display" style={{ height: `${height}px`, padding: '2px 4px' }}>
            <span className={`led-text text-xs ${scroll ? '' : 'no-scroll'}`}>
                {text}
            </span>
        </div>
    );
}

interface LEDTimeProps {
    seconds: number;
}

export function LEDTime({ seconds }: LEDTimeProps) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return (
        <div className="inset-panel flex items-center justify-center px-2 py-1">
            <span className="led-time">
                {String(mins).padStart(2, '0')}:{String(secs).padStart(2, '0')}
            </span>
        </div>
    );
}

interface StatusLEDProps {
    status: 'off' | 'on' | 'loading' | 'error';
    label?: string;
}

export function StatusLED({ status, label }: StatusLEDProps) {
    return (
        <div className="flex items-center gap-1">
            <div className={`status-led ${status}`} />
            {label && <span className="text-[8px] text-gray-400 uppercase">{label}</span>}
        </div>
    );
}

interface ProgressBarProps {
    progress: number;
}

export function ProgressBar({ progress }: ProgressBarProps) {
    return (
        <div className="winamp-progress w-full">
            <div className="winamp-progress-fill" style={{ width: `${progress}%` }} />
        </div>
    );
}

interface VolumeSliderProps {
    value: number;
    onChange: (value: number) => void;
    label: string;
}

export function VolumeSlider({ value, onChange, label }: VolumeSliderProps) {
    return (
        <div className="stem-slider-container">
            <span className="stem-slider-label">{label}</span>
            <input
                type="range"
                min="0"
                max="100"
                value={value}
                onChange={(e) => onChange(Number(e.target.value))}
                className="stem-slider"
            />
        </div>
    );
}
