import { useRef } from 'react';
import { LEDDisplay, LEDTime, StatusLED, ProgressBar } from './ui/LEDDisplay';
import { Button } from './ui/Button';
import { Check, Loader2 } from 'lucide-react';


interface MainPlayerProps {
    fileName: string | null;
    status: string;
    progress: number;
    duration: number;
    modelLoaded: boolean;
    modelLoading: boolean;
    audioLoaded: boolean;
    separating: boolean;
    onLoadModel: () => void;
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

    const handleFileClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            onLoadAudio(file);
        }
    };

    const displayText = fileName || 'LOAD AUDIO TO BEGIN';
    const modelStatus = modelLoading ? 'loading' : modelLoaded ? 'on' : 'off';

    return (
        <div className="flex flex-col gap-3">
            {/* Top row: clutter/mono/stereo indicators, EQ, PL */}
            <div className="flex items-center justify-between px-2">
                <div className="flex gap-2">
                    <StatusLED status={modelStatus} label="MODEL" />
                    <StatusLED status={audioLoaded ? 'on' : 'off'} label="AUDIO" />
                </div>
            </div>

            {/* Main display area */}
            <div>
                <div className="flex gap-2">
                    {/* Time display */}
                    <LEDTime seconds={duration} />

                    {/* Info display with scrolling text */}
                    <div className="flex-1">
                        <LEDDisplay text={displayText} scroll={fileName !== null} height={50} />
                    </div>
                </div>
            </div>

            {/* Status bar */}
            <div className="flex items-center gap-3">
                <span className="text-sm text-[var(--led-green)] flex-1 truncate">{status}</span>
                <ProgressBar progress={progress} />
            </div>

            {/* Transport controls */}
            <div className="flex items-center gap-2 mt-2 flex-wrap justify-center sm:justify-start">
                <Button
                    onClick={onLoadModel}
                    disabled={modelLoaded || modelLoading}
                    variant={modelLoaded ? 'default' : 'primary'}
                    className="w-40 flex justify-center items-center whitespace-nowrap"
                >
                    {modelLoading ? (
                        <>
                            <Loader2 className="animate-spin mr-2" size={14} />
                            LOADING
                        </>
                    ) : modelLoaded ? (
                        <>
                            <Check className="mr-2" size={14} />
                            MODEL
                        </>
                    ) : 'LOAD MODEL'}
                </Button>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    className="hidden"
                />

                <Button
                    onClick={handleFileClick}
                    className="w-40 flex justify-center items-center whitespace-nowrap"
                >
                    LOAD AUDIO
                </Button>

                <Button
                    onClick={onSeparate}
                    disabled={!modelLoaded || !audioLoaded || separating}
                    variant="primary"
                    className="w-40 flex justify-center items-center whitespace-nowrap"
                >
                    {separating ? (
                        <>
                            <Loader2 className="animate-spin mr-2" size={14} />
                            SEPARATING
                        </>
                    ) : 'SEPARATE'}
                </Button>
            </div>
        </div>
    );
}

