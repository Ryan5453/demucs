import React from 'react';

interface VinylProps {
    className?: string;
    style?: React.CSSProperties;
    progress?: number; // 0 to 100, if provided renders the progress ring
    variant?: 'terracotta' | 'cyan';
    animate?: boolean;
    artworkUrl?: string | null;
}

export const Vinyl: React.FC<VinylProps> = ({
    className = '',
    style = {},
    progress,
    variant = 'terracotta',
    animate = true,
    artworkUrl
}) => {
    const labelGradient = variant === 'terracotta'
        ? 'from-terracotta-400 via-terracotta-500 to-terracotta-700'
        : 'from-cyan-400 via-cyan-500 to-cyan-700';

    const spindleColor = variant === 'terracotta' ? 'bg-brown-900' : 'bg-slate-900';
    const spindleSize = variant === 'terracotta' ? 'w-3 h-3' : 'w-2 h-2'; // Matching the slight diff in original

    // If showing progress ring, we need space for it (inset-3), otherwise full size (inset-0)
    const vinylInset = progress !== undefined ? 'inset-3' : 'inset-0';

    return (
        <div className={`relative ${className}`} style={style}>
            {/* Progress ring - only if progress is defined */}
            {progress !== undefined && (
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 208 208">
                    <circle cx="104" cy="104" r="96" fill="none" stroke="#334155" strokeWidth="10" opacity="0.5" />
                    <circle
                        cx="104" cy="104" r="96"
                        fill="none" stroke="#06B6D4" strokeWidth="10"
                        strokeLinecap="round"
                        strokeDasharray="603"
                        strokeDashoffset={603 - (603 * progress / 100)}
                        transform="rotate(-90 104 104)"
                    />
                </svg>
            )}

            {/* Vinyl record */}
            <div
                className={`absolute ${vinylInset} rounded-full vinyl vinyl-rim`}
                style={{ animation: animate ? 'vinyl-spin 3s linear infinite' : 'none' }}
            >
                <div className="absolute inset-0 rounded-full vinyl-shine" />
                <div className="absolute inset-[12%] border border-[#2a2a2a] rounded-full" />
                <div className="absolute inset-[20%] border border-[#333] rounded-full" />
                <div className="absolute inset-[28%] border border-[#2a2a2a] rounded-full" />
                <div className="absolute inset-[36%] border border-[#333] rounded-full" />

                {/* Center Label - artwork or gradient fallback */}
                {artworkUrl ? (
                    <div
                        className="absolute inset-[38%] rounded-full flex items-center justify-center overflow-hidden"
                        style={{ boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.3), inset 0 -1px 4px rgba(255,255,255,0.1)' }}
                    >
                        <img
                            src={artworkUrl}
                            alt="Album artwork"
                            className="absolute inset-0 w-full h-full object-cover"
                        />
                        <div className={`${spindleSize} ${spindleColor} rounded-full shadow-inner relative z-10`} />
                    </div>
                ) : (
                    <div
                        className={`absolute inset-[38%] bg-gradient-to-br ${labelGradient} rounded-full flex items-center justify-center`}
                        style={{ boxShadow: 'inset 0 2px 8px rgba(0,0,0,0.3), inset 0 -1px 4px rgba(255,255,255,0.1)' }}
                    >
                        <div className={`${spindleSize} ${spindleColor} rounded-full shadow-inner`} />
                    </div>
                )}
            </div>
        </div>
    );
};
