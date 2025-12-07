

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'primary' | 'danger';
}

export function Button({
    children,
    variant = 'default',
    className = '',
    ...props
}: ButtonProps) {
    const variantClass = variant === 'default' ? '' : variant;

    return (
        <button
            className={`winamp-btn ${variantClass} ${className}`}
            {...props}
        >
            {children}
        </button>
    );
}

interface TransportButtonProps {
    icon: 'play' | 'stop' | 'load' | 'eject';
    onClick?: () => void;
    disabled?: boolean;
}

export function TransportButton({ icon, onClick, disabled }: TransportButtonProps) {
    const icons = {
        play: (
            <svg viewBox="0 0 10 10">
                <polygon points="2,1 9,5 2,9" />
            </svg>
        ),
        stop: (
            <svg viewBox="0 0 10 10">
                <rect x="2" y="2" width="6" height="6" />
            </svg>
        ),
        load: (
            <svg viewBox="0 0 10 10">
                <path d="M1,8 L5,8 L5,5 L7,5 L7,8 L9,8 L9,9 L1,9 Z" />
                <path d="M5,1 L5,4 L3,4 L5,1 L7,4 L5,4" />
            </svg>
        ),
        eject: (
            <svg viewBox="0 0 10 10">
                <polygon points="5,1 9,5 1,5" />
                <rect x="1" y="7" width="8" height="2" />
            </svg>
        ),
    };

    return (
        <button
            className="transport-btn"
            onClick={onClick}
            disabled={disabled}
        >
            {icons[icon]}
        </button>
    );
}
