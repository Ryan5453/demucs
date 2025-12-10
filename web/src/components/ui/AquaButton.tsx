import type { ButtonHTMLAttributes, ReactNode } from 'react';

interface AquaButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'primary' | 'danger';
    children: ReactNode;
}

export function AquaButton({
    children,
    variant = 'default',
    className = '',
    ...props
}: AquaButtonProps) {
    const variantClass = variant === 'default' ? '' : variant;

    return (
        <button
            className={`aqua-btn ${variantClass} ${className}`}
            {...props}
        >
            {children}
        </button>
    );
}

interface AquaToolbarButtonProps {
    icon: ReactNode;
    label: string;
    onClick?: () => void;
    disabled?: boolean;
    active?: boolean;
}

export function AquaToolbarButton({
    icon,
    label,
    onClick,
    disabled = false,
    active = false
}: AquaToolbarButtonProps) {
    return (
        <button
            className={`aqua-toolbar-btn ${active ? 'active' : ''}`}
            onClick={onClick}
            disabled={disabled}
        >
            <span className="aqua-toolbar-btn-icon">{icon}</span>
            <span className="aqua-toolbar-btn-label">{label}</span>
        </button>
    );
}

export function AquaToolbarDivider() {
    return <div className="aqua-toolbar-divider" />;
}
