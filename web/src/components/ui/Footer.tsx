import { Link } from 'react-router-dom';

export default function Footer() {
    return (
        <footer className="w-full mt-auto bg-slate-800/90 py-3">
            <nav className="flex items-center justify-center gap-6 text-xs font-medium text-slate-400">
                <Link to="/" className="footer-link hover:text-cyan-400">Home</Link>
                <Link to="/about" className="footer-link hover:text-cyan-400">About</Link>
                <Link to="/privacy" className="footer-link hover:text-cyan-400">Privacy</Link>
                <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer" className="footer-link hover:text-cyan-400">GitHub</a>
            </nav>
        </footer>
    );
}
