import { Link } from 'react-router-dom';

export function About() {
    return (
        <>
            <div className="parchment-overlay" />

            <div className="min-h-screen flex flex-col">
                <div className="flex-1 flex flex-col">
                    <div className="w-full max-w-4xl mx-auto p-8 flex-1">
                        {/* Header */}
                        <header className="text-center mb-10">
                            <Link to="/">
                                <h1
                                    className="text-5xl font-bold text-brown-800 tracking-wide hover:opacity-80 transition-opacity cursor-pointer"
                                    style={{ textShadow: '0 1px 0 rgba(255,255,255,0.3)' }}
                                >
                                    demucs.app
                                </h1>
                            </Link>
                        </header>

                        {/* Content */}
                        <div className="bg-white/60 rounded-3xl p-8 card-shadow">
                            <h2 className="text-3xl font-bold text-brown-800 mb-6">About</h2>

                            <div className="space-y-4 text-brown-700">
                                <p>
                                    <strong>demucs.app</strong> is a free, browser-based audio stem separation tool powered by
                                    Meta's Demucs AI model. It runs entirely in your browser using WebAssembly and ONNX Runtime,
                                    meaning your audio files never leave your device.
                                </p>

                                <h3 className="text-xl font-semibold text-brown-800 mt-6 mb-3">How It Works</h3>
                                <p>
                                    Demucs uses deep learning to separate mixed audio into individual stems: drums, bass,
                                    vocals, and other instruments.
                                </p>
                                <p className="mt-2">
                                    The 6-source model is experimental, adding guitar and piano stems. Testing shows
                                    reasonable quality for guitar, but the piano source may have noticeable bleeding and artifacts.
                                </p>

                                <h3 className="text-xl font-semibold text-brown-800 mt-6 mb-3">Privacy</h3>
                                <p>
                                    All audio processing happens locally in your browser. Your files are never uploaded to
                                    any server. We don't collect any personal data or use analytics.
                                </p>

                                <h3 className="text-xl font-semibold text-brown-800 mt-6 mb-3">Credits</h3>
                                <p>
                                    This project uses the <a href="https://github.com/facebookresearch/demucs" target="_blank" rel="noopener noreferrer" className="text-sage-600 hover:underline">Demucs</a> model
                                    created by Meta AI Research. The web interface is open source and available on <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer" className="text-sage-600 hover:underline">GitHub</a>.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <footer className="w-full mt-auto bg-parchment-600/90 py-3">
                    <nav className="flex items-center justify-center gap-6 text-xs font-medium text-parchment-100/80">
                        <Link to="/" className="footer-link hover:text-white">Home</Link>
                        <Link to="/about" className="footer-link hover:text-white">About</Link>
                        <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer" className="footer-link hover:text-white">GitHub</a>
                        <Link to="/privacy" className="footer-link hover:text-white">Privacy</Link>
                    </nav>
                </footer>
            </div>
        </>
    );
}
