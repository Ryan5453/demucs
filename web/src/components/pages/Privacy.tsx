import { Link } from 'react-router-dom';

export function Privacy() {
    return (
        <div className="w-full max-w-4xl mx-auto p-8 flex-1">
            {/* Header */}
            <header className="text-center mb-10">
                <Link to="/">
                    <h1
                        className="text-5xl font-bold text-slate-100 tracking-wide hover:opacity-80 transition-opacity cursor-pointer"
                        style={{ textShadow: '0 2px 4px rgba(0,0,0,0.3)' }}
                    >
                        demucs.app
                    </h1>
                </Link>
            </header>

            {/* Content */}
            <div className="bg-slate-800/60 rounded-3xl p-8 card-shadow">
                <h2 className="text-3xl font-bold text-slate-100 mb-6">Privacy Policy</h2>

                <div className="space-y-4 text-slate-300">
                    <p>
                        <strong>Effective Date:</strong> December 2024
                    </p>

                    <h3 className="text-xl font-semibold text-slate-100 mt-6 mb-3">Data Collection</h3>
                    <p>
                        <strong>We don't collect any data.</strong> This application runs entirely in your browser.
                        Your audio files are processed locally on your device and are <b>never</b> uploaded to any server.</p>

                    <h3 className="text-xl font-semibold text-slate-100 mt-6 mb-3">Local Processing</h3>
                    <p>
                        All audio separation is performed using WebGPU / the ONNX Web Runtime directly in your browser.
                        The model is downloaded once and cached locally by your browser. No audio data ever leaves your device.
                    </p>

                    <h3 className="text-xl font-semibold text-slate-100 mt-6 mb-3">Cookies & Analytics</h3>
                    <p>
                        We do not use cookies or any in-app tracking technologies. This site is hosted on
                        Cloudflare Pages, which collects basic, privacy-respecting web analytics (e.g., page views)
                        at the infrastructure level. No personal data is collected or stored by us.
                    </p>

                    <h3 className="text-xl font-semibold text-slate-100 mt-6 mb-3">Third-Party Services</h3>
                    <p>
                        We do not upload any data to any third-party services.
                        However, some resources are loaded from third-party services:
                    </p>
                    <ul className="list-disc list-inside ml-4 space-y-1">
                        <li>Hugging Face: ONNX model files</li>
                        <li>jsDelivr: ONNX Web Runtime, ffmpeg.wasm</li>
                    </ul>

                    <h3 className="text-xl font-semibold text-slate-100 mt-6 mb-3">Open Source</h3>
                    <p>
                        This application is open source. You can verify all claims by viewing
                        the <a href="https://github.com/Ryan5453/demucs-next" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline">source code on GitHub</a>.
                    </p>

                    <h3 className="text-xl font-semibold text-slate-100 mt-6 mb-3">Contact</h3>
                    <p>
                        If you have any questions about this privacy policy, please open an issue on our
                        <a href="https://github.com/Ryan5453/demucs-next/issues" target="_blank" rel="noopener noreferrer" className="text-cyan-400 hover:underline"> GitHub repository</a>.
                    </p>
                </div>
            </div>
        </div>
    );
}
