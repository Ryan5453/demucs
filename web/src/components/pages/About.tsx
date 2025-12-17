import { Link } from 'react-router-dom';

export function About() {
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
                <h2 className="text-3xl font-bold text-slate-100 mb-6">About</h2>

                <div className="space-y-5 text-slate-300 leading-relaxed">
                    <p>
                        <strong className="text-slate-100">demucs.app</strong> is a free, open-source audio stem separation tool powered by Meta AI's Demucs model.
                        Everything runs entirely in your browser, so your audio files never leave your device.
                    </p>

                    <h3 className="text-xl font-semibold text-slate-100 pt-4">The Technology</h3>

                    <p>
                        Demucs is a machine learning model that separates mixed audio into individual stems.
                        The standard 4-source model separates audio into drums, bass, vocals, and other instruments.
                        There's also an experimental 6-source model that adds guitar and piano stems, though piano separation is less reliable.
                    </p>

                    <p>
                        This app uses Demucs models converted to ONNX format for in-browser inference.
                        When you select a model, either the WebGPU runtime (~24MB) or WebAssembly runtime (~12MB) is downloaded based on your device's capabilities, along with the model weights.
                    </p>

                    <p>
                        Audio files are decoded using <a href="https://mediabunny.dev/" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2 transition-colors">MediaBunny</a>, which leverages your browser's native capabilities.
                        For files that can't be decoded natively, the app falls back to <a href="https://ffmpegwasm.netlify.app/" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-2 transition-colors">ffmpeg.wasm</a> (~32MB).
                    </p>
                </div>
            </div>
        </div>
    );
}
