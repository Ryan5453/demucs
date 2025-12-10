import { useState, useEffect } from 'react';
import { useDemucs } from './hooks/useDemucs';
import { AquaWindow } from './components/ui/AquaWindow';
import { MainPlayer } from './components/MainPlayer';
import { StemControls } from './components/StemControls';
import './index.css';

function App() {
  const {
    modelLoaded,
    modelLoading,
    audioLoaded,
    audioBuffer,
    audioFile,
    separating,
    progress,
    status,
    stemUrls,
    loadModel,
    loadAudio,
    separateAudio,
  } = useDemucs();

  const [webGpuAvailable, setWebGpuAvailable] = useState<boolean>(true);

  useEffect(() => {
    if (!navigator.gpu) {
      setWebGpuAvailable(false);
    }
  }, []);

  if (!webGpuAvailable) {
    return (
      <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
        <AquaWindow title="Error - Hardware Unsupported">
          <div className="aqua-inset-panel p-6 text-center max-w-md">
            <h2 className="text-red-600 text-lg mb-4 font-bold">WebGPU Not Detected</h2>
            <p className="mb-4 leading-relaxed text-gray-700">
              This application requires a browser with WebGPU support (e.g., Chrome/Edge 113+).
            </p>
            <p className="mb-6 leading-relaxed text-gray-700">
              Please update your browser or run the application locally using the Python version.
            </p>
            <a
              href="https://github.com/ryan5453/demucs"
              target="_blank"
              rel="noreferrer"
              className="aqua-btn primary"
            >
              View on GitHub
            </a>
          </div>
        </AquaWindow>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {/* Main Player Window */}
      <AquaWindow title="Demucs" width={700}>
        <MainPlayer
          fileName={audioFile?.name || null}
          status={status}
          progress={progress}
          duration={audioBuffer?.duration || 0}
          modelLoaded={modelLoaded}
          modelLoading={modelLoading}
          audioLoaded={audioLoaded}
          separating={separating}
          onLoadModel={() => loadModel()}
          onLoadAudio={loadAudio}
          onSeparate={separateAudio}
        />
      </AquaWindow>

      {/* Stems Panel */}
      <AquaWindow title="Audio Stems" width={700}>
        <StemControls stemUrls={stemUrls} />
      </AquaWindow>
    </div>
  );
}

export default App;
