import { useState, useEffect } from 'react';
import { useDemucs } from './hooks/useDemucs';
import { WinampWindow } from './components/WinampWindow';
import { MainPlayer } from './components/MainPlayer';
import { StemControls } from './components/StemControls';
import { LogConsole } from './components/LogConsole';
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
    logs,
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
      <div className="fixed inset-0 bg-black flex items-center justify-center p-4">
        <WinampWindow title="Error - Hardware Unsupported">
          <div className="inset-panel p-4 text-center max-w-md">
            <h2 className="text-red-500 text-lg mb-4 font-bold">WebGPU Not Detected</h2>
            <p className="mb-4 leading-relaxed">
              This application requires a browser with WebGPU support (e.g., Chrome/Edge 113+).
            </p>
            <p className="mb-6 leading-relaxed">
              Please update your browser or run the application locally using the Python version.
            </p>
            <a
              href="https://github.com/ryan5453/demucs"
              target="_blank"
              rel="noreferrer"
              className="inline-block px-4 py-2 border-2 border-gray-400 bg-gray-800 text-green-400 hover:bg-gray-700 active:border-b-gray-600 active:border-r-gray-600"
            >
              View on GitHub
            </a>
          </div>
        </WinampWindow>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-[2px]">
      {/* Main Player Window */}
      <WinampWindow title="demucs" width={550}>
        <MainPlayer
          fileName={audioFile?.name || null}
          status={status}
          progress={progress}
          duration={audioBuffer?.duration || 0}
          modelLoaded={modelLoaded}
          modelLoading={modelLoading}
          audioLoaded={audioLoaded}
          separating={separating}
          onLoadModel={loadModel}
          onLoadAudio={loadAudio}
          onSeparate={separateAudio}
        />
      </WinampWindow>

      {/* Stems Panel (like EQ window) */}
      <WinampWindow title="Stems" width={550}>
        <StemControls
          stemUrls={stemUrls}
        />
      </WinampWindow>

      {/* Log Console (like Playlist window) */}
      <WinampWindow title="Console" width={550}>
        <LogConsole logs={logs} />
      </WinampWindow>
    </div>
  );
}

export default App;
