/**
 * App.jsx — Aegis Root Layout (v3)
 *
 * Two-mode interface:
 *   1. Live Camera — real-time webcam + safe mode
 *   2. Image Upload — register face + upload group image
 */

import { useState } from 'react';
import VideoStream from './components/VideoStream';
import Controls from './components/Controls';
import Dashboard from './components/Dashboard';
import ImageUpload from './components/ImageUpload';
import './index.css';

function App() {
  const [mode, setMode] = useState('live'); // 'live' | 'upload'
  const [redact, setRedact] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [safeMode, setSafeMode] = useState(false);
  const [riskData, setRiskData] = useState({
    score: 0,
    level: 'LOW',
    labels: [],
  });

  return (
    <div className="app-container">
      <header>
        <h1>Aegis</h1>
        <p>Context-Aware Multi-Modal Privacy Detection & Redaction</p>
      </header>

      {/* Mode Tabs */}
      <div className="mode-tabs">
        <button
          className={`mode-tab ${mode === 'live' ? 'active' : ''}`}
          onClick={() => { setMode('live'); setStreaming(false); }}
        >
          📹 Live Camera
        </button>
        <button
          className={`mode-tab ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => { setMode('upload'); setStreaming(false); }}
        >
          🖼️ Image Upload
        </button>
      </div>

      <main className="main-content">
        {mode === 'live' ? (
          <>
            {/* Left: video feed */}
            <div className="glass-panel video-panel">
              <VideoStream
                redact={redact}
                streaming={streaming}
                onRiskUpdate={setRiskData}
              />
            </div>

            {/* Right: controls + dashboard */}
            <aside className="sidebar">
              <div className="glass-panel">
                <Controls
                  redact={redact}
                  setRedact={setRedact}
                  streaming={streaming}
                  setStreaming={setStreaming}
                  safeMode={safeMode}
                  setSafeMode={setSafeMode}
                />
              </div>
              <div className="glass-panel">
                <Dashboard riskData={riskData} />
              </div>
            </aside>
          </>
        ) : (
          <div className="upload-mode-container">
            <ImageUpload onRiskUpdate={setRiskData} />
            <div className="glass-panel upload-dashboard">
              <Dashboard riskData={riskData} />
            </div>
          </div>
        )}
      </main>

      <footer>
        Built with YOLOv8 · OpenCV · FastAPI · React
      </footer>
    </div>
  );
}

export default App;
