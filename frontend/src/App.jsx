/**
 * App.jsx — Aegis Application Root
 *
 * Lays out the three core components:
 *   1. VideoStream  — webcam capture & processed display
 *   2. Controls     — toggle redaction, start/stop scanner
 *   3. Dashboard    — real-time risk score & labels
 */

import { useState } from 'react';
import VideoStream from './components/VideoStream';
import Controls from './components/Controls';
import Dashboard from './components/Dashboard';
import './index.css';

function App() {
  const [redact, setRedact] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [riskData, setRiskData] = useState({
    score: 0,
    level: 'LOW',
    labels: [],
  });

  return (
    <div className="app-container">
      {/* ── Header ────────────────────────────────────────── */}
      <header>
        <h1>Aegis</h1>
        <p>Context-Aware Multi-Modal Privacy Detection & Redaction</p>
      </header>

      {/* ── Main grid layout ──────────────────────────────── */}
      <main className="main-content">
        {/* Left: video feed */}
        <div className="glass-panel video-panel">
          <VideoStream
            redact={redact}
            streaming={streaming}
            onRiskUpdate={setRiskData}
          />
        </div>

        {/* Right: controls + dashboard stacked */}
        <aside className="sidebar">
          <div className="glass-panel">
            <Controls
              redact={redact}
              setRedact={setRedact}
              streaming={streaming}
              setStreaming={setStreaming}
            />
          </div>
          <div className="glass-panel">
            <Dashboard riskData={riskData} />
          </div>
        </aside>
      </main>

      {/* ── Footer ────────────────────────────────────────── */}
      <footer>
        Built with YOLOv8 · EasyOCR · FastAPI · React
      </footer>
    </div>
  );
}

export default App;
