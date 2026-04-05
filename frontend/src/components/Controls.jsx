/**
 * Controls.jsx — Live Camera Controls (v3)
 *
 * Buttons:
 *   - Start/Stop Scanner
 *   - Set Me As Safe (captures current frame, registers center person)
 *   - Reset Safe
 *   - Enable/Disable Redaction toggle
 */

import { useRef } from 'react';

const API_BASE = 'http://localhost:8000';

function Controls({ redact, setRedact, streaming, setStreaming, safeMode, setSafeMode }) {

  const handleSetSafe = async () => {
    // Capture current frame from the hidden video element
    const video = document.querySelector('video');
    if (!video) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.8));
    if (!blob) return;

    const formData = new FormData();
    formData.append('file', blob, 'safe.jpg');

    try {
      const res = await fetch(`${API_BASE}/set_safe_mode`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setSafeMode(true);
        alert('✓ ' + data.message);
      } else {
        alert('✗ ' + data.error);
      }
    } catch {
      alert('✗ Server not responding.');
    }
  };

  const handleResetSafe = async () => {
    try {
      await fetch(`${API_BASE}/reset_safe`, { method: 'POST' });
      setSafeMode(false);
    } catch {
      alert('✗ Server not responding.');
    }
  };

  return (
    <div className="controls" id="controls">
      <h2 className="controls-header">Controls</h2>
      <hr className="divider" />

      {/* Redaction toggle */}
      <label className="toggle-row" htmlFor="redact-toggle">
        <span>Enable Redaction</span>
        <input
          id="redact-toggle"
          type="checkbox"
          className="toggle"
          checked={redact}
          onChange={() => setRedact(!redact)}
        />
      </label>

      {/* Start / Stop */}
      <button
        className={streaming ? 'btn-main btn-danger' : 'btn-main'}
        onClick={() => setStreaming(!streaming)}
      >
        {streaming ? '■ Stop Scanner' : '▶ Start Scanner'}
      </button>

      {/* Safe Mode controls */}
      <div className="safe-controls">
        <button
          className="btn-safe"
          onClick={handleSetSafe}
          disabled={!streaming}
          title={!streaming ? 'Start the scanner first' : 'Set center person as safe'}
        >
          🛡️ Set Me As Safe
        </button>
        <button
          className="btn-reset"
          onClick={handleResetSafe}
          disabled={!safeMode}
        >
          ↻ Reset Safe
        </button>
      </div>

      {safeMode && (
        <div className="safe-badge">
          <span className="safe-dot" /> Safe Mode Active
        </div>
      )}
    </div>
  );
}

export default Controls;
