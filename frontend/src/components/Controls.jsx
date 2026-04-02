/**
 * Controls.jsx — Scanner Controls
 *
 * Provides the redaction toggle switch and the start/stop
 * scanner button. Kept minimal — risk display is in Dashboard.
 */

function Controls({ redact, setRedact, streaming, setStreaming }) {
  return (
    <div className="controls-panel" id="controls-panel">
      <h2 className="controls-header">Controls</h2>

      {/* ── Redaction toggle ─────────────────────────────── */}
      <div className="toggle-container">
        <span className="toggle-label">Enable Redaction</span>
        <label className="switch">
          <input
            type="checkbox"
            checked={redact}
            onChange={(e) => setRedact(e.target.checked)}
          />
          <span className="slider" />
        </label>
      </div>

      {/* ── Start / Stop button ──────────────────────────── */}
      <button
        className={`btn-primary ${streaming ? 'btn-danger' : ''}`}
        onClick={() => setStreaming(!streaming)}
        id="scanner-toggle"
      >
        {streaming ? '⏹  Stop Scanner' : '▶  Start Scanner'}
      </button>
    </div>
  );
}

export default Controls;
