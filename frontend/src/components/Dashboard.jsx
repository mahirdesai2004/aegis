/**
 * Dashboard.jsx — Real-Time Risk Score Display
 *
 * Renders a circular risk indicator and a list of triggered
 * detection labels. Colour-coded by severity.
 */

function Dashboard({ riskData }) {
  const { score, level, labels } = riskData;

  return (
    <div className="risk-dashboard" id="risk-dashboard">
      {/* ── Circular score indicator ─────────────────────── */}
      <div className={`risk-score-circle ${level}`}>
        <span className="risk-value">{score}</span>
        <span className="risk-label-text">Risk</span>
      </div>

      {/* ── Level badge ──────────────────────────────────── */}
      <div className="level-badge-row">
        <span className={`level-badge ${level}`}>{level}</span>
      </div>

      {/* ── Detection labels ─────────────────────────────── */}
      <h4 className="section-label">Detected Labels</h4>
      <div className="labels-container">
        {labels.length === 0 ? (
          <span className="no-risk-text">No risks detected</span>
        ) : (
          labels.map((label, i) => (
            <span key={i} className={`risk-pill ${label.startsWith('pii_') ? 'pii' : ''}`}>
              {label.replaceAll('_', ' ')}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

export default Dashboard;
