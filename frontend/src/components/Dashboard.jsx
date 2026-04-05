/**
 * Dashboard.jsx — Risk Score Display (v3)
 *
 * Circular risk indicator + level badge + detection label pills.
 * Labels are colour-coded: unknown=red, safe=green, device=orange.
 */

function Dashboard({ riskData }) {
  const { score, level, labels } = riskData;
  const lc = level.toLowerCase();

  const getLabelClass = (label) => {
    const l = label.toLowerCase();
    if (l.includes('unknown')) return 'unknown';
    if (l.includes('safe'))    return 'safe';
    if (l.includes('device'))  return 'device';
    return '';
  };

  return (
    <div className="risk-dashboard" id="risk-dashboard">
      {/* Score circle */}
      <div className={`risk-score-circle ${lc}`}>
        <span className="score">{score}</span>
        <span className="label">RISK</span>
      </div>

      {/* Level badge */}
      <span className={`risk-badge ${lc}`}>{level}</span>

      {/* Detection labels */}
      <span className="labels-title">DETECTED LABELS</span>
      <div className="label-pills">
        {labels.length === 0 ? (
          <span className="no-risk-text">No risks detected</span>
        ) : (
          labels.map((label, i) => (
            <span key={i} className={`label-pill ${getLabelClass(label)}`}>
              {label}
            </span>
          ))
        )}
      </div>
    </div>
  );
}

export default Dashboard;
