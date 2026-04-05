/**
 * ImageUpload.jsx — Image Upload Mode (v3)
 *
 * Two sections:
 *   1. Register Face — upload a face photo + name
 *   2. Upload Group Image — upload any image, see processed result
 */

import { useState, useRef, useEffect } from 'react';

const API_BASE = 'http://localhost:8000';

function ImageUpload({ onRiskUpdate }) {
  const [faces, setFaces] = useState([]);
  const [name, setName] = useState('');
  const [faceMsg, setFaceMsg] = useState('');
  const [processing, setProcessing] = useState(false);
  const [resultSrc, setResultSrc] = useState(null);
  const faceFileRef = useRef(null);
  const groupFileRef = useRef(null);

  useEffect(() => { fetchFaces(); }, []);

  const fetchFaces = async () => {
    try {
      const res = await fetch(`${API_BASE}/faces`);
      const data = await res.json();
      setFaces(data.faces || []);
    } catch { /* skip */ }
  };

  // ── Register Face ──────────────────────────────────────────
  const handleRegister = async () => {
    const file = faceFileRef.current?.files?.[0];
    if (!file || !name.trim()) {
      setFaceMsg('Enter a name and select a photo.');
      return;
    }

    const form = new FormData();
    form.append('file', file);
    form.append('name', name.trim());

    try {
      const res = await fetch(`${API_BASE}/register_face`, { method: 'POST', body: form });
      const data = await res.json();
      setFaceMsg(res.ok ? `✓ ${data.message}` : `✗ ${data.error}`);
      if (res.ok) {
        setName('');
        faceFileRef.current.value = '';
        fetchFaces();
      }
    } catch {
      setFaceMsg('✗ Server not responding.');
    }
  };

  const handleRemove = async (n) => {
    await fetch(`${API_BASE}/faces/${n}`, { method: 'DELETE' });
    fetchFaces();
  };

  // ── Process Group Image ────────────────────────────────────
  const handleGroupUpload = async () => {
    const file = groupFileRef.current?.files?.[0];
    if (!file) return;

    setProcessing(true);
    setResultSrc(null);

    const form = new FormData();
    form.append('file', file);

    try {
      const res = await fetch(`${API_BASE}/process_image`, { method: 'POST', body: form });
      const data = await res.json();

      if (res.ok) {
        setResultSrc(`data:image/jpeg;base64,${data.processed_frame}`);
        onRiskUpdate({ score: data.risk_score, level: data.risk_level, labels: data.labels });
      }
    } catch {
      alert('Server not responding.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="image-upload-mode">
      {/* Register Face */}
      <div className="glass-panel">
        <h2 className="controls-header">1. Register Safe Face</h2>
        <p className="face-desc">Upload a clear face photo of someone who should NOT be masked.</p>

        <div className="face-upload-form">
          <input
            type="text"
            placeholder="Person's name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="face-input"
          />
          <input type="file" accept="image/*" ref={faceFileRef} className="face-file-input" />
          <button className="btn-register" onClick={handleRegister}>+ Register Face</button>
        </div>

        {faceMsg && (
          <p className={`face-message ${faceMsg.startsWith('✓') ? 'success' : 'error'}`}>{faceMsg}</p>
        )}

        {faces.length > 0 && (
          <div className="face-list">
            {faces.map((f, i) => (
              <div key={i} className="face-chip">
                <span className="face-chip-icon">🛡️</span>
                <span className="face-chip-name">{f}</span>
                <button className="face-chip-remove" onClick={() => handleRemove(f)}>×</button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload Group Image */}
      <div className="glass-panel">
        <h2 className="controls-header">2. Upload Group Image</h2>
        <p className="face-desc">Upload any image. Registered faces stay visible, others get emoji masks.</p>

        <div className="face-upload-form">
          <input type="file" accept="image/*" ref={groupFileRef} className="face-file-input" />
          <button
            className="btn-main"
            onClick={handleGroupUpload}
            disabled={processing}
          >
            {processing ? '⏳ Processing...' : '🔍 Analyze Image'}
          </button>
        </div>

        {resultSrc && (
          <div className="result-preview">
            <img src={resultSrc} alt="Processed result" className="result-img" />
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageUpload;
