/**
 * VideoStream.jsx — Live Webcam Feed (v3)
 *
 * Sequential async loop: capture → send → display → wait → repeat.
 * Never overlaps requests. Uses AbortController for timeout safety.
 * Target: ~6-8 FPS.
 */

import { useRef, useEffect, useState, useCallback } from 'react';

const API_URL = 'http://localhost:8000/process_frame';
const FRAME_INTERVAL = 150;   // ms between frames
const FETCH_TIMEOUT = 5000;   // ms timeout per request
const MAX_WIDTH = 640;

function VideoStream({ redact, streaming, onRiskUpdate }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const displayRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);

  // Start webcam once on mount
  useEffect(() => {
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } })
      .then((stream) => {
        if (!cancelled && videoRef.current) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          setCameraReady(true);
        }
      })
      .catch((err) => console.error('Camera error:', err));

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const updateRisk = useCallback((d) => onRiskUpdate(d), [onRiskUpdate]);

  // Frame processing loop
  useEffect(() => {
    if (!streaming || !cameraReady) return;
    let active = true;

    const loop = async () => {
      while (active) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) {
          await sleep(100);
          continue;
        }

        // Draw frame to canvas (resized)
        const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

        // Send to backend
        try {
          const blob = await new Promise((r) => canvas.toBlob(r, 'image/jpeg', 0.7));
          if (!blob || !active) break;

          const form = new FormData();
          form.append('file', blob, 'frame.jpg');
          form.append('redact', String(redact));

          const ctrl = new AbortController();
          const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT);

          const res = await fetch(API_URL, { method: 'POST', body: form, signal: ctrl.signal });
          clearTimeout(timer);

          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const data = await res.json();

          if (!active) break;

          updateRisk({ score: data.risk_score, level: data.risk_level, labels: data.labels });

          if (displayRef.current) {
            displayRef.current.src = `data:image/jpeg;base64,${data.processed_frame}`;
          }
        } catch (err) {
          if (err.name !== 'AbortError') console.error('Frame:', err.message);
        }

        await sleep(FRAME_INTERVAL);
      }
    };

    loop();
    return () => { active = false; };
  }, [streaming, cameraReady, redact, updateRisk]);

  return (
    <div className="video-container" id="video-stream">
      <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {streaming ? (
        <>
          <div className="live-badge">● LIVE</div>
          <img ref={displayRef} alt="Processed feed" className="processed-img" />
        </>
      ) : (
        <div className="idle-message">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M23 7l-7 5 7 5V7z" /><rect x="1" y="5" width="15" height="14" rx="2" />
          </svg>
          <h3>Camera Ready</h3>
          <p>Press <strong>Start Scanner</strong> to begin detection</p>
        </div>
      )}
    </div>
  );
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

export default VideoStream;
