/**
 * VideoStream.jsx — Webcam Capture & Frame Processing
 *
 * Captures frames from the user's webcam, resizes them for
 * efficiency, and sends them to the backend at a controlled
 * ~10 FPS rate. Displays the processed (redacted) frame returned
 * by the API.
 */

import { useRef, useEffect, useState, useCallback } from 'react';

// ── Constants ────────────────────────────────────────────────
const API_URL = 'http://localhost:8000/process_frame';
const TARGET_FPS = 10;
const FRAME_INTERVAL_MS = 1000 / TARGET_FPS; // ~100ms
const MAX_CANVAS_WIDTH = 640; // resize before sending

function VideoStream({ redact, streaming, onRiskUpdate }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const displayRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);

  // ── Start / stop webcam on mount ──────────────────────────
  useEffect(() => {
    let stream = null;

    const initCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 640 } },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setCameraReady(true);
      } catch (err) {
        console.error('Camera access denied:', err);
      }
    };

    initCamera();

    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // ── Stable callback for risk updates ──────────────────────
  const updateRisk = useCallback(
    (data) => onRiskUpdate(data),
    [onRiskUpdate]
  );

  // ── Frame capture loop (throttled to ~10 FPS) ─────────────
  useEffect(() => {
    if (!streaming || !cameraReady) return;

    let timerId = null;
    let isBusy = false;

    const tick = async () => {
      if (isBusy) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < video.HAVE_ENOUGH_DATA) return;

      isBusy = true;

      // Resize canvas to max width for network efficiency
      const scale = Math.min(1, MAX_CANVAS_WIDTH / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to JPEG blob
      canvas.toBlob(
        async (blob) => {
          if (!blob) { isBusy = false; return; }

          const formData = new FormData();
          formData.append('file', blob, 'frame.jpg');
          formData.append('redact', String(redact));

          try {
            const res = await fetch(API_URL, { method: 'POST', body: formData });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);

            const data = await res.json();

            // Update risk dashboard
            updateRisk({
              score: data.risk_score,
              level: data.risk_level,
              labels: data.labels,
            });

            // Display the processed frame
            if (displayRef.current) {
              displayRef.current.src = `data:image/jpeg;base64,${data.processed_frame}`;
            }
          } catch (err) {
            console.error('Frame processing error:', err);
          } finally {
            isBusy = false;
          }
        },
        'image/jpeg',
        0.75
      );
    };

    timerId = setInterval(tick, FRAME_INTERVAL_MS);

    return () => clearInterval(timerId);
  }, [streaming, cameraReady, redact, updateRisk]);

  // ── Render ────────────────────────────────────────────────
  return (
    <div className="video-container" id="video-stream">
      {/* Hidden: native video + offscreen canvas */}
      <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {streaming ? (
        <>
          <div className="pulse-ring">LIVE</div>
          <img ref={displayRef} alt="Processed feed" />
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

export default VideoStream;
