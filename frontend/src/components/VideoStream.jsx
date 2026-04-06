/**
 * VideoStream.jsx — Live Webcam Feed (v7 - Initialization Fixes)
 *
 * CRITICAL FIXES:
 * - PART 3: Frontend initialization delay (warmup)
 * - Add "ready" state before starting scanning
 * - Delay scanning by ~1 second after component mount
 * - Skip first few frames (warmup)
 * - Independent UI loop (always displays last frame)
 * - Decoupled from backend response timing
 * - Capture loop: 100ms (10 FPS) - independent
 * - Display loop: 50ms (20 FPS) - independent
 * - Backend requests: async, non-blocking
 * - isProcessing flag prevents overlapping requests
 * - Drop frames if backend busy (don't queue)
 * - AbortController timeout (250ms)
 *
 * Architecture:
 *   Capture Loop (100ms) → Send Async → Backend
 *   Display Loop (50ms) → Always show last frame
 *   Result: Smooth UI, no freeze, responsive
 */

import { useRef, useEffect, useState, useCallback } from 'react';

const API_URL = 'http://localhost:8000/process_frame';
const CAPTURE_INTERVAL = 100;         // ms between captures (10 FPS)
const DISPLAY_INTERVAL = 50;          // ms between display updates (20 FPS)
const FETCH_TIMEOUT = 250;            // ms timeout per request (abort if slow)
const MAX_WIDTH = 640;
const WARMUP_DELAY_MS = 1000;         // PART 3: Delay before starting (1 second)
const WARMUP_FRAMES = 5;              // PART 3: Skip first N frames

function VideoStream({ redact, streaming, onRiskUpdate }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const displayRef = useRef(null);
  const streamRef = useRef(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [isReady, setIsReady] = useState(false);  // PART 3: Ready state

  // Processing state
  const isProcessingRef = useRef(false);        // Is a request in flight?
  const lastDisplayRef = useRef(null);          // Last successfully processed frame (for display)
  const lastCaptureRef = useRef(null);          // Last captured frame (for sending)
  const lastRiskRef = useRef(null);             // Last risk data (avoid redundant updates)
  const frameCountRef = useRef(0);              // For logging
  const displayCountRef = useRef(0);            // For display logging
  const warmupCountRef = useRef(0);             // PART 3: Warmup frame counter

  // Start webcam once on mount
  useEffect(() => {
    let cancelled = false;

    navigator.mediaDevices
      .getUserMedia({ 
        video: { 
          facingMode: 'user', 
          width: { ideal: 640 }, 
          height: { ideal: 480 } 
        } 
      })
      .then((stream) => {
        if (!cancelled && videoRef.current) {
          videoRef.current.srcObject = stream;
          streamRef.current = stream;
          setCameraReady(true);
          
          // PART 3: Delay before marking as ready (warmup)
          console.log('[INIT] Camera ready, waiting for warmup...');
          const warmupTimer = setTimeout(() => {
            if (!cancelled) {
              setIsReady(true);
              console.log('[INIT] Warmup complete, ready to scan');
            }
          }, WARMUP_DELAY_MS);
          
          return () => clearTimeout(warmupTimer);
        }
      })
      .catch((err) => console.error('Camera error:', err));

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const updateRisk = useCallback((d) => onRiskUpdate(d), [onRiskUpdate]);

  // ─────────────────────────────────────────────────────────────────────
  // LOOP 1: CAPTURE LOOP (100ms = 10 FPS)
  // Captures frames and sends to backend (non-blocking)
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!streaming || !cameraReady || !isReady) return;  // PART 3: Check isReady

    const captureIntervalId = setInterval(async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      // Skip if video not ready
      if (!video || !canvas || video.readyState < 2) return;

      // CRITICAL: Skip if already processing (prevent overlapping requests)
      if (isProcessingRef.current) {
        console.log('[SKIP] Backend busy, dropping frame');
        return;
      }

      try {
        // PART 3: Skip warmup frames
        warmupCountRef.current++;
        if (warmupCountRef.current <= WARMUP_FRAMES) {
          console.log(`[WARMUP] Skipping frame ${warmupCountRef.current}/${WARMUP_FRAMES}`);
          return;
        }

        // 1. Capture frame from video
        const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

        // 2. Convert to blob
        const blob = await new Promise((resolve) => {
          canvas.toBlob(resolve, 'image/jpeg', 0.7);
        });

        if (!blob) return;

        // 3. Store for sending (non-blocking)
        lastCaptureRef.current = { blob, redact };
        frameCountRef.current++;

      } catch (err) {
        console.error('Capture error:', err.message);
      }
    }, CAPTURE_INTERVAL);

    return () => clearInterval(captureIntervalId);
  }, [streaming, cameraReady, isReady, redact]);

  // ─────────────────────────────────────────────────────────────────────
  // LOOP 2: SEND LOOP (async, triggered by capture)
  // Sends captured frames to backend (non-blocking, with timeout)
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!streaming || !cameraReady) return;

    const sendIntervalId = setInterval(async () => {
      // Skip if no frame to send or already processing
      if (!lastCaptureRef.current || isProcessingRef.current) return;

      const { blob, redact: shouldRedact } = lastCaptureRef.current;
      lastCaptureRef.current = null; // Clear for next frame

      try {
        // Mark as processing (prevent concurrent requests)
        isProcessingRef.current = true;
        const startTime = performance.now();

        // Send to backend with timeout
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        form.append('redact', String(shouldRedact));

        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT);

        const res = await fetch(API_URL, {
          method: 'POST',
          body: form,
          signal: ctrl.signal,
        });
        clearTimeout(timer);

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update display frame (will be shown by display loop)
        lastDisplayRef.current = `data:image/jpeg;base64,${data.processed_frame}`;

        // Update risk if changed
        const riskKey = `${data.risk_score}:${data.risk_level}`;
        if (lastRiskRef.current !== riskKey) {
          updateRisk({
            score: data.risk_score,
            level: data.risk_level,
            labels: data.labels,
          });
          lastRiskRef.current = riskKey;
        }

        // Log performance
        const elapsed = performance.now() - startTime;
        console.log(`[SEND ${frameCountRef.current}] ${elapsed.toFixed(0)}ms`);

      } catch (err) {
        if (err.name !== 'AbortError') {
          console.error('Send error:', err.message);
        }
        // Keep displaying last frame (display loop continues)
      } finally {
        // Mark as done (allow next request)
        isProcessingRef.current = false;
      }
    }, 50); // Check frequently for new frames to send

    return () => clearInterval(sendIntervalId);
  }, [streaming, cameraReady, updateRisk]);

  // ─────────────────────────────────────────────────────────────────────
  // LOOP 3: DISPLAY LOOP (50ms = 20 FPS)
  // Always displays last available frame (independent of backend)
  // ─────────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!streaming || !cameraReady) return;

    const displayIntervalId = setInterval(() => {
      const display = displayRef.current;
      if (!display) return;

      // Always update display with last available frame
      if (lastDisplayRef.current && display.src !== lastDisplayRef.current) {
        display.src = lastDisplayRef.current;
        displayCountRef.current++;
        console.log(`[DISPLAY ${displayCountRef.current}] Updated`);
      }
    }, DISPLAY_INTERVAL);

    return () => clearInterval(displayIntervalId);
  }, [streaming, cameraReady]);

  return (
    <div className="video-container" id="video-stream">
      <video ref={videoRef} autoPlay playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      {streaming ? (
        <>
          <div className="live-badge">● LIVE</div>
          <img 
            ref={displayRef} 
            alt="Processed feed" 
            className="processed-img"
            src={lastDisplayRef.current || 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="640" height="480"%3E%3Crect fill="%23222" width="640" height="480"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23666" font-size="18"%3EInitializing camera...%3C/text%3E%3C/svg%3E'}
          />
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
