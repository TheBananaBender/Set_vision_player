import React, { useEffect, useRef, useState, useCallback } from 'react';

export default function WebcamFeed({ gameStarted, onAgentResult, onBridgeReady }) {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const streamRef = useRef(null);

  // WS bridge
  const wsRef = useRef(null);
  const offscreenRef = useRef(null); // for sending (fixed 320x320)
  const timerRef = useRef(null);
  const sendingRef = useRef(false);
  const [wsConnected, setWsConnected] = useState(false);
  const lastResultRef = useRef(null);

  // --- camera on/off (unchanged behavior) ---
  useEffect(() => {
    if (gameStarted) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
          streamRef.current = stream;
          if (videoRef.current) videoRef.current.srcObject = stream;
        })
        .catch((err) => console.error('Webcam error:', err));
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        if (videoRef.current) videoRef.current.srcObject = null;
      }
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
    };
  }, [gameStarted]);

  // --- actions we expose to parent ---
  const save = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'save' }));
    }
  }, []);

  const ping = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping', t: Date.now() / 1000 }));
    }
  }, []);

  // let parent know about API/states whenever they change
  useEffect(() => {
    onBridgeReady?.({ save, ping, wsConnected });
  }, [onBridgeReady, save, ping, wsConnected]);

  // --- WS connect + frame push loop ---
  useEffect(() => {
    if (!gameStarted) {
      // stop frame loop + close socket if we had one
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      sendingRef.current = false;
      if (wsRef.current) { try { wsRef.current.close(); } catch {} wsRef.current = null; }
      setWsConnected(false);
      return;
    }

    // connect WS (Vite proxy should forward '/ws' -> http://127.0.0.1:8000)
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${window.location.host}/ws`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setWsConnected(true);
      // update parent with latest wsConnected
      onBridgeReady?.({ save, ping, wsConnected: true });
    };
    ws.onclose = () => {
      setWsConnected(false);
      sendingRef.current = false;
      onBridgeReady?.({ save, ping, wsConnected: false });
    };
    ws.onerror = () => {
      setWsConnected(false);
      onBridgeReady?.({ save, ping, wsConnected: false });
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        // pass any result up if requested
        if (msg.type === 'result') {
          handleAgentResult(msg);
        }
        // release backpressure after any server reply
        sendingRef.current = false;
      } catch {
        // non-JSON messages are ignored
      }
    };

    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }

    // ~6-8 FPS loop pushing 320x320 frames as binary (webp) with backpressure
    timerRef.current = setInterval(async () => {
      const video = videoRef.current;
      const socket = wsRef.current;
      if (!video || !socket || socket.readyState !== WebSocket.OPEN) return;
      if (sendingRef.current) return;

      const vw = video.videoWidth || 0;
      const vh = video.videoHeight || 0;
      if (!vw || !vh) return;

      const canvas = offscreenRef.current;
      // sending at 320x320
      if (canvas.width !== 320 || canvas.height !== 320) {
        canvas.width = 320;
        canvas.height = 320;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // draw video scaled into 320x320
      ctx.drawImage(video, 0, 0, 320, 320);

      sendingRef.current = true;
      try {
        const blob = await new Promise((res) => canvas.toBlob(res, 'image/webp', 0.7));
        if (blob && blob.size > 0) {
          const ab = await blob.arrayBuffer();
          socket.send(ab); // binary frame to FastAPI
        } else {
          // nothing sent; allow next tick
          sendingRef.current = false;
        }
      } catch {
        sendingRef.current = false;
      }
    }, 150);

    // cleanup on toggle/unmount
    return () => {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      sendingRef.current = false;
      try { ws.close(); } catch {}
    };
  }, [gameStarted, onAgentResult, onBridgeReady, save, ping]);

  // Draw polygons overlay when result updates
  useEffect(() => {
    const draw = () => {
      const result = lastResultRef.current;
      const video = videoRef.current;
      const canvas = overlayRef.current;
      if (!result || !video || !canvas) return;
      const w = video.videoWidth || 0;
      const h = video.videoHeight || 0;
      if (!w || !h) return;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#00FF00';
      const sx = w / 320;
      const sy = h / 320;
      (result.polygons || []).forEach((poly) => {
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
          const px = x * sx;
          const py = y * sy;
          if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
        });
        ctx.closePath();
        ctx.stroke();
      });
    };
    const id = setInterval(draw, 100);
    return () => clearInterval(id);
  }, []);

  // capture backend results and store for overlay
  const handleAgentResult = useCallback((msg) => {
    lastResultRef.current = msg;
    onAgentResult?.(msg);
  }, [onAgentResult]);

  return (
    <div className="webcam-container">
      <div className="video-wrapper">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="webcam-video"
        />
        <canvas ref={overlayRef} className="webcam-overlay" />
        <div className="camera-overlay">
          <p>
            {!gameStarted
              ? '📷 Camera is off — Press "Start Game"'
              : `📷 Camera is on — agent ${wsConnected ? 'linked' : 'linking…'}`}
          </p>
        </div>
      </div>
    </div>
  );
}
