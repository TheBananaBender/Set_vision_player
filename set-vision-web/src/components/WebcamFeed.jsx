import React, { useEffect, useRef, useState, useCallback } from 'react';

export default function WebcamFeed({ gameStarted, onAgentResult, onBridgeReady }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // WS bridge
  const wsRef = useRef(null);
  const offscreenRef = useRef(null);
  const timerRef = useRef(null);
  const sendingRef = useRef(false);
  const [wsConnected, setWsConnected] = useState(false);

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
        if (msg.type === 'result' && typeof onAgentResult === 'function') {
          onAgentResult(msg);
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

    // ~8 FPS loop pushing frames as binary (webp) with simple backpressure
    timerRef.current = setInterval(async () => {
      const video = videoRef.current;
      const socket = wsRef.current;
      if (!video || !socket || socket.readyState !== WebSocket.OPEN) return;
      if (sendingRef.current) return;

      const w = video.videoWidth || 0;
      const h = video.videoHeight || 0;
      if (!w || !h) return;

      const canvas = offscreenRef.current;
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, w, h);

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
    }, 120);

    // cleanup on toggle/unmount
    return () => {
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      sendingRef.current = false;
      try { ws.close(); } catch {}
    };
  }, [gameStarted, onAgentResult, onBridgeReady, save, ping]);

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
