/*import React, { useEffect, useRef, useState } from "react";

// In dev, Vite proxy (see vite.config.js below) forwards /ws to FastAPI.
// In prod, your reverse-proxy should do the same.
const WS_PATH = "/ws";

export default function SetAgentLive() {
  const videoRef = useRef(null);
  const overlayRef = useRef(null);
  const workRef = useRef(null);
  const wsRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [result, setResult] = useState(null);
  const [sending, setSending] = useState(false);
  const sendingRef = useRef(false);

  // --- Utils ---
  const wsUrl = () => {
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    return `${proto}://${window.location.host}${WS_PATH}`;
  };

  // --- Camera setup ---
  useEffect(() => {
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const setSizes = () => {
        const w = videoRef.current?.videoWidth || 640;
        const h = videoRef.current?.videoHeight || 480;
        [overlayRef.current, workRef.current].forEach((c) => {
          if (c) { c.width = w; c.height = h; }
        });
      };
      videoRef.current?.addEventListener("loadedmetadata", setSizes);
    })();
  }, []);

  // --- WebSocket setup ---
  useEffect(() => {
    const ws = new WebSocket(wsUrl());
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => { setConnected(false); sendingRef.current = false; setSending(false); };
    ws.onerror = () => setConnected(false);

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "result") {
          setResult(msg);
          sendingRef.current = false;
          setSending(false);
        } else if (msg.type === "save_ack") {
          console.log("Saved:", msg);
        }
      } catch (e) {
        console.warn("Non-JSON message:", e);
      }
    };

    return () => ws.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Frame sending loop ---
  useEffect(() => {
    const interval = setInterval(async () => {
      if (!connected || !videoRef.current || !workRef.current || !wsRef.current) return;
      if (sendingRef.current) return;

      const wctx = workRef.current.getContext("2d");
      if (!wctx) return;

      wctx.drawImage(videoRef.current, 0, 0, workRef.current.width, workRef.current.height);

      sendingRef.current = true;
      setSending(true);
      try {
        const blob = await new Promise((res) =>
          workRef.current.toBlob(res, "image/webp", 0.7) // adjust quality/FPS to taste
        );
        if (blob && blob.size > 0) {
          const ab = await blob.arrayBuffer();
          wsRef.current.send(ab); // send as binary
        } else {
          sendingRef.current = false;
          setSending(false);
        }
      } catch {
        sendingRef.current = false;
        setSending(false);
      }
    }, 120); // ~8 FPS
    return () => clearInterval(interval);
  }, [connected]);

  // --- Draw overlays ---
  useEffect(() => {
    if (!result || !overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, overlayRef.current.width, overlayRef.current.height);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#00FF00";

    (result.polygons || []).forEach((poly) => {
      ctx.beginPath();
      poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.stroke();
    });
  }, [result]);

  // --- Commands ---
  const handleSave = () => wsRef.current?.send(JSON.stringify({ type: "save" }));

  return (
    <div className="set-live">
      <div className="set-live__status">
        WS: {connected ? "connected ✅" : "disconnected ❌"} · sending: {sending ? "yes" : "no"}
      </div>

      <div className="set-live__controls">
        <button className="btn" onClick={handleSave} disabled={!connected}>Save frame + crops</button>
      </div>

      <div className="set-live__stage">
        <video ref={videoRef} className="set-live__video" muted playsInline />
        <canvas ref={overlayRef} className="set-live__overlay" />
      </div>

      <div className="set-live__meta">
        {result ? (
          <>
            <div>frame #{result.frame_num} · {result.latency_ms} ms</div>
            <div>hands: {result.hands ? "ON" : "OFF"}</div>
            <div>scores → Human: {result.scores?.human ?? 0} | AI: {result.scores?.ai ?? 0}</div>
          </>
        ) : (
          <div>waiting for results…</div>
        )}
      </div>

      <canvas ref={workRef} style={{ display: "none" }} />
    </div>
  );
}
*/