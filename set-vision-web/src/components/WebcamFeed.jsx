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
  const currentPolygonsRef = useRef(new Set()); // Store polygons as strings for comparison
  const aiClaimedPolygonsRef = useRef([]); // AI claimed SET polygons (to draw in red)

  // --- camera on/off (unchanged behavior) ---
  useEffect(() => {
    if (gameStarted) {
      // Use camera index 1 (USB camera)
      navigator.mediaDevices.enumerateDevices()
        .then((devices) => {
          const videoDevices = devices.filter(device => device.kind === 'videoinput');
          console.log('[WebcamFeed] Available cameras:', videoDevices.map((d, i) => `${i}: ${d.label}`));
          
          // Use camera at index 1 (USB camera)
          const targetCamera = videoDevices[1] || videoDevices[0]; // Fallback to first if only one exists
          console.log('[WebcamFeed] Using camera:', targetCamera.label);
          
          return navigator.mediaDevices.getUserMedia({ 
            video: { deviceId: { exact: targetCamera.deviceId } } 
          });
        })
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

  const reset = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'reset' }));
      // Clear AI claimed polygons on reset
      aiClaimedPolygonsRef.current = [];
      console.log('[WebcamFeed] Reset command sent to backend');
    }
  }, []);

  const ping = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'ping', t: Date.now() / 1000 }));
    }
  }, []);

  // Helper function to create polygon key for comparison
  const polygonKey = (poly) => JSON.stringify(poly);

  // capture backend results and store for overlay
  const handleAgentResult = useCallback((msg) => {
    lastResultRef.current = msg;
    
    // Store AI claimed polygons (for red highlight)
    if (msg.ai_claimed_polygons) {
      aiClaimedPolygonsRef.current = msg.ai_claimed_polygons;
    }
    
    // Handle incremental polygon updates
    // If update_type is missing, treat as full update (backwards compatibility)
    if (msg.update_type === 'incremental') {
      // Remove polygons
      if (msg.polygons_removed) {
        msg.polygons_removed.forEach((poly) => {
          const key = polygonKey(poly);
          currentPolygonsRef.current.delete(key);
        });
      }
      // Add polygons
      if (msg.polygons_added) {
        msg.polygons_added.forEach((poly) => {
          const key = polygonKey(poly);
          currentPolygonsRef.current.add(key);
        });
      }
      // Convert back to array for rendering
      msg.polygons = Array.from(currentPolygonsRef.current).map(key => JSON.parse(key));
    } else {
      // Full update - replace all polygons (or no update_type specified)
      currentPolygonsRef.current.clear();
      if (msg.polygons) {
        msg.polygons.forEach((poly) => {
          const key = polygonKey(poly);
          currentPolygonsRef.current.add(key);
        });
      }
    }
    
    onAgentResult?.(msg);
  }, [onAgentResult]);

  // let parent know about API/states whenever they change
  useEffect(() => {
    onBridgeReady?.({ save, reset, ping, wsConnected });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsConnected]); // Only notify when connection state actually changes

  // --- WS connect + frame push loop ---
  useEffect(() => {
    if (!gameStarted) {
      // stop frame loop + close socket if we had one
      if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
      sendingRef.current = false;
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {}
        wsRef.current = null;
      }
      setWsConnected(false);
      return;
    }

    // Prevent duplicate connections
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      console.log('[WebcamFeed] WebSocket already exists, skipping new connection');
      return;
    }

    // connect WS (Vite proxy should forward '/ws' -> http://127.0.0.1:8000)
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${window.location.host}/ws`;
    console.log('[WebcamFeed] Connecting to WebSocket:', url);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WebcamFeed] WebSocket connected!');
      setWsConnected(true);
      // update parent with latest wsConnected
      onBridgeReady?.({ save, reset, ping, wsConnected: true });
    };
    ws.onclose = (event) => {
      console.log('[WebcamFeed] WebSocket closed:', event.code, event.reason);
      setWsConnected(false);
      sendingRef.current = false;
      onBridgeReady?.({ save, reset, ping, wsConnected: false });
      // Clear the ref only if we're still in the stopped state
      if (!gameStarted) {
        wsRef.current = null;
      }
    };
    ws.onerror = (error) => {
      console.error('[WebcamFeed] WebSocket error:', error);
      setWsConnected(false);
      onBridgeReady?.({ save, reset, ping, wsConnected: false });
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        // pass any result up if requested
        if (msg.type === 'result') {
          handleAgentResult(msg);
        } else if (msg.type === 'reset_ack') {
          console.log('[WebcamFeed] Backend reset confirmed:', msg);
        }
        // release backpressure after any server reply
        sendingRef.current = false;
      } catch (e) {
        console.warn('[WebcamFeed] Failed to parse message:', e);
      }
    };

    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement('canvas');
    }

    // ~6-8 FPS loop pushing frames as binary (webp) with backpressure
    // Keep aspect ratio to avoid distortion
    timerRef.current = setInterval(async () => {
      const video = videoRef.current;
      const socket = wsRef.current;
      if (!video || !socket || socket.readyState !== WebSocket.OPEN) return;
      if (sendingRef.current) return;

      const vw = video.videoWidth || 0;
      const vh = video.videoHeight || 0;
      if (!vw || !vh) return;

      const canvas = offscreenRef.current;
      
      // Maintain aspect ratio: scale to fit within 640x640 (better quality than 320x320)
      const maxDim = 640;
      const scale = Math.min(maxDim / vw, maxDim / vh);
      const scaledW = Math.floor(vw * scale);
      const scaledH = Math.floor(vh * scale);
      
      if (canvas.width !== scaledW || canvas.height !== scaledH) {
        canvas.width = scaledW;
        canvas.height = scaledH;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Draw video maintaining aspect ratio
      ctx.drawImage(video, 0, 0, scaledW, scaledH);

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
      // Close the WebSocket if it exists
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {}
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameStarted]); // Only depend on gameStarted to prevent reconnects

  // Draw polygons overlay when result updates
  useEffect(() => {
    const draw = () => {
      const result = lastResultRef.current;
      const video = videoRef.current;
      const canvas = overlayRef.current;
      if (!result || !video || !canvas) return;
      
      // Get actual video internal dimensions (what the backend sees)
      const vw = video.videoWidth || 0;
      const vh = video.videoHeight || 0;
      if (!vw || !vh) return;
      
      // Get the actual rendered video rectangle (accounting for object-fit)
      const videoRect = video.getBoundingClientRect();
      
      // Use the actual displayed dimensions from bounding rect
      const displayW = Math.round(videoRect.width);
      const displayH = Math.round(videoRect.height);
      
      // Set canvas internal dimensions to match video
      if (canvas.width !== displayW || canvas.height !== displayH) {
        canvas.width = displayW;
        canvas.height = displayH;
      }
      
      // Also set CSS dimensions to exactly match (override any CSS)
      canvas.style.width = `${displayW}px`;
      canvas.style.height = `${displayH}px`;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 3;
      
      // Calculate scaling from sent image dimensions to display dimensions
      const maxDim = 640;
      const sendScale = Math.min(maxDim / vw, maxDim / vh);
      const scaledW = Math.floor(vw * sendScale);
      const scaledH = Math.floor(vh * sendScale);
      
      // Debug log (show every 30 frames to see live values)
      if (result.frame_num % 30 === 1) {
        console.log('[WebcamFeed] Video internal:', vw, 'x', vh, '→ Display:', displayW, 'x', displayH);
        console.log('[WebcamFeed] Sent to backend:', scaledW, 'x', scaledH, '→ Canvas:', canvas.width, 'x', canvas.height);
        console.log('[WebcamFeed] Scale factors: sx=', displayW / scaledW, 'sy=', displayH / scaledH);
      }
      
      // Scale polygons: backend coords → sent dimensions → display dimensions
      // Backend gives coords in terms of sent image (scaledW x scaledH)
      // We need to map to display dimensions (displayW x displayH)
      const sx = displayW / scaledW;
      const sy = displayH / scaledH;
      
      // Helper function to draw a polygon
      const drawPolygon = (poly, color, lineWidth) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        poly.forEach(([x, y], i) => {
          const px = x * sx;
          const py = y * sy;
          if (i === 0) {
            ctx.moveTo(px, py);
          } else {
            ctx.lineTo(px, py);
          }
        });
        ctx.closePath();
        ctx.stroke();
      };
      
      // Draw regular board polygons in green
      (result.polygons || []).forEach((poly) => {
        drawPolygon(poly, '#00FF00', 3);
      });
      
      // Draw AI claimed polygons in red (drawn on top, even if removed from board)
      aiClaimedPolygonsRef.current.forEach((poly) => {
        drawPolygon(poly, '#FF0000', 5);
      });
    };
    const id = setInterval(draw, 100);
    return () => clearInterval(id);
  }, []);

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
