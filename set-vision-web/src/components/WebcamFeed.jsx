import React, { useEffect, useRef } from 'react';

export default function WebcamFeed({ gameStarted }) {
  const videoRef = useRef();
  const streamRef = useRef();

  useEffect(() => {
    if (gameStarted) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        })
        .catch(err => console.error("Webcam error:", err));
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
      }
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    };
  }, [gameStarted]);

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
            <p>{!gameStarted ? '📷 Camera is off — Press "Start Game"' :'📷 Camera is on — good luck'}</p>
          </div>

      </div>
    </div>
  );
}
