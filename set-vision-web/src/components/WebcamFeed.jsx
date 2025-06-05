import React, { useEffect, useRef } from 'react';

export default function WebcamFeed({ gameStarted }) {
  const videoRef = useRef();

  useEffect(() => {
    if (gameStarted) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
        }).catch(err => {
          console.error('Camera error:', err);
        });
    }
  }, [gameStarted]);

  return (
    <div className="webcam-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="webcam-video"
      />
    </div>
  );
}
