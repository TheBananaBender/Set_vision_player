import React, { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
import GameControls from './components/GameControls';
import StatusPanel from './components/StatusPanel';
import HeaderMenu from './components/HeaderMenu';
import './styles.css';

export default function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [hasStartedBefore, setHasStartedBefore] = useState(false); // 🔄 new flag
  const [status, setStatus] = useState('Waiting to start...');
  

  // 🟢 Start Game Logic
  const handleStart = () => {
    setGameStarted(true);
    setHasStartedBefore(true); // 🔄 flag is now true after first start
    setStatus("Game started. Awaiting agent move...");
    fetch('http://localhost:8000/control', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({ action: 'start' }),
    });
  };

  // 🟥 Stop Game Logic
  const handleStop = () => {
    setGameStarted(false);
    setStatus("Game stopped.");
    // Send stop command to backend
    fetch('http://localhost:8000/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' }),
    });
  };

  return (
    <div className="app">
      <div className="app-header">
        <div className="title-strip">
          <h1>🃏 SET Vision Agent</h1>
        </div>
      </div>

      <HeaderMenu />
      <WebcamFeed gameStarted={gameStarted} />
      
      <GameControls 
        gameStarted={gameStarted}
        onStart={handleStart}
        onStop={handleStop}
    />

      

      
      <StatusPanel status={status} />
    </div>
  );
}
