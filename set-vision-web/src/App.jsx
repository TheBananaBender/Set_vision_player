import React, { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
import GameControls from './components/GameControls';
import StatusPanel from './components/StatusPanel';
import HeaderMenu from './components/HeaderMenu';
import './styles.css';

export default function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [status, setStatus] = useState('Waiting to start...');

  const handleStart = () => {
    setGameStarted(true);
    setStatus("Game started. Awaiting agent move...");
  };

  return (
    <div className="app">
      <div className="app-header">
        <h1>SET Vision Agent</h1>
      </div>
      <HeaderMenu />
      <WebcamFeed gameStarted={gameStarted} />
      <GameControls onStart={handleStart} isVisible={!gameStarted} />
      <StatusPanel status={status} />

    </div>
  );
}
