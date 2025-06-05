import React, { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
import GameControls from './components/GameControls';
import StatusPanel from './components/StatusPanel';
import CardGrid from './components/CardGrid';
import './styles.css';

export default function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [status, setStatus] = useState('Waiting to start...');
  
  return (
    <div className="app">
      <h1>🥷 SET Vision Agent</h1>
      <WebcamFeed gameStarted={gameStarted} />
      <GameControls onStart={() => {
        setGameStarted(true);
        setStatus("Game started. Awaiting agent move...");
      }} />
      <StatusPanel status={status} />
      <CardGrid />
    </div>
  );
}
