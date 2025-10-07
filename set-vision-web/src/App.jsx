import React, { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
import GameControls from './components/GameControls';
import StatusPanel from './components/StatusPanel';
import HeaderMenu from './components/HeaderMenu';
import './styles.css';

export default function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [hasStartedBefore, setHasStartedBefore] = useState(false);
  const [status, setStatus] = useState('Waiting to start...');
  const [agentApi, setAgentApi] = useState(null); // { save, wsConnected }

  // Start/Stop/Reset are now purely local (no fetch to /control)
  const handleStart = () => {
    setGameStarted(true);
    setHasStartedBefore(true);
    setStatus('Game started. Streaming frames to agent…');
  };

  const handleStop = () => {
    setGameStarted(false);
    setStatus('Game stopped.');
  };

  const handleReset = () => {
    setGameStarted(false);
    setHasStartedBefore(false);
    setStatus('Game reset.');
  };

  // Optional: wire a Save action through the WS bridge
  const handleSave = () => {
    agentApi?.save?.(); // sends {"type":"save"} over /ws
    setStatus('Save requested…');
  };

  return (
    <div className="app">
      <div className="app-header">
        <div className="title-strip">
          <h1>🃏 SET Vision Agent</h1>
        </div>
      </div>

      <HeaderMenu />

      {/* IMPORTANT: pass onBridgeReady so we can call save() from here */}
      <WebcamFeed gameStarted={gameStarted} onBridgeReady={setAgentApi} />

      <GameControls
        gameStarted={gameStarted}
        hasStartedBefore={hasStartedBefore}
        onStart={handleStart}
        onStop={handleStop}
        onReset={handleReset}
        onSave={handleSave}         // ← add a Save button in GameControls if you want
        wsConnected={!!agentApi?.wsConnected} // (optional) show link status
      />

      <StatusPanel status={status} />
    </div>
  );
}
