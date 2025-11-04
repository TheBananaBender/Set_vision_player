import React, { useState } from 'react';
import WebcamFeed from './components/WebcamFeed';
import GameControls from './components/GameControls';
import StatusPanel from './components/StatusPanel';
import HeaderMenu from './components/HeaderMenu';
import AIMessageBox from './components/AIMessageBox';
import './styles.css';

export default function App() {
  const [gameStarted, setGameStarted] = useState(false);
  const [hasStartedBefore, setHasStartedBefore] = useState(false);
  const [status, setStatus] = useState('Waiting to start...');
  const [agentApi, setAgentApi] = useState(null); // { save, wsConnected }
  const [settings, setSettings] = useState({ difficulty: 'medium', delay_scale: 1.0, sound_on: true });
  const [scores, setScores] = useState({ human: 0, ai: 0 });
  const [aiStatus, setAiStatus] = useState({ state: 'idle', message: 'Waiting for game to start...' });

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
    // Stop the game first
    setGameStarted(false);
    
    // Reset backend through WebSocket
    if (agentApi?.reset) {
      agentApi.reset();
    }
    
    // Reset all frontend state
    setHasStartedBefore(false);
    setScores({ human: 0, ai: 0 });
    setAiStatus({ state: 'idle', message: 'Game reset. Ready to start!' });
    setStatus('Game reset successfully! Ready to start a new game.');
    
    console.log('[App] Full game reset complete');
  };

  // Optional: wire a Save action through the WS bridge
  const handleSave = () => {
    agentApi?.save?.(); // sends {"type":"save"} over /ws
    setStatus('Save requested…');
  };

  // settings change: send to backend only when not running
  const handleSettingsChange = async (next) => {
    setSettings(next);
    if (gameStarted) return;
    try {
      await fetch('/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(next),
      });
      setStatus('Settings updated');
    } catch (e) {
      setStatus('Failed to update settings');
    }
  };

  const handleAgentResult = (msg) => {
    if (msg?.scores) setScores(msg.scores);
    if (msg?.ai_status) setAiStatus(msg.ai_status);
  };

  return (
    <div className="app">
      <div className="app-header">
        <div className="title-strip">
          <h1>🃏 SET Vision Agent</h1>
        </div>
      </div>

      <HeaderMenu
        settings={settings}
        onSettingsChange={handleSettingsChange}
        settingsDisabled={gameStarted}
      />

      {/* IMPORTANT: pass onBridgeReady so we can call save() from here */}
      <WebcamFeed gameStarted={gameStarted} onBridgeReady={setAgentApi} onAgentResult={handleAgentResult} />

      {/* AI message box - shown only when game is running */}
      {gameStarted && <AIMessageBox message={aiStatus.message} state={aiStatus.state} />}

      <GameControls
        gameStarted={gameStarted}
        hasStartedBefore={hasStartedBefore}
        onStart={handleStart}
        onStop={handleStop}
        onReset={handleReset}
        onSave={handleSave}
        wsConnected={!!agentApi?.wsConnected}
      />

      <StatusPanel status={`${status} · Scores → Human: ${scores.human} | AI: ${scores.ai}`} />
    </div>
  );
}
