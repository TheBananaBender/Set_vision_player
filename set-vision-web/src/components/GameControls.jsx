import React from 'react';

export default function GameControls({ onStart }) {
  return (
    <div className="controls">
      <button onClick={onStart}>Start Game</button>
    </div>
  );
}
