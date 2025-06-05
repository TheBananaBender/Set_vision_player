import React from 'react';

export default function GameControls({ onStart, isVisible }) {
  if (!isVisible) return null; // 🔥 vanish when hidden

  return (
    <div className="controls-container">
      <div className="start-game-wrapper">
        <button onClick={onStart} className="start-game-button">
          🎮 Start Game
        </button>
      </div>
    </div>
  );
}
