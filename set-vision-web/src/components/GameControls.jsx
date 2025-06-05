import React from 'react'; 
import "../styles.css"; // Import global styles

export default function GameControls({ gameStarted, hasStartedBefore, onStart, onStop }) {
  const handleClick = () => {
    gameStarted ? onStop() : onStart();
  };

  const buttonLabel = () => {
    if (gameStarted) return '🟥 Stop Game';
    if (hasStartedBefore) return '🔁 Restart Game';
    return '🟢 Start Game';
  };

 return (
    <div className="controls-container">
      <div className='start-game-wrapper'>
      <button onClick={handleClick} className={`${gameStarted ? 'stop' : 'start'}-game-button `}>
        {buttonLabel()}
      </button>
      </div>
    </div>
  );
}