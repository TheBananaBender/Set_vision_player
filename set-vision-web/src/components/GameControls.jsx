import "../styles.css";

export default function GameControls({ gameStarted, hasStartedBefore, onStart, onStop, onReset }) {
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
      <div className="start-game-wrapper">

        {/* Start/Stop/Restart Button */}
        <button
          onClick={handleClick}
          className={`${gameStarted ? 'stop' : 'start'}-game-button`}
        >
          {buttonLabel()}
        </button>
      </div>
      {/* Always Visible Reset Button */}
        <button
          onClick={onReset}
          className="reset-game-button"
          disabled={!hasStartedBefore}
        >
          🔄 Reset Game
        </button>
    </div>
  );
}
