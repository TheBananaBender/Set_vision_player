
export default function StopGameControls({ onStop }) {
  return (
    <div className="controls-container">
      <div className="start-game-wrapper">
        <button className="start-game-button" onClick={onStart}>
        </button>
      </div>
    </div>
  );
}
