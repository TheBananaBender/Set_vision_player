import React, { useState } from 'react';
import './HeaderMenu.css';

export default function HeaderMenu() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [soundOn, setSoundOn] = useState(true); // 🔊 sound state

  const toggleSound = () => {
    setSoundOn(prev => !prev);
    // Optional: trigger audio mute logic here
  };

  return (
    <div className="header-menu">
      {/* ❓ Help */}
      <div className="menu-item hoverable">
        <button className="menu-button">❓</button>
        <div className="menu-dropdown help-dropdown">
          <h4>Game Help</h4>
          <p>This is a SET game between you and an AI agent.</p>
          <p>Find sets of 3 cards with matching or different features.</p>
        </div>
      </div>

      {/* ⚙️ Settings */}
      <div className="menu-item">
        <button
          className="menu-button"
          onClick={() => setSettingsOpen(!settingsOpen)}
        >
          ⚙️
        </button>
        {settingsOpen && (
          <div className="menu-dropdown">
            <h4>Settings</h4>
            <label>
              Agent Speed:
              <select>
                <option>Easy</option>
                <option>Medium</option>
                <option>Hard</option>
              </select>
            </label>

            <label>
              Delay (sec):
              <input type="number" defaultValue={3} min={0} max={10} />
            </label>

            {/* 🎵 Mute/Unmute Toggle */}
            <label className="mute-toggle">
              <button className="mute-button" onClick={toggleSound}>
                {soundOn ? '🔊' : '🔇'}
              </button>
              <span>{soundOn ? 'Sound On' : 'Muted'}</span>
            </label>
          </div>
        )}
      </div>
    </div>
  );
}
