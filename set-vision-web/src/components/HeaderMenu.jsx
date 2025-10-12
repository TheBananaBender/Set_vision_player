import React, { useState } from 'react';
import './HeaderMenu.css';
import './InstructionsModal.css'; // add this
import { FiHelpCircle, FiSettings } from "react-icons/fi";
import InstructionsModal from './InstructionsModal';

export default function HeaderMenu() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [soundOn, setSoundOn] = useState(true);
  const [helpOpen, setHelpOpen] = useState(false);

  const toggleSound = () => setSoundOn(prev => !prev);

  return (
    <div className="header-menu">
      {/* ❓ Help */}
      <div className="menu-item">
        <button className="menu-button" aria-label='Help' onClick={() => setHelpOpen(true)}>
          <FiHelpCircle size={20} />
        </button>
      </div>

      {/* ⚙️ Settings */}
      <div className="menu-item">
        <button
          className="menu-button"
          onClick={() => setSettingsOpen(!settingsOpen)} 
          aria-label="Settings"
        >
          <FiSettings size={20} />
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

            <label className="mute-toggle">
              <button className="mute-button" onClick={toggleSound}>
                {soundOn ? '🔊' : '🔇'}
              </button>
              <span>{soundOn ? 'Sound On' : 'Muted'}</span>
            </label>
          </div>
        )}
      </div>

      {/* Modal */}
      <InstructionsModal open={helpOpen} onClose={() => setHelpOpen(false)} />
    </div>
  );
}
