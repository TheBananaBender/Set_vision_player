import React, { useState } from 'react';
import './HeaderMenu.css';
import './InstructionsModal.css'; // add this
import { FiHelpCircle, FiSettings } from "react-icons/fi";
import InstructionsModal from './InstructionsModal';
import SettingsPanel from './SettingsPanel';

export default function HeaderMenu({ settings, onSettingsChange, settingsDisabled }) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [soundOn, setSoundOn] = useState(settings ? !!settings.sound_on : true);
  const [helpOpen, setHelpOpen] = useState(false);

  // when sound is toggled via settings panel, propagate using onSettingsChange
  const toggleSound = () => {
    setSoundOn(prev => {
      const next = !prev;
      if (settings && onSettingsChange) {
        onSettingsChange({ ...settings, sound_on: next });
      }
      return next;
    });
  };

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
            <SettingsPanel
              disabled={settingsDisabled}
              settings={settings}
              onChange={onSettingsChange}
            />
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
