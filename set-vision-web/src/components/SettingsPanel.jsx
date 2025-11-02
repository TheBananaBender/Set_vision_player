import React from 'react';

export default function SettingsPanel({ disabled, settings, onChange }) {
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const v = type === 'checkbox' ? checked : value;
    onChange({ ...settings, [name]: name === 'delay_scale' ? parseFloat(v) : v });
  };

  return (
    <div className="settings-panel">
      <fieldset disabled={disabled}>
        <legend>Settings</legend>
        <label>
          Difficulty:
          <select name="difficulty" value={settings.difficulty} onChange={handleChange}>
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </label>
        <label>
          Delay scale:
          <input name="delay_scale" type="number" step="0.1" min="0.1" max="3" value={settings.delay_scale}
                 onChange={handleChange} />
        </label>
        <label>
          Sound on:
          <input name="sound_on" type="checkbox" checked={!!settings.sound_on} onChange={handleChange} />
        </label>
      </fieldset>
      {disabled && <div className="hint">Stop the game to change settings</div>}
    </div>
  );
}


