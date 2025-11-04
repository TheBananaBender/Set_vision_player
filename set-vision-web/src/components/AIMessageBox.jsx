import React from 'react';
import './AIMessageBox.css';

export default function AIMessageBox({ message, state }) {
  // Map states to visual styles
  const getStateClass = () => {
    // Check if message contains "Darn!" for special styling
    if (message && message.includes("Darn!")) {
      return 'ai-frustrated';
    }
    
    switch (state) {
      case 'thinking': return 'ai-thinking';
      case 'found_set': return 'ai-success';
      case 'no_sets': return 'ai-waiting';
      default: return 'ai-idle';
    }
  };

  return (
    <div className={`ai-message-box ${getStateClass()}`}>
      <div className="ai-label">🤖 AI Agent</div>
      <div className="ai-message">{message || 'Waiting...'}</div>
    </div>
  );
}

