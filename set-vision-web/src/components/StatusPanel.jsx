import React from 'react';

export default function StatusPanel({ status }) {
  return (
    <div className="status-panel">
      <p>Status: {status}</p>
    </div>
  );
}
