import React from 'react';

export default function CardGrid() {
  const fakeCards = Array.from({ length: 12 }, (_, i) => `Card ${i + 1}`);
  return (
    <div className="card-grid">
      {fakeCards.map(card => (
        <div key={card} className="card">{card}</div>
      ))}
    </div>
  );
}
