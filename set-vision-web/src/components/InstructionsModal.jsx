// InstructionsModal.jsx
// InstructionsModal.jsx
import React, { useEffect } from "react";
import { FiX } from "react-icons/fi";
  
// import your pics (adjust names if different)
import goodSetImg from "/src/assets/set/good-set.png";
import badTripletImg from "/src/assets/set/bad-triplet.png";

export default function InstructionsModal({ open, onClose }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="setInstrTitle">
      <div className="modal-window">
        {/* Sticky header keeps X visible */}
        <div className="modal-header">
          <h2 id="setInstrTitle">How to Play SET</h2>
          <button className="modal-close" aria-label="Close" onClick={onClose}>
            <FiX size={20} />
          </button>
        </div>

        <p>
          Find sets of 3 cards where, for each attribute{" "}
          <em>(color, shape, fill, quantity)</em>, the three cards are either
          all the same or all different.
        </p>

        {/* Your images */}
        <div className="fig-grid">
          <figure className="fig">
            <img src={goodSetImg} alt="Example of a valid SET (good triplet)" loading="lazy" />
            <figcaption>Valid SET — every attribute is either all same or all different ✅</figcaption>
          </figure>

          <figure className="fig">
            <img src={badTripletImg} alt="Example of an invalid triplet (not a SET)" loading="lazy" />
            <figcaption>Not a SET — at least one attribute has 2 same + 1 different ❌</figcaption>
          </figure>
        </div>

        <h3>Attributes</h3>
        <ul className="attr-list">
          <li><strong>Color:</strong> red, blue, green</li>
          <li><strong>Shape:</strong> diamond, oval, squiggle</li>
          <li><strong>Fill:</strong> solid, striped, empty</li>
          <li><strong>Quantity:</strong> 1, 2, or 3 symbols</li>
        </ul>

        <h3>Game Flow vs AI</h3>
        <ol>
          <li>Start with an empty camera view.</li>
          <li>Deal 12 cards, then remove your hands from the frame.</li>
          <li>Try to find a SET before the AI agent.</li>
          <li>Remove the set from the board and deal replacements.</li>
          <li>Repeat until the deck is empty.</li>
        </ol>

        <p className="tip">
          Tip: For each attribute, scan: <em>all same or all different?</em> If any attribute has exactly
          two the same and one different — it’s NOT a set.
        </p>
      </div>
    </div>
  );
}



/* --- tiny, self-contained “card” with inline SVGs --- */
function MiniCard({ color = "#e63946", shape = "oval", fill = "solid", qty = 1 }) {
  const symbols = Array.from({ length: qty });

  const symbol = (i) => {
    const commonProps = {
      stroke: color,
      strokeWidth: 2,
      fill:
        fill === "solid"
          ? color
          : fill === "empty"
          ? "none"
          : "url(#stripePattern)",
    };

    switch (shape) {
      case "diamond":
        return (
          <polygon
            key={i}
            points="50,10 90,50 50,90 10,50"
            {...commonProps}
          />
        );
      case "squiggle":
        return (
          <path
            key={i}
            d="M20,60 C30,20 70,100 80,60 C90,20 30,100 20,60 Z"
            {...commonProps}
          />
        );
      default: // oval
        return <ellipse key={i} cx="50" cy="50" rx="35" ry="22" {...commonProps} />;
    }
  };

  return (
    <div className="mini-card">
      <svg viewBox="0 0 100 100" width="100%" height="100%">
        {/* stripes pattern */}
        <defs>
          <pattern id="stripePattern" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="3" height="6" fill={color} />
          </pattern>
        </defs>

        {/* Layout symbols vertically depending on qty */}
        {symbols.map((_, i) => (
          <g key={i} transform={`translate(0, ${qty === 1 ? 0 : qty === 2 ? (i === 0 ? -15 : 15) : i === 0 ? -24 : i === 1 ? 0 : 24})`}>
            {symbol(i)}
          </g>
        ))}
      </svg>
    </div>
  );
}
