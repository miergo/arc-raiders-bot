import { useState } from "react";

interface BotMessageProps {
  answer: string;
  sources: string[];
}

export default function BotMessage({ answer, sources }: BotMessageProps) {
  const [showSources, setShowSources] = useState(false);

  return (
    <div className="bot-message">
      <div className="bot-avatar">ARC</div>
      <div className="bot-bubble">
        <p className="bot-answer">{answer}</p>
        {sources.length > 0 && (
          <div className="sources-section">
            <button
              className="sources-toggle"
              onClick={() => setShowSources((v) => !v)}
            >
              {showSources ? "Hide" : "Show"} sources ({sources.length})
            </button>
            {showSources && (
              <ul className="sources-list">
                {sources.map((url, i) => (
                  <li key={i}>
                    <a href={url} target="_blank" rel="noopener noreferrer">
                      {url}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
