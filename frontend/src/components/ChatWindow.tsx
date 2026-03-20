import { useEffect, useRef } from "react";
import BotMessage from "./BotMessage";

export interface ChatMessage {
  id: string;
  role: "user" | "bot";
  text: string;
  sources?: string[];
}

interface ChatWindowProps {
  messages: ChatMessage[];
  loading: boolean;
}

export default function ChatWindow({ messages, loading }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="chat-window">
      {messages.length === 0 && !loading && (
        <div className="chat-empty">
          <div className="chat-empty-icon">⚡</div>
          <h2>ARC-RAIDERS WikiBot</h2>
          <p>Ask me anything about ARC Raiders — weapons, maps, lore, tips, and more.</p>
        </div>
      )}

      {messages.map((msg) =>
        msg.role === "user" ? (
          <div key={msg.id} className="user-message">
            <div className="user-bubble">{msg.text}</div>
          </div>
        ) : (
          <BotMessage
            key={msg.id}
            answer={msg.text}
            sources={msg.sources ?? []}
          />
        )
      )}

      {loading && (
        <div className="bot-message">
          <div className="bot-avatar">ARC</div>
          <div className="bot-bubble loading-bubble">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
