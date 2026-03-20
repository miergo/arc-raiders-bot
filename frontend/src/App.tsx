import { useCallback, useRef, useState } from "react";
import ChatWindow, { type ChatMessage } from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import { askBot } from "./api";
import "./App.css";

let nextId = 0;
const uid = () => String(++nextId);

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const sessionRef = useRef<string | undefined>(undefined);

  const handleSend = useCallback(
    async (question: string) => {
      const userMsg: ChatMessage = { id: uid(), role: "user", text: question };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const data = await askBot(question, sessionRef.current);
        sessionRef.current = data.session_id;
        const botMsg: ChatMessage = {
          id: uid(),
          role: "bot",
          text: data.answer,
          sources: data.sources,
        };
        setMessages((prev) => [...prev, botMsg]);
      } catch (err) {
        const errMsg: ChatMessage = {
          id: uid(),
          role: "bot",
          text: `Something went wrong: ${err instanceof Error ? err.message : "Unknown error"}`,
        };
        setMessages((prev) => [...prev, errMsg]);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="header-icon">⚡</span>
        <h1>ARC-RAIDERS WikiBot</h1>
      </header>
      <ChatWindow messages={messages} loading={loading} />
      <ChatInput onSend={handleSend} disabled={loading} />
    </div>
  );
}
