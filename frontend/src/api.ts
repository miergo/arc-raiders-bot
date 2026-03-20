export interface ArcRaidersResponse {
  answer: string;
  sources: string[];
  session_id: string;
}

export async function askBot(
  question: string,
  sessionId?: string
): Promise<ArcRaidersResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }

  return res.json();
}
