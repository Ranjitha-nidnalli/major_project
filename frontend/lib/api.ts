export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  searchScore?: number;
  accuracyScore?: number;
  sources?: string[];
}

export async function sendMessage(
  query: string,
  sessionId: string
): Promise<ChatMessage> {
  const res = await fetch("http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Failed to send message");
  }

  const data = await res.json();

  return {
    role: "assistant",
    content: data.answer,
    searchScore: data.search_score,
    accuracyScore: data.accuracy_score,
    sources: data.sources,
  };
}

export async function getHistory(sessionId: string): Promise<ChatMessage[]> {
  const res = await fetch(`http://localhost:8000/history/${sessionId}`);
  if (!res.ok) throw new Error("Failed to fetch history");
  const data = await res.json();
  return data.messages.map((m: any) => ({
    role: m.role,
    content: m.content,
  }));
}
