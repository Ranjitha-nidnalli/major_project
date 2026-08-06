"use client";

import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Leaf,
  Send,
  Sprout,
  Sun,
  Pencil,
  Trash2,
  Droplets,
  Bug,
  Tractor,
  Plus,
  Wheat,
  CloudRain,
} from "lucide-react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  searchScore?: number;
  accuracyScore?: number;
}

interface ChatSession {
  id: string;
  title: string;
  date: string;
}

const suggestedQuestions = [
  {
    icon: Wheat,
    title: "ಕಬ್ಬು ನೆಡುವಿಕೆ",
    subtitle: "Sugarcane Planting",
    text: "ಕಬ್ಬು ನೆಡುವ ಉತ್ತಮ ಸಮಯ ಯಾವುದು?",
  },
  {
    icon: Bug,
    title: "ಕೀಟ ನಿಯಂತ್ರಣ",
    subtitle: "Pest Control",
    text: "ಕಬ್ಬಿನ ಕೀಟಗಳನ್ನು ಹೇಗೆ ನಿಯಂತ್ರಿಸುವುದು?",
  },
  {
    icon: Droplets,
    title: "ನೀರಾವರಿ",
    subtitle: "Irrigation",
    text: "ಕಬ್ಬಿಗೆ ನೀರಾವರಿ ವೇಳಾಪಟ್ಟಿ ಹೇಗಿರಬೇಕು?",
  },
  {
    icon: CloudRain,
    title: "ಗೊಬ್ಬರ ಪ್ರಮಾಣ",
    subtitle: "Fertilizer Dose",
    text: "ಕಬ್ಬಿಗೆ ಸಾರಜನಕ ಪ್ರಮಾಣ ಎಷ್ಟು ಕೊಡಬೇಕು?",
  },
];

function getMatchStrength(searchScore?: number) {
  if (searchScore === undefined) return null;
  if (searchScore >= 0.02) return { label: "ಬಲವಾದ ಹೊಂದಾಣಿಕೆ", className: "text-primary" };
  if (searchScore >= 0.008) return { label: "ಮಧ್ಯಮ ಹೊಂದಾಣಿಕೆ", className: "text-amber-600 dark:text-amber-400" };
  return { label: "ದುರ್ಬಲ ಹೊಂದಾಣಿಕೆ", className: "text-muted-foreground" };
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isSidebarVisible, setIsSidebarVisible] = useState(true);
  const [showSidebarOpenButton, setShowSidebarOpenButton] = useState(false);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [sessionId, setSessionId] = useState(() => `session_${Date.now()}`);
  const [editingChatId, setEditingChatId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (messages.length === 0) return;
    const firstUserMessage = messages.find((m) => m.role === "user");
    if (!firstUserMessage) return;

    const sessionTitle =
      firstUserMessage.content.length > 28
        ? `${firstUserMessage.content.slice(0, 28)}...`
        : firstUserMessage.content;

    setChatSessions((prev) => {
      const existingIndex = prev.findIndex((session) => session.id === sessionId);
      if (existingIndex >= 0) {
        const updated = [...prev];
        updated[existingIndex] = {
          ...updated[existingIndex],
          title: updated[existingIndex].title || sessionTitle,
          date: "ಈಗ",
        };
        return updated;
      }
      return [{ id: sessionId, title: sessionTitle, date: "ಈಗ" }, ...prev];
    });
  }, [messages, sessionId]);

  useEffect(() => {
    if (isSidebarVisible) {
      setShowSidebarOpenButton(false);
      return;
    }

    const timeout = window.setTimeout(() => {
      setShowSidebarOpenButton(true);
    }, 300);

    return () => window.clearTimeout(timeout);
  }, [isSidebarVisible]);

  const handleNewConversation = () => {
    setMessages([]);
    setInput("");
    const newSessionId = `session_${Date.now()}`;
    setSessionId(newSessionId);
    setEditingChatId(null);
    setEditingTitle("");
    if (window.innerWidth < 768) {
      setIsSidebarVisible(false);
    }
  };

  const startRenameChat = (chat: ChatSession) => {
    setEditingChatId(chat.id);
    setEditingTitle(chat.title);
  };

  const saveRenameChat = (chatId: string) => {
    const newTitle = editingTitle.trim();
    if (!newTitle) {
      setEditingChatId(null);
      setEditingTitle("");
      return;
    }
    setChatSessions((prev) =>
      prev.map((chat) => (chat.id === chatId ? { ...chat, title: newTitle } : chat))
    );
    setEditingChatId(null);
    setEditingTitle("");
  };

  const deleteChat = (chatId: string) => {
    if (!window.confirm("Are you sure?")) return;
    setChatSessions((prev) => prev.filter((chat) => chat.id !== chatId));
    if (sessionId === chatId) {
      handleNewConversation();
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userText = input.trim();
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: userText,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsTyping(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          query: userText,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to connect to the backend.");
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        timestamp: new Date(),
        searchScore: data.search_score,
        accuracyScore: data.accuracy_score,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat API Error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "ಕ್ಷಮಿಸಿ, ತಾಂತ್ರಿಕ ದೋಷ. ಸರ್ವರ್ ಸಂಪರ್ಕ ಹೊಂದಿಲ್ಲ. (Sorry, connection error.)",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSuggestionClick = (text: string) => {
    setInput(text);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-background">
      {showSidebarOpenButton && (
        <button
          type="button"
          onClick={() => setIsSidebarVisible(true)}
          className="fixed left-4 top-3 z-50 flex h-10 w-10 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-md cursor-pointer transition-transform duration-200 hover:scale-110 hover:opacity-90"
          aria-label="Show sidebar"
        >
          <Leaf className="h-5 w-5" />
        </button>
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "relative z-auto flex flex-col bg-sidebar text-sidebar-foreground transition-all duration-300 ease-in-out",
          isSidebarVisible ? "w-64 overflow-hidden" : "w-0 overflow-hidden"
        )}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between border-b border-sidebar-border p-4">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setIsSidebarVisible(false)}
              className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary cursor-pointer transition-transform duration-200 hover:scale-110 hover:opacity-80"
              aria-label="Hide sidebar"
            >
              <Leaf className="h-5 w-5 text-primary-foreground" />
            </button>
            <div>
              <h1 className="font-semibold tracking-tight">ಕೃಷಿ ಸಹಾಯಕ</h1>
              <p className="text-xs text-sidebar-foreground/70">
                AI Farm Assistant
              </p>
            </div>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <Button
            className="w-full justify-start gap-2 bg-sidebar-accent text-sidebar-accent-foreground hover:bg-sidebar-accent/80"
            onClick={handleNewConversation}
          >
            <Plus className="h-4 w-4" />
            ಹೊಸ ಸಂಭಾಷಣೆ
          </Button>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <h2 className="mb-3 text-xs font-medium uppercase tracking-wider text-sidebar-foreground/50">
            ಇತ್ತೀಚಿನ ಚಾಟ್‌ಗಳು
          </h2>
          {chatSessions.length === 0 ? (
            <p className="px-3 py-2 text-sm text-sidebar-foreground/60">
              ಇನ್ನೂ ಚಾಟ್ ಇತಿಹಾಸ ಇಲ್ಲ.
            </p>
          ) : (
            <div className="space-y-1">
              {chatSessions.map((chat) => (
                <div
                  key={chat.id}
                  className="group relative flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-colors hover:bg-sidebar-accent"
                >
                  <Sprout className="h-4 w-4 shrink-0 text-sidebar-foreground/50" />
                  <div className="min-w-0 flex-1 pr-14">
                    {editingChatId === chat.id ? (
                      <input
                        value={editingTitle}
                        onChange={(e) => setEditingTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") saveRenameChat(chat.id);
                        }}
                        onBlur={() => saveRenameChat(chat.id)}
                        autoFocus
                        className="w-full rounded bg-sidebar-accent px-2 py-1 text-sm text-sidebar-foreground outline-none"
                      />
                    ) : (
                      <p className="truncate font-medium">{chat.title}</p>
                    )}
                    <p className="text-xs text-sidebar-foreground/50">
                      {chat.date}
                    </p>
                  </div>
                  <div className="absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        startRenameChat(chat);
                      }}
                      className="rounded p-1 text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground"
                      aria-label="Rename chat"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteChat(chat.id);
                      }}
                      className="rounded p-1 text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground"
                      aria-label="Delete chat"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Sidebar Footer */}
        <div className="border-t border-sidebar-border p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-sidebar-accent">
              <Tractor className="h-4 w-4" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">ಕೃಷಿ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್</p>
              <p className="text-xs text-sidebar-foreground/50">
                ನಿಮ್ಮ ಹೊಲದ ಮಾಹಿತಿ
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex flex-1 flex-col">
        {/* Header */}
        <header
          className={cn(
            "flex h-16 items-center justify-end border-b border-border bg-card px-4",
            !isSidebarVisible && "pl-16"
          )}
        >
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Sun className="h-4 w-4" />
            <span>ಕರ್ನಾಟಕ</span>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center px-4 py-12">
              <div className="mb-8 flex h-20 w-20 items-center justify-center rounded-2xl bg-primary/10">
                <Leaf className="h-10 w-10 text-primary" />
              </div>
              <h2 className="mb-2 text-2xl font-semibold tracking-tight text-foreground text-center">
                ನಮಸ್ಕಾರ! ಕೃಷಿ ಸಹಾಯಕಕ್ಕೆ ಸ್ವಾಗತ
              </h2>
              <p className="mb-2 text-lg text-muted-foreground">
                Welcome to Agriculture Assistant
              </p>
              <p className="mb-8 max-w-md text-center text-muted-foreground text-sm">
                ಬೆಳೆಗಳು, ಮಣ್ಣಿನ ಆರೋಗ್ಯ, ಕೀಟ ನಿರ್ವಹಣೆ, ಮಳೆ ಮಾಹಿತಿ ಅಥವಾ ಯಾವುದೇ ಕೃಷಿ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಕನ್ನಡದಲ್ಲಿ ಕೇಳಿ.
              </p>

              {/* Suggested Questions */}
              <div className="grid w-full max-w-2xl gap-3 sm:grid-cols-2">
                {suggestedQuestions.map((q, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestionClick(q.text)}
                    className="group flex items-center gap-3 rounded-xl border border-border bg-card p-4 text-left transition-all hover:border-primary/50 hover:bg-primary/5 hover:shadow-sm"
                  >
                    <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-primary/10 transition-colors group-hover:bg-primary/20">
                      <q.icon className="h-6 w-6 text-primary" />
                    </div>
                    <div>
                      <span className="block text-sm font-semibold text-foreground">
                        {q.title}
                      </span>
                      <span className="block text-xs text-muted-foreground">
                        {q.subtitle}
                      </span>
                    </div>
                  </button>
                ))}
              </div>

              {/* Karnataka Crops Info */}
              <div className="mt-8 flex flex-wrap justify-center gap-2">
                {["ಕಬ್ಬು", "ಕೆಂಪು ಕೊಳೆ ರೋಗ", "ಕಬ್ಬು ಕೀಟ ನಿಯಂತ್ರಣ", "ಸಾರಜನಕ ಪ್ರಮಾಣ", "NPK ವೇಳಾಪಟ್ಟಿ"].map((crop) => (
                  <span
                    key={crop}
                    className="rounded-full bg-secondary px-3 py-1 text-xs font-medium text-secondary-foreground"
                  >
                    {crop}
                  </span>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl px-4 py-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "mb-6 flex gap-4",
                    message.role === "user" ? "flex-row-reverse" : ""
                  )}
                >
                  <div
                    className={cn(
                      "flex h-9 w-9 shrink-0 items-center justify-center rounded-full",
                      message.role === "assistant"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-secondary-foreground"
                    )}
                  >
                    {message.role === "assistant" ? (
                      <Leaf className="h-4 w-4" />
                    ) : (
                      <span className="text-sm font-medium">ನಾ</span>
                    )}
                  </div>
                  <div
                    className={cn(
                      "max-w-[80%] rounded-2xl px-4 py-3",
                      message.role === "assistant"
                        ? "bg-card border border-border text-card-foreground"
                        : "bg-primary text-primary-foreground"
                    )}
                  >
                    <p className="text-sm leading-relaxed">{message.content}</p>
                    {message.role === "assistant" && message.accuracyScore !== undefined && (
                      <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                        <span>
                          ವಿಶ್ವಾಸಾರ್ಹತೆ: {Math.round(message.accuracyScore * 100)}%
                        </span>
                        {getMatchStrength(message.searchScore) && (
                          <>
                            <span aria-hidden="true">·</span>
                            <span className={getMatchStrength(message.searchScore)!.className}>
                              {getMatchStrength(message.searchScore)!.label}
                            </span>
                          </>
                        )}
                      </div>
                    )}
                    <p
                      className={cn(
                        "mt-2 text-xs",
                        message.role === "assistant"
                          ? "text-muted-foreground"
                          : "text-primary-foreground/70"
                      )}
                    >
                      {message.timestamp.toLocaleTimeString("kn-IN", {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </p>
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="mb-6 flex gap-4">
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                    <Leaf className="h-4 w-4" />
                  </div>
                  <div className="rounded-2xl border border-border bg-card px-4 py-3">
                    <div className="flex gap-1">
                      <span className="h-2 w-2 animate-bounce rounded-full bg-primary/60 [animation-delay:-0.3s]" />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-primary/60 [animation-delay:-0.15s]" />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-primary/60" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-border bg-card p-4">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-end gap-3 rounded-2xl border border-border bg-background p-2 focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/20">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕನ್ನಡದಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ..."
                className="max-h-32 min-h-[44px] flex-1 resize-none bg-transparent px-2 py-2 text-sm placeholder:text-muted-foreground focus:outline-none"
                rows={1}
              />
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isTyping}
                size="icon"
                className="h-10 w-10 shrink-0 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <p className="mt-2 text-center text-xs text-muted-foreground">
              ಕೃಷಿ ಸಹಾಯಕ ತಪ್ಪು ಮಾಡಬಹುದು. ಮುಖ್ಯ ನಿರ್ಧಾರಗಳನ್ನು ಸ್ಥಳೀಯ ಕೃಷಿ ತಜ್ಞರೊಂದಿಗೆ ಪರಿಶೀಲಿಸಿ.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
