"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ShieldAlert, Phone, BookOpen } from "lucide-react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  searchScore?: number;
  accuracyScore?: number;
  sources?: string[];
}

function getConfidenceBadge(score: number | undefined) {
  if (score === undefined) return null;
  if (score >= 0.5) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
        🟢 High confidence
      </span>
    );
  }
  if (score >= 0.35) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-yellow-100 px-2 py-0.5 text-xs font-medium text-yellow-800">
        🟡 Medium confidence
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-800">
      🔴 Low confidence
    </span>
  );
}

export default function ChatMessage({
  role,
  content,
  searchScore,
  accuracyScore,
  sources,
}: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false);
  const isUser = role === "user";

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser
            ? "bg-blue-600 text-white"
            : "bg-white text-gray-900 shadow-sm border border-gray-200"
        }`}
      >
        {/* Message text */}
        <div className="whitespace-pre-wrap text-sm leading-relaxed">{content}</div>

        {/* Assistant-only metadata */}
        {!isUser && (
          <div className="mt-3 space-y-2 border-t border-gray-100 pt-2">
            {/* Confidence & faithfulness badges */}
            <div className="flex flex-wrap gap-2">
              {getConfidenceBadge(searchScore)}
              {accuracyScore !== undefined && (
                <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                  <ShieldAlert className="h-3 w-3" />
                  Faithfulness: {(accuracyScore * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* Sources expander */}
            {sources && sources.length > 0 && (
              <div>
                <button
                  onClick={() => setShowSources(!showSources)}
                  className="flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-800"
                >
                  <BookOpen className="h-3 w-3" />
                  {showSources ? "Hide sources" : `Show sources (${sources.length})`}
                  {showSources ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                </button>
                {showSources && (
                  <div className="mt-2 space-y-2 rounded-lg bg-gray-50 p-3">
                    {sources.map((src, i) => (
                      <div key={i} className="text-xs text-gray-700">
                        <span className="font-semibold text-gray-900">Source {i + 1}:</span>
                        <p className="mt-0.5 line-clamp-4">{src}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Escalation line — always visible */}
            <div className="flex items-center gap-1.5 text-xs text-gray-500">
              <Phone className="h-3 w-3" />
              <span>Need help? Kisan Call Centre: <strong>1800-180-1551</strong></span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
