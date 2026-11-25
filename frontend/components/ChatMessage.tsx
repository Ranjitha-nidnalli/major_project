"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, ShieldAlert, BookOpen } from "lucide-react";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  searchScore?: number | null;
  accuracyScore?: number | null;
  sources?: string[];
  factsStatus?: string | null;
  recordIds?: string[] | null;
  tier?: string | null;
}

const REFUSAL_REASONS: Record<string, string> = {
  refused_unknown_pest: "Refused: pest or disease not in the verified records",
  refused_out_of_scope: "Refused: outside the sugarcane topics covered",
  refused_not_verified: "Refused: no person-confirmed record for this yet",
};

const KCC_NUMBER = "1800-180-1551";

// Tier A: person-confirmed records. Tier B: general advice from cited documents. Tier C: refusal.
function getTierBadge(tier: string | null | undefined) {
  if (tier === "B") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-800">
        ℹ️ General advice (not a dose) · source shown below
      </span>
    );
  }
  return null;
}

function getVerificationBadge(status: string | null | undefined, recordIds: string[] | null | undefined) {
  if (!status) return null;
  if (status === "answered") {
    const n = recordIds?.length ?? 0;
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-800">
        ✅ Verified answer · {n} record{n === 1 ? "" : "s"} checked by a person against the printed page
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800">
      🛑 {REFUSAL_REASONS[status] ?? "Refused"}
    </span>
  );
}

function getConfidenceBadge(score: number | null | undefined) {
  // null = the v2 fact path, which has no retrieval score to show
  if (score == null) return null;
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
  factsStatus,
  recordIds,
  tier,
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
              {getVerificationBadge(factsStatus, recordIds)}
              {getTierBadge(tier)}
              {getConfidenceBadge(searchScore)}
              {accuracyScore != null && (
                <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                  <ShieldAlert className="h-3 w-3" />
                  Faithfulness: {(accuracyScore * 100).toFixed(0)}%
                </span>
              )}
            </div>

            {/* Source line: always visible for verified (A) and general (B) answers */}
            {(tier === "A" || tier === "B") && sources && sources.length > 0 && (
              <p className="text-xs text-gray-700">
                <span className="font-semibold">Source:</span>{" "}
                {Array.from(new Set(sources.map((s) => s.replace(/\s*\[[^\]]*\]\s*$/, "")))).join("; ")}
              </p>
            )}

            {/* A refusal always shows the Kisan Call Centre number, even if the text is cut */}
            {tier === "C" && (
              <p className="rounded-lg bg-amber-50 p-2 text-xs font-medium text-amber-900">
                📞 Kisan Call Centre: <a className="underline" href={`tel:${KCC_NUMBER}`}>{KCC_NUMBER}</a> (toll-free)
              </p>
            )}

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

          </div>
        )}
      </div>
    </div>
  );
}
