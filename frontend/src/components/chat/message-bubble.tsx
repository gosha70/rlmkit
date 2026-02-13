"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, Loader2, Clock, Coins, Hash, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/lib/use-chat";

interface MessageBubbleProps {
  message: ChatMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3", isUser && "flex-row-reverse")} {...(!isUser ? { "aria-live": "polite" } : {})}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
        )}
        aria-hidden="true"
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div className={cn("flex max-w-[75%] flex-col gap-1", isUser && "items-end")}>
        {!isUser && message.mode_used && (
          <Badge variant="outline" className="w-fit text-xs">
            {message.mode_used.toUpperCase()}
          </Badge>
        )}

        <div
          className={cn(
            "rounded-lg px-4 py-3 text-sm",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted",
          )}
        >
          {message.isStreaming && !message.content ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>

        {!isUser && message.isStreaming && message.steps && message.steps.length > 0 && (
          <div className="flex items-center gap-2 text-xs text-muted-foreground" role="status">
            <Activity className="h-3 w-3 animate-pulse" aria-hidden="true" />
            <span>
              Step {message.steps.length}
              {message.steps[message.steps.length - 1]?.content
                ? `: ${message.steps[message.steps.length - 1].content.slice(0, 60)}${message.steps[message.steps.length - 1].content.length > 60 ? "..." : ""}`
                : ""}
            </span>
          </div>
        )}

        {!isUser && message.metrics && !message.isStreaming && (
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Hash className="h-3 w-3" aria-hidden="true" />
              {message.metrics.total_tokens.toLocaleString()} tokens
            </span>
            <span className="flex items-center gap-1">
              <Coins className="h-3 w-3" aria-hidden="true" />
              ${message.metrics.cost_usd.toFixed(4)}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-3 w-3" aria-hidden="true" />
              {message.metrics.elapsed_seconds.toFixed(1)}s
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
