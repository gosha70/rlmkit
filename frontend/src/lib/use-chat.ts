"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { connectChatWS, type WSMessage, type ChatMode } from "./api";

export interface StepData {
  step: number;
  role: string;
  content: string;
  input_tokens?: number;
  output_tokens?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: ChatMode;
  mode_used?: ChatMode;
  metrics?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    cost_usd: number;
    elapsed_seconds: number;
    steps: number;
  };
  steps?: StepData[];
  isStreaming?: boolean;
}

interface UseChatReturn {
  messages: ChatMessage[];
  isConnected: boolean;
  isStreaming: boolean;
  sendQuery: (query: string, content: string, mode?: ChatMode, fileId?: string) => void;
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
}

const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

export function useChat(sessionId: string | null): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const streamBufferRef = useRef<string>("");
  const currentMsgIdRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const intentionalCloseRef = useRef(false);

  useEffect(() => {
    if (!sessionId) return;

    function connect() {
      const ws = connectChatWS(sessionId!);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        reconnectAttemptsRef.current = 0;
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;

        if (
          !intentionalCloseRef.current &&
          reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS
        ) {
          const delay = Math.min(
            BASE_RECONNECT_DELAY_MS * Math.pow(2, reconnectAttemptsRef.current),
            MAX_RECONNECT_DELAY_MS,
          );
          reconnectAttemptsRef.current += 1;
          console.debug(
            `WebSocket closed unexpectedly. Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`,
          );
          reconnectTimerRef.current = setTimeout(connect, delay);
        }
      };

      ws.onerror = () => {
        // WS connection errors are expected when the backend is restarting
        // or during initial connection attempts. Reconnect logic handles recovery.
      };

      ws.onmessage = (event) => {
        const msg: WSMessage = JSON.parse(event.data);

        switch (msg.type) {
          case "connected":
            break;

          case "token":
            if (msg.id) {
              streamBufferRef.current += String(msg.data ?? "");
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === msg.id ? { ...m, content: streamBufferRef.current } : m,
                ),
              );
            }
            break;

          case "step": {
            const stepData = msg.data as StepData;
            if (msg.id) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === msg.id
                    ? { ...m, steps: [...(m.steps ?? []), stepData] }
                    : m,
                ),
              );
            }
            break;
          }

          case "metrics": {
            const metricsData = msg.data as ChatMessage["metrics"];
            if (msg.id && metricsData) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === msg.id ? { ...m, metrics: metricsData } : m,
                ),
              );
            }
            break;
          }

          case "complete": {
            setIsStreaming(false);
            const data = msg.data as Record<string, unknown>;
            const metrics = data?.metrics as ChatMessage["metrics"];
            if (msg.id) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === msg.id
                    ? {
                        ...m,
                        content: String(data?.answer ?? m.content),
                        mode_used: (data?.mode as ChatMode) ?? m.mode_used,
                        metrics,
                        isStreaming: false,
                      }
                    : m,
                ),
              );
            }
            currentMsgIdRef.current = null;
            break;
          }

          case "error": {
            setIsStreaming(false);
            const errData = msg.data as Record<string, unknown>;
            if (msg.id) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === msg.id
                    ? {
                        ...m,
                        content: `Error: ${errData?.message ?? "Unknown error"}`,
                        isStreaming: false,
                      }
                    : m,
                ),
              );
            }
            currentMsgIdRef.current = null;
            break;
          }

          case "ping":
            ws.send(JSON.stringify({ type: "pong" }));
            break;
        }
      };
    }

    intentionalCloseRef.current = false;
    reconnectAttemptsRef.current = 0;
    connect();

    return () => {
      intentionalCloseRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [sessionId]);

  const sendQuery = useCallback(
    (query: string, content: string, mode: ChatMode = "auto", fileId?: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      const msgId = crypto.randomUUID();
      currentMsgIdRef.current = msgId;
      streamBufferRef.current = "";

      // Add user message
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: query },
      ]);

      // Add placeholder assistant message
      setMessages((prev) => [
        ...prev,
        { id: msgId, role: "assistant", content: "", mode, isStreaming: true },
      ]);

      setIsStreaming(true);

      const msg: Record<string, unknown> = {
        type: "query",
        id: msgId,
        query,
        content,
        mode,
      };
      if (fileId) msg.file_id = fileId;

      wsRef.current.send(JSON.stringify(msg));
    },
    [],
  );

  return { messages, isConnected, isStreaming, sendQuery, setMessages };
}
