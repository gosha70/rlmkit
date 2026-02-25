"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { ModeSelector, type Strategy } from "@/components/chat/mode-selector";
import { MessageBubble } from "@/components/chat/message-bubble";
import { ChatInput } from "@/components/chat/chat-input";
import { FileAttachment } from "@/components/chat/file-attachment";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat, type ChatMessage } from "@/lib/use-chat";
import {
  getConfig,
  getSession,
  getProviders,
  getTrace,
  submitChat,
  uploadFile,
  type AppConfig,
  type ProviderInfo,
  type FileUploadResponse,
} from "@/lib/api";

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("rlmkit_active_session") || null;
    }
    return null;
  });
  const [modes, setModes] = useState<Strategy[]>(["direct"]);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [isPolling, setIsPolling] = useState(false);
  const skipSessionLoadRef = useRef(false);
  const { messages, isConnected, isStreaming, sendQuery, setMessages } = useChat(sessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { data: providers = [] } = useSWR<ProviderInfo[]>("providers", getProviders);
  const { data: config } = useSWR<AppConfig>("config", getConfig);

  // Persist active session to localStorage
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem("rlmkit_active_session", sessionId);
    } else {
      localStorage.removeItem("rlmkit_active_session");
    }
  }, [sessionId]);

  // Select provider from config.active_provider, falling back to first configured
  useEffect(() => {
    if (selectedProvider) return;
    if (config?.active_provider) {
      setSelectedProvider(config.active_provider);
      return;
    }
    const connected = providers.find((p) => p.status === "connected");
    const configured = providers.find((p) => p.status === "configured");
    const pick = connected || configured;
    if (pick) setSelectedProvider(pick.name);
  }, [providers, config, selectedProvider]);

  // Load session messages when sessionId changes (e.g., clicking a session in sidebar).
  // Skip if we just created this session via submitChat — messages are already set correctly.
  useEffect(() => {
    if (!sessionId) return;
    if (skipSessionLoadRef.current) {
      skipSessionLoadRef.current = false;
      return;
    }
    getSession(sessionId)
      .then((session) => {
        setMessages(
          session.messages.map((m) => ({
            id: m.id,
            role: m.role as "user" | "assistant",
            content: m.content,
            mode: m.mode ?? undefined,
            mode_used: m.mode_used ?? undefined,
            metrics: m.metrics ?? undefined,
          })),
        );
      })
      .catch((err) => {
        // Session no longer exists (e.g., backend restarted) — clear stale ID
        if (String(err).includes("404")) {
          setSessionId(null);
          setMessages([]);
        }
      });
  }, [sessionId, setMessages]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const pollForResult = useCallback(
    (executionId: string) => {
      setIsPolling(true);
      pollingRef.current = setInterval(async () => {
        try {
          const trace = await getTrace(executionId);
          if (trace.status === "complete") {
            clearInterval(pollingRef.current!);
            pollingRef.current = null;
            setIsPolling(false);
            setMessages((prev: ChatMessage[]) =>
              prev.map((msg) =>
                msg.id === executionId
                  ? {
                      ...msg,
                      content: trace.result.answer,
                      isStreaming: false,
                      mode_used: trace.mode,
                      metrics: {
                            input_tokens: trace.budget.tokens_used,
                            output_tokens: 0,
                            total_tokens: trace.budget.tokens_used,
                            cost_usd: trace.budget.cost_used,
                            elapsed_seconds:
                              trace.completed_at && trace.started_at
                                ? (new Date(trace.completed_at).getTime() -
                                    new Date(trace.started_at).getTime()) /
                                  1000
                                : 0,
                            steps: trace.budget.steps_used,
                          },
                    }
                  : msg,
              ),
            );
          } else if (trace.status === "error") {
            clearInterval(pollingRef.current!);
            pollingRef.current = null;
            setIsPolling(false);
            setMessages((prev: ChatMessage[]) =>
              prev.map((msg) =>
                msg.id === executionId
                  ? {
                      ...msg,
                      content: `Error: ${trace.result?.answer || "Execution failed"}`,
                      isStreaming: false,
                    }
                  : msg,
              ),
            );
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 1000);
    },
    [setMessages],
  );

  const handleSend = useCallback(
    async (text: string) => {
      const providerInfo = providers.find((p) => p.name === selectedProvider);
      // Use config.active_model when the selected provider matches the active one
      const model =
        (config?.active_provider === selectedProvider ? config?.active_model : undefined) ||
        providerInfo?.default_model ||
        undefined;

      // Submit one request per selected mode
      for (const currentMode of modes) {
        if (isConnected && sessionId) {
          sendQuery(text, fileContent || "", currentMode, uploadedFile?.id);
        } else {
          try {
            const resp = await submitChat({
              query: text,
              // When a file is uploaded, use file_id so backend uses its extracted text.
              // Only send raw content for plain text typed/pasted without file upload.
              content: uploadedFile ? null : (text || null),
              file_id: uploadedFile?.id || null,
              mode: currentMode,
              provider: selectedProvider || undefined,
              model: model || undefined,
              session_id: sessionId,
            });
            if (!sessionId) {
              // Prevent the session-load useEffect from overwriting our messages
              skipSessionLoadRef.current = true;
              setSessionId(resp.session_id);
            }

            setMessages((prev: ChatMessage[]) => [
              ...prev,
              ...(prev.some((m) => m.role === "user" && m.content === text)
                ? []
                : [{ id: crypto.randomUUID(), role: "user" as const, content: text }]),
              {
                id: resp.execution_id,
                role: "assistant" as const,
                content: "Processing...",
                mode: currentMode,
                isStreaming: true,
              },
            ]);

            // Start polling for the result
            pollForResult(resp.execution_id);
          } catch (err) {
            console.error("Failed to submit chat:", err);
          }
        }
      }
    },
    [isConnected, sessionId, fileContent, uploadedFile, modes, selectedProvider, providers, config, sendQuery, setMessages, pollForResult],
  );

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      const resp = await uploadFile(file);
      setUploadedFile(resp);
      // For text files, read content directly; for binary files (PDF etc.),
      // rely on the backend's extracted text via file_id.
      if (file.type === "text/plain" || file.name.endsWith(".txt") || file.name.endsWith(".md")) {
        const text = await file.text();
        setFileContent(text);
      } else {
        // Backend already extracted text during upload — use file_id to reference it
        setFileContent("");
      }
    } catch (err) {
      console.error("Failed to upload file:", err);
    }
  }, []);

  const handleNewSession = useCallback(() => {
    setSessionId(null);
    setMessages([]);
    setUploadedFile(null);
    setFileContent("");
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setIsPolling(false);
  }, [setMessages]);

  const showTyping = isStreaming || isPolling;

  return (
    <AppShell
      activeSessionId={sessionId}
      onSelectSession={setSessionId}
      onNewSession={handleNewSession}
    >
      <div className="flex h-full flex-col overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center gap-3 border-b px-4 py-2" role="toolbar" aria-label="Chat options">
          <ModeSelector
            modes={modes}
            onModesChange={setModes}
            providers={providers}
            selectedProvider={selectedProvider}
            onProviderChange={setSelectedProvider}
          />
        </div>

        {/* File attachment bar */}
        {uploadedFile && (
          <div className="border-b px-4 py-2">
            <FileAttachment
              name={uploadedFile.name}
              sizeBytes={uploadedFile.size_bytes}
              tokenCount={uploadedFile.token_count}
              onRemove={() => {
                setUploadedFile(null);
                setFileContent("");
              }}
            />
          </div>
        )}

        {messages.length === 0 ? (
          /* Empty state: centered welcome + input like ChatGPT / Claude */
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-4">
            <div className="mb-8 text-center text-muted-foreground">
              <p className="text-lg font-medium">Welcome to RLM Studio</p>
              <p className="mt-2 text-sm">
                Upload a document and ask questions. The RLM engine will recursively
                explore the content using code generation to find answers efficiently.
              </p>
            </div>
            <div className="w-full max-w-3xl">
              <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={showTyping} />
            </div>
          </div>
        ) : (
          /* Active conversation: messages scroll, input pinned at bottom */
          <>
            <ScrollArea className="min-h-0 flex-1" aria-label="Message history">
              <div className="mx-auto max-w-3xl space-y-6 px-4 py-6" role="log" aria-live="polite">
                {messages.map((msg) => (
                  <MessageBubble key={msg.id} message={msg} />
                ))}
                {showTyping && <TypingIndicator mode={modes[0]} />}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
            <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={showTyping} />
          </>
        )}
      </div>
    </AppShell>
  );
}
