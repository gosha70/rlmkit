"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot } from "lucide-react";
import { AppShell } from "@/components/shared/app-shell";
import { ChatProviderSelector } from "@/components/chat/chat-provider-selector";
import { ChatInput } from "@/components/chat/chat-input";
import { FileAttachment } from "@/components/chat/file-attachment";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  getConfig,
  getSession,
  getChatProviders,
  getTrace,
  submitChat,
  uploadFile,
  getProviders,
  type AppConfig,
  type ChatProviderConfig,
  type ProviderInfo,
  type FileUploadResponse,
  type MessageMetrics,
} from "@/lib/api";

interface ChatTurn {
  id: string;
  userMessage: {
    content: string;
    timestamp: string;
  };
  responses: Record<
    string,
    {
      chatProviderId: string;
      chatProviderName: string;
      executionId: string;
      content: string;
      isStreaming: boolean;
      metrics?: MessageMetrics;
    }
  >;
}

interface PollingState {
  [executionId: string]: ReturnType<typeof setInterval>;
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("rlmkit_active_session") || null;
    }
    return null;
  });

  const [selectedChatProviderIds, setSelectedChatProviderIds] = useState<string[]>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("rlmkit_selected_chat_providers");
      return saved ? JSON.parse(saved) : [];
    }
    return [];
  });

  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isAnyPolling, setIsAnyPolling] = useState(false);
  const skipSessionLoadRef = useRef(false);
  const pollingStateRef = useRef<PollingState>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: chatProviders = [] } = useSWR<ChatProviderConfig[]>("chat-providers", getChatProviders);
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

  // Persist selected chat providers to localStorage
  useEffect(() => {
    localStorage.setItem("rlmkit_selected_chat_providers", JSON.stringify(selectedChatProviderIds));
  }, [selectedChatProviderIds]);

  // Auto-select first available chat provider if none selected
  useEffect(() => {
    if (selectedChatProviderIds.length === 0 && chatProviders.length > 0) {
      setSelectedChatProviderIds([chatProviders[0].id]);
    }
  }, [chatProviders, selectedChatProviderIds.length]);

  // Load session messages and reconstruct turns
  useEffect(() => {
    if (!sessionId) return;
    if (skipSessionLoadRef.current) {
      skipSessionLoadRef.current = false;
      return;
    }
    getSession(sessionId)
      .then((session) => {
        // Reconstruct turns from session.conversations dict or flat messages
        if (session.conversations && Object.keys(session.conversations).length > 0) {
          // Group by conversation (chat provider)
          const turnMap = new Map<string, ChatTurn>();

          // Iterate through each chat provider's conversation
          for (const [cpId, messages] of Object.entries(session.conversations)) {
            for (const msg of messages) {
              let turnId: string | undefined;

              if (msg.role === "user") {
                // Find or create a turn with this user message
                const existing = Array.from(turnMap.values()).find(
                  (t) => t.userMessage.content === msg.content
                );
                turnId = existing?.id || crypto.randomUUID();
              } else {
                // Assistant message — find the corresponding turn
                const existing = Array.from(turnMap.values()).find(
                  (t) => !t.responses[cpId]
                );
                turnId = existing?.id;
              }

              if (!turnId) turnId = crypto.randomUUID();

              let turn = turnMap.get(turnId);
              if (!turn) {
                turn = {
                  id: turnId,
                  userMessage: { content: "", timestamp: "" },
                  responses: {},
                };
                turnMap.set(turnId, turn);
              }

              if (msg.role === "user") {
                turn.userMessage = {
                  content: msg.content,
                  timestamp: msg.timestamp,
                };
              } else {
                turn.responses[cpId] = {
                  chatProviderId: cpId,
                  chatProviderName: msg.chat_provider_name || cpId,
                  executionId: msg.execution_id || crypto.randomUUID(),
                  content: msg.content,
                  isStreaming: false,
                  metrics: msg.metrics ?? undefined,
                };
              }
            }
          }

          setTurns(Array.from(turnMap.values()));
        } else {
          // Fallback: reconstruct from flat messages list
          const newTurns: ChatTurn[] = [];
          let currentTurn: ChatTurn | null = null;

          for (const msg of session.messages) {
            if (msg.role === "user") {
              currentTurn = {
                id: crypto.randomUUID(),
                userMessage: {
                  content: msg.content,
                  timestamp: msg.timestamp,
                },
                responses: {},
              };
              newTurns.push(currentTurn);
            } else if (currentTurn && msg.chat_provider_id) {
              currentTurn.responses[msg.chat_provider_id] = {
                chatProviderId: msg.chat_provider_id,
                chatProviderName: msg.chat_provider_name || msg.chat_provider_id,
                executionId: msg.execution_id || msg.id,
                content: msg.content,
                isStreaming: false,
                metrics: msg.metrics ?? undefined,
              };
            }
          }

          setTurns(newTurns);
        }
      })
      .catch((err) => {
        if (String(err).includes("404")) {
          setSessionId(null);
          setTurns([]);
        }
      });
  }, [sessionId]);

  // Auto-scroll on new turns
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      Object.values(pollingStateRef.current).forEach((interval) => {
        clearInterval(interval);
      });
    };
  }, []);

  const pollForResult = useCallback(
    (executionId: string, chatProviderId: string, turnId: string) => {
      setIsAnyPolling(true);

      const interval = setInterval(async () => {
        try {
          const trace = await getTrace(executionId);

          if (trace.status === "complete") {
            clearInterval(interval);
            delete pollingStateRef.current[executionId];

            setTurns((prev) =>
              prev.map((turn) =>
                turn.id === turnId
                  ? {
                      ...turn,
                      responses: {
                        ...turn.responses,
                        [chatProviderId]: {
                          ...turn.responses[chatProviderId],
                          content: trace.result.answer,
                          isStreaming: false,
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
                        },
                      },
                    }
                  : turn,
              ),
            );

            // Check if any polling is still active
            if (Object.keys(pollingStateRef.current).length === 0) {
              setIsAnyPolling(false);
            }
          } else if (trace.status === "error") {
            clearInterval(interval);
            delete pollingStateRef.current[executionId];

            setTurns((prev) =>
              prev.map((turn) =>
                turn.id === turnId
                  ? {
                      ...turn,
                      responses: {
                        ...turn.responses,
                        [chatProviderId]: {
                          ...turn.responses[chatProviderId],
                          content: `Error: ${trace.result?.answer || "Execution failed"}`,
                          isStreaming: false,
                        },
                      },
                    }
                  : turn,
              ),
            );

            if (Object.keys(pollingStateRef.current).length === 0) {
              setIsAnyPolling(false);
            }
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 1000);

      pollingStateRef.current[executionId] = interval;
    },
    [],
  );

  const handleSend = useCallback(
    async (text: string) => {
      // Add user message
      const turnId = crypto.randomUUID();
      const newTurn: ChatTurn = {
        id: turnId,
        userMessage: {
          content: text,
          timestamp: new Date().toISOString(),
        },
        responses: {},
      };

      // Initialize empty responses for each selected provider
      for (const cpId of selectedChatProviderIds) {
        newTurn.responses[cpId] = {
          chatProviderId: cpId,
          chatProviderName: chatProviders.find((cp) => cp.id === cpId)?.name || cpId,
          executionId: "",
          content: "",
          isStreaming: true,
        };
      }

      setTurns((prev) => [...prev, newTurn]);

      // Submit one request per selected chat provider in parallel
      const submitPromises = selectedChatProviderIds.map(async (cpId) => {
        try {
          const resp = await submitChat({
            query: text,
            content: uploadedFile ? null : text || null,
            file_id: uploadedFile?.id || null,
            chat_provider_id: cpId,
            session_id: sessionId,
          });

          if (!sessionId) {
            skipSessionLoadRef.current = true;
            setSessionId(resp.session_id);
          }

          // Update the turn with the execution ID and start polling
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [cpId]: {
                        ...turn.responses[cpId],
                        executionId: resp.execution_id,
                      },
                    },
                  }
                : turn,
            ),
          );

          pollForResult(resp.execution_id, cpId, turnId);
        } catch (err) {
          console.error(`Failed to submit chat for provider ${cpId}:`, err);
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [cpId]: {
                        ...turn.responses[cpId],
                        content: `Error: Failed to submit request`,
                        isStreaming: false,
                      },
                    },
                  }
                : turn,
            ),
          );
        }
      });

      await Promise.allSettled(submitPromises);
    },
    [sessionId, uploadedFile, selectedChatProviderIds, chatProviders, pollForResult],
  );

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      const resp = await uploadFile(file);
      setUploadedFile(resp);
      if (file.type === "text/plain" || file.name.endsWith(".txt") || file.name.endsWith(".md")) {
        const text = await file.text();
        setFileContent(text);
      } else {
        setFileContent("");
      }
    } catch (err) {
      console.error("Failed to upload file:", err);
    }
  }, []);

  const handleNewSession = useCallback(() => {
    setSessionId(null);
    setTurns([]);
    setUploadedFile(null);
    setFileContent("");
    Object.values(pollingStateRef.current).forEach((interval) => {
      clearInterval(interval);
    });
    pollingStateRef.current = {};
    setIsAnyPolling(false);
  }, []);

  return (
    <AppShell
      activeSessionId={sessionId}
      onSelectSession={setSessionId}
      onNewSession={handleNewSession}
    >
      <div className="flex h-full flex-col overflow-hidden">
        {/* Toolbar */}
        <div className="flex items-center gap-3 border-b px-4 py-2" role="toolbar" aria-label="Chat options">
          <ChatProviderSelector
            chatProviders={chatProviders}
            providers={providers}
            selectedIds={selectedChatProviderIds}
            onSelectionChange={setSelectedChatProviderIds}
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

        {turns.length === 0 ? (
          /* Empty state: centered welcome + input */
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-4">
            <div className="mb-8 text-center text-muted-foreground">
              <p className="text-lg font-medium">Welcome to RLM Studio</p>
              <p className="mt-2 text-sm">
                Upload a document and ask questions. The RLM engine will recursively
                explore the content using code generation to find answers efficiently.
              </p>
            </div>
            <div className="w-full max-w-3xl">
              <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isAnyPolling} />
            </div>
          </div>
        ) : (
          /* Active conversation: turns scroll, input pinned at bottom */
          <>
            <ScrollArea className="min-h-0 flex-1" aria-label="Message history">
              <div className="mx-auto w-full max-w-full space-y-8 px-4 py-6" role="log" aria-live="polite">
                {turns.map((turn) => (
                  <div key={turn.id} className="space-y-4">
                    {/* User message - full width */}
                    <div className="flex gap-3">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <User className="h-4 w-4" aria-hidden="true" />
                      </div>
                      <div className="flex flex-col gap-1">
                        <div className="rounded-lg bg-primary px-4 py-3 text-sm text-primary-foreground">
                          <p className="whitespace-pre-wrap">{turn.userMessage.content}</p>
                        </div>
                      </div>
                    </div>

                    {/* Provider responses - grid layout */}
                    <div
                      className="grid gap-4"
                      style={{
                        gridTemplateColumns: `repeat(${selectedChatProviderIds.length}, 1fr)`,
                      }}
                    >
                      {selectedChatProviderIds.map((cpId) => {
                        const resp = turn.responses[cpId];
                        return (
                          <div
                            key={cpId}
                            className="rounded-lg border bg-muted/30 p-4"
                          >
                            {/* Provider name header */}
                            <div className="mb-2 flex items-center gap-2">
                              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                                <Bot className="h-3 w-3" aria-hidden="true" />
                              </div>
                              <span className="text-xs font-medium text-muted-foreground">
                                {resp?.chatProviderName}
                              </span>
                            </div>

                            {/* Response content or loading indicator */}
                            {resp && resp.content ? (
                              <>
                                <div className="prose prose-sm dark:prose-invert max-w-none">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {resp.content}
                                  </ReactMarkdown>
                                </div>
                                {resp.metrics && (
                                  <div className="mt-3 flex flex-col gap-2 border-t pt-2 text-xs text-muted-foreground">
                                    <div className="flex gap-4">
                                      <span>{resp.metrics.total_tokens} tokens</span>
                                      <span>${resp.metrics.cost_usd.toFixed(4)}</span>
                                      <span>{resp.metrics.elapsed_seconds.toFixed(1)}s</span>
                                    </div>
                                  </div>
                                )}
                              </>
                            ) : resp ? (
                              <TypingIndicator />
                            ) : null}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                {isAnyPolling && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
            <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isAnyPolling} />
          </>
        )}
      </div>
    </AppShell>
  );
}
