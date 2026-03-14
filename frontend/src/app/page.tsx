"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, Sparkles, Loader2 } from "lucide-react";
import { AppShell } from "@/components/shared/app-shell";
import { ChatProviderSelector } from "@/components/chat/chat-provider-selector";
import { ChatInput } from "@/components/chat/chat-input";
import { FileAttachment } from "@/components/chat/file-attachment";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { ResponseRating } from "@/components/chat/response-rating";
import { PickWinner } from "@/components/chat/pick-winner";
import { JudgeScores } from "@/components/chat/judge-scores";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  getConfig,
  getSession,
  getChatProviders,
  getTrace,
  submitChat,
  uploadFile,
  getProviders,
  connectChatWS,
  getSessionEvaluations,
  submitThumbRating,
  removeThumbRating,
  submitBestPick,
  triggerJudge,
  type AppConfig,
  type ChatProviderConfig,
  type ProviderInfo,
  type FileUploadResponse,
  type MessageMetrics,
  type JudgeScoreData,
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
      profileName?: string | null;
      executionId: string;
      content: string;
      isStreaming: boolean;
      metrics?: MessageMetrics;
    }
  >;
}

// Map execution_id -> { turnId, chatProviderId } for WS event matching
interface ExecMapping {
  turnId: string;
  chatProviderId: string;
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

  const [chatProviderOrder, setChatProviderOrder] = useState<string[]>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("rlmkit-cp-order");
      return saved ? JSON.parse(saved) : [];
    }
    return [];
  });

  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [isAnyStreaming, setIsAnyStreaming] = useState(false);
  const [ratings, setRatings] = useState<Record<string, "up" | "down">>({});
  const [bestPicks, setBestPicks] = useState<Record<string, string>>({});
  const [judgeScores, setJudgeScores] = useState<Record<string, JudgeScoreData>>({});
  const [isJudging, setIsJudging] = useState(false);
  const skipSessionLoadRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const execMapRef = useRef<Record<string, ExecMapping>>({});
  const pendingCountRef = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsReadyRef = useRef(false);
  const pollTimersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());
  // Monotonic generation counter — incremented on every session change.
  // Callbacks captured under a stale generation bail out immediately.
  const sessionGenRef = useRef(0);

  // Sort selected IDs by chip order for consistent grid column ordering
  const orderedSelectedIds = chatProviderOrder.length > 0
    ? [...selectedChatProviderIds].sort((a, b) => {
        const ai = chatProviderOrder.indexOf(a);
        const bi = chatProviderOrder.indexOf(b);
        return (ai === -1 ? Infinity : ai) - (bi === -1 ? Infinity : bi);
      })
    : selectedChatProviderIds;

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

  // Persist chip order
  useEffect(() => {
    localStorage.setItem("rlmkit-cp-order", JSON.stringify(chatProviderOrder));
  }, [chatProviderOrder]);

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

  // Load evaluations when session changes
  useEffect(() => {
    if (!sessionId) {
      setRatings({});
      setBestPicks({});
      setJudgeScores({});
      return;
    }
    getSessionEvaluations(sessionId)
      .then((evals) => {
        const newRatings: Record<string, "up" | "down"> = {};
        for (const r of evals.thumb_ratings) {
          newRatings[r.execution_id] = r.rating;
        }
        setRatings(newRatings);

        const newPicks: Record<string, string> = {};
        for (const p of evals.best_picks) {
          newPicks[p.winner_execution_id] = p.winner_execution_id;
        }
        setBestPicks(newPicks);

        const newScores: Record<string, JudgeScoreData> = {};
        for (const s of evals.judge_scores) {
          newScores[s.execution_id] = s;
        }
        setJudgeScores(newScores);
      })
      .catch(() => {
        // Evaluations endpoint may not exist yet
      });
  }, [sessionId]);

  // Auto-scroll on new turns
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  // Connect WebSocket when session exists
  useEffect(() => {
    if (!sessionId) {
      wsReadyRef.current = false;
      return;
    }

    // Close previous connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      wsReadyRef.current = false;
    }

    let ws: WebSocket;
    try {
      ws = connectChatWS(sessionId);
    } catch {
      console.debug("WebSocket connection failed, will use REST fallback");
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      wsReadyRef.current = true;
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const execId = msg.id as string | undefined;
        if (!execId) return;

        const mapping = execMapRef.current[execId];
        if (!mapping) return;

        const { turnId, chatProviderId } = mapping;

        if (msg.type === "token") {
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [chatProviderId]: {
                        ...turn.responses[chatProviderId],
                        content: turn.responses[chatProviderId].content + (msg.data || ""),
                      },
                    },
                  }
                : turn,
            ),
          );
        } else if (msg.type === "complete") {
          const data = msg.data || {};
          delete execMapRef.current[execId];
          pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);

          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [chatProviderId]: {
                        ...turn.responses[chatProviderId],
                        content: data.answer || turn.responses[chatProviderId].content,
                        isStreaming: false,
                        metrics: data.metrics || {
                          input_tokens: data.tokens_used || 0,
                          output_tokens: 0,
                          total_tokens: data.tokens_used || 0,
                          cost_usd: data.cost_used || 0,
                          elapsed_seconds: data.elapsed_seconds || 0,
                          steps: data.steps_used || 0,
                        },
                      },
                    },
                  }
                : turn,
            ),
          );

          if (pendingCountRef.current === 0) {
            setIsAnyStreaming(false);
          }
        } else if (msg.type === "error") {
          delete execMapRef.current[execId];
          pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);

          const errorMsg = msg.data?.message || "Execution failed";
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [chatProviderId]: {
                        ...turn.responses[chatProviderId],
                        content: `Error: ${errorMsg}`,
                        isStreaming: false,
                      },
                    },
                  }
                : turn,
            ),
          );

          if (pendingCountRef.current === 0) {
            setIsAnyStreaming(false);
          }
        }
      } catch {
        // Ignore non-JSON messages (pings, etc.)
      }
    };

    ws.onclose = () => {
      wsReadyRef.current = false;
      console.debug("WebSocket closed for session", sessionId);
    };

    return () => {
      ws.close();
      wsRef.current = null;
      wsReadyRef.current = false;
    };
  }, [sessionId]);

  // REST polling fallback — tracks timeouts so they can be cancelled on cleanup
  const pollForResult = useCallback(
    (executionId: string, chatProviderId: string, turnId: string) => {
      setIsAnyStreaming(true);
      // cancelled flag lets in-flight fetches bail out after cleanup
      let cancelled = false;
      const cancelToken = { cancel: () => { cancelled = true; } };
      // Store on execMap so handleNewSession can cancel
      const mapping = execMapRef.current[executionId];
      if (mapping) (mapping as ExecMapping & { cancelPoll?: () => void }).cancelPoll = cancelToken.cancel;

      const schedulePoll = (delayMs: number) => {
        const timer = setTimeout(poll, delayMs);
        pollTimersRef.current.add(timer);
      };

      const poll = async () => {
        if (cancelled) return;
        try {
          const trace = await getTrace(executionId);
          if (cancelled) return;

          if (trace.status === "complete") {
            delete execMapRef.current[executionId];
            pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);

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

            if (pendingCountRef.current === 0) {
              setIsAnyStreaming(false);
            }
            return;
          } else if (trace.status === "error") {
            delete execMapRef.current[executionId];
            pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);

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

            if (pendingCountRef.current === 0) {
              setIsAnyStreaming(false);
            }
            return;
          }

          // Still running — poll again
          schedulePoll(1000);
        } catch (err) {
          if (cancelled) return;
          console.error("Polling error:", err);
          schedulePoll(2000);
        }
      };

      poll();
    },
    [],
  );

  const handleRate = useCallback(
    async (executionId: string, rating: "up" | "down" | null) => {
      if (!sessionId) return;
      if (rating === null) {
        // Toggle off
        setRatings((prev) => {
          const next = { ...prev };
          delete next[executionId];
          return next;
        });
        try { await removeThumbRating(executionId); } catch { /* ignore */ }
      } else {
        // Find chat_provider_id from turns
        let cpId = "";
        for (const turn of turns) {
          for (const [id, resp] of Object.entries(turn.responses)) {
            if (resp.executionId === executionId) {
              cpId = id;
              break;
            }
          }
          if (cpId) break;
        }
        setRatings((prev) => ({ ...prev, [executionId]: rating }));
        try {
          await submitThumbRating({
            execution_id: executionId,
            session_id: sessionId,
            chat_provider_id: cpId,
            rating,
          });
        } catch { /* ignore */ }
      }
    },
    [sessionId, turns],
  );

  const handlePickWinner = useCallback(
    async (winnerExecutionId: string) => {
      if (!sessionId) return;
      setBestPicks((prev) => ({ ...prev, [winnerExecutionId]: winnerExecutionId }));
      try {
        await submitBestPick({
          session_id: sessionId,
          winner_execution_id: winnerExecutionId,
        });
      } catch { /* ignore */ }
    },
    [sessionId],
  );

  const handleTriggerJudge = useCallback(
    async (executionIds: string[]) => {
      if (!sessionId || executionIds.length === 0) return;
      setIsJudging(true);
      try {
        const result = await triggerJudge({
          session_id: sessionId,
          execution_ids: executionIds,
          mode: "both",
        });
        const newScores: Record<string, JudgeScoreData> = { ...judgeScores };
        for (const s of result.pointwise) {
          newScores[s.execution_id] = s;
        }
        setJudgeScores(newScores);
      } catch (err) {
        console.error("Judge failed:", err);
      } finally {
        setIsJudging(false);
      }
    },
    [sessionId, judgeScores],
  );

  const handleSend = useCallback(
    async (text: string) => {
      // Capture generation at send-time so stale callbacks can bail out
      const sendGen = sessionGenRef.current;

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
        const matchedCp = chatProviders.find((cp) => cp.id === cpId);
        newTurn.responses[cpId] = {
          chatProviderId: cpId,
          chatProviderName: matchedCp?.name || cpId,
          profileName: matchedCp?.profile_name,
          executionId: "",
          content: "",
          isStreaming: true,
        };
      }

      setTurns((prev) => [...prev, newTurn]);
      pendingCountRef.current += selectedChatProviderIds.length;
      setIsAnyStreaming(true);

      // Ensure all providers share one session. When sessionId is null,
      // send the first provider sequentially to establish a session, then
      // fan out the rest in parallel with that shared id.
      let effectiveSessionId = sessionId;

      const submitOne = async (cpId: string) => {
        try {
          const resp = await submitChat({
            query: text,
            content: uploadedFile ? null : text || null,
            file_id: uploadedFile?.id || null,
            chat_provider_id: cpId,
            session_id: effectiveSessionId,
          });

          // If session changed while we were awaiting, discard this result
          if (sessionGenRef.current !== sendGen) return;

          // Capture session id from the first response
          if (!effectiveSessionId) {
            effectiveSessionId = resp.session_id;
            skipSessionLoadRef.current = true;
            setSessionId(resp.session_id);
          }

          // Register mapping so WS events can find the turn
          execMapRef.current[resp.execution_id] = { turnId, chatProviderId: cpId };

          // Update the turn with the execution ID
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

          // Always poll: the WS backend only emits events for WS-initiated
          // queries, not REST-submitted ones. REST polling is the reliable path.
          pollForResult(resp.execution_id, cpId, turnId);
        } catch (err) {
          if (sessionGenRef.current !== sendGen) return;
          console.error(`Failed to submit chat for provider ${cpId}:`, err);
          pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);
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
          if (pendingCountRef.current === 0) {
            setIsAnyStreaming(false);
          }
        }
      };

      if (!sessionId && selectedChatProviderIds.length > 1) {
        // Sequential first request to establish session, then parallel rest
        await submitOne(selectedChatProviderIds[0]);
        if (sessionGenRef.current !== sendGen) return;
        await Promise.allSettled(
          selectedChatProviderIds.slice(1).map((cpId) => submitOne(cpId)),
        );
      } else {
        await Promise.allSettled(
          selectedChatProviderIds.map((cpId) => submitOne(cpId)),
        );
      }
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

  const cancelAllPollers = useCallback(() => {
    sessionGenRef.current += 1;
    for (const mapping of Object.values(execMapRef.current)) {
      (mapping as ExecMapping & { cancelPoll?: () => void }).cancelPoll?.();
    }
    for (const timer of pollTimersRef.current) {
      clearTimeout(timer);
    }
    pollTimersRef.current.clear();
    execMapRef.current = {};
    pendingCountRef.current = 0;
    setIsAnyStreaming(false);
  }, []);

  // Cleanup poll timers on unmount
  useEffect(() => {
    return () => { cancelAllPollers(); };
  }, [cancelAllPollers]);

  const handleNewSession = useCallback(() => {
    cancelAllPollers();
    setSessionId(null);
    setTurns([]);
    setUploadedFile(null);
    setFileContent("");
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      wsReadyRef.current = false;
    }
  }, [cancelAllPollers]);

  const handleSelectSession = useCallback(
    (id: string) => {
      if (id === sessionId) return;
      cancelAllPollers();
      setTurns([]);
      setSessionId(id);
    },
    [sessionId, cancelAllPollers],
  );

  return (
    <AppShell
      activeSessionId={sessionId}
      onSelectSession={handleSelectSession}
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
            orderedIds={chatProviderOrder}
            onOrderChange={setChatProviderOrder}
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
              <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isAnyStreaming} />
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
                        gridTemplateColumns: `repeat(${orderedSelectedIds.length}, 1fr)`,
                      }}
                    >
                      {orderedSelectedIds.map((cpId) => {
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
                                {resp?.profileName && (
                                  <span className="ml-1 font-normal opacity-70">
                                    · {resp.profileName}
                                  </span>
                                )}
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
                                    <div className="flex items-center gap-4">
                                      <span>{resp.metrics.total_tokens} tokens</span>
                                      <span>${resp.metrics.cost_usd.toFixed(4)}</span>
                                      <span>{resp.metrics.elapsed_seconds.toFixed(1)}s</span>
                                      <ResponseRating
                                        executionId={resp.executionId}
                                        sessionId={sessionId || ""}
                                        chatProviderId={cpId}
                                        currentRating={ratings[resp.executionId]}
                                        onRate={handleRate}
                                      />
                                    </div>
                                    {judgeScores[resp.executionId] && (
                                      <JudgeScores
                                        dimensions={judgeScores[resp.executionId].dimensions}
                                        overallScore={judgeScores[resp.executionId].overall_score}
                                      />
                                    )}
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

                    {/* Comparison summary for multi-provider turns */}
                    {(() => {
                      const completed = Object.entries(turn.responses).filter(
                        ([, r]) => r?.metrics,
                      );
                      if (completed.length < 2) return null;
                      const cheapest = completed.reduce((a, b) =>
                        a[1].metrics!.cost_usd <= b[1].metrics!.cost_usd ? a : b,
                      );
                      const fastest = completed.reduce((a, b) =>
                        a[1].metrics!.elapsed_seconds <= b[1].metrics!.elapsed_seconds ? a : b,
                      );
                      return (
                        <div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
                          <div className="flex gap-6">
                            {completed.map(([cpId, r]) => (
                              <div key={cpId} className="space-y-0.5">
                                <span className="font-medium text-foreground">{r.chatProviderName}</span>
                                <div>
                                  {r.metrics!.total_tokens.toLocaleString()} tokens
                                  {" · "}${r.metrics!.cost_usd.toFixed(4)}
                                  {" · "}{r.metrics!.elapsed_seconds.toFixed(1)}s
                                </div>
                              </div>
                            ))}
                          </div>
                          <p className="mt-1">
                            Cheapest:{" "}
                            <span className="font-medium text-emerald-600 dark:text-emerald-400">
                              {cheapest[1].chatProviderName}
                            </span>
                            {" · "}Fastest:{" "}
                            <span className="font-medium text-blue-600 dark:text-blue-400">
                              {fastest[1].chatProviderName}
                            </span>
                            {(() => {
                              // Find best quality from judge scores
                              const scored = completed
                                .filter(([, r]) => judgeScores[r.executionId])
                                .map(([, r]) => ({
                                  name: r.chatProviderName,
                                  score: judgeScores[r.executionId].overall_score,
                                }));
                              if (scored.length === 0) return null;
                              const best = scored.reduce((a, b) =>
                                a.score >= b.score ? a : b
                              );
                              return (
                                <>
                                  {" · "}Best Quality:{" "}
                                  <span className="font-medium text-violet-600 dark:text-violet-400">
                                    {best.name}
                                  </span>
                                </>
                              );
                            })()}
                          </p>
                          <div className="mt-2 flex items-center gap-3">
                            <PickWinner
                              sessionId={sessionId || ""}
                              responses={completed.map(([cpId, r]) => ({
                                executionId: r.executionId,
                                cpId,
                                name: r.chatProviderName,
                              }))}
                              currentWinnerId={
                                completed.find(
                                  ([, r]) => bestPicks[r.executionId]
                                )?.[1]?.executionId
                              }
                              onPick={handlePickWinner}
                            />
                            {config?.judge_chat_provider_id && (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 gap-1 px-2 text-xs"
                                disabled={isJudging}
                                onClick={() =>
                                  handleTriggerJudge(
                                    completed.map(([, r]) => r.executionId)
                                  )
                                }
                              >
                                {isJudging ? (
                                  <Loader2 className="h-3 w-3 animate-spin" />
                                ) : (
                                  <Sparkles className="h-3 w-3" />
                                )}
                                Auto-Judge
                              </Button>
                            )}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                ))}
                {isAnyStreaming && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
            <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isAnyStreaming} />
          </>
        )}
      </div>
    </AppShell>
  );
}
