"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { AppShell } from "@/components/shared/app-shell";
import { ModeSelector } from "@/components/chat/mode-selector";
import { ProviderBadge } from "@/components/chat/provider-badge";
import { MessageBubble } from "@/components/chat/message-bubble";
import { ChatInput } from "@/components/chat/chat-input";
import { FileAttachment } from "@/components/chat/file-attachment";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat, type ChatMessage } from "@/lib/use-chat";
import {
  getSession,
  getProviders,
  submitChat,
  uploadFile,
  type ChatMode,
  type ProviderInfo,
  type FileUploadResponse,
} from "@/lib/api";

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [mode, setMode] = useState<ChatMode>("auto");
  const [uploadedFile, setUploadedFile] = useState<FileUploadResponse | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const { messages, isConnected, isStreaming, sendQuery, setMessages } = useChat(sessionId);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { data: providers } = useSWR<ProviderInfo[]>("providers", getProviders);
  const activeProvider = providers?.find((p) => p.status === "connected");

  // Load session messages when sessionId changes
  useEffect(() => {
    if (!sessionId) return;
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
        console.error("Failed to load session:", err);
      });
  }, [sessionId, setMessages]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(
    async (text: string) => {
      if (isConnected && sessionId) {
        sendQuery(text, fileContent || "", mode);
      } else {
        try {
          const resp = await submitChat({
            query: text,
            content: fileContent || null,
            file_id: uploadedFile?.id || null,
            mode,
            session_id: sessionId,
          });
          if (!sessionId) setSessionId(resp.session_id);

          setMessages((prev: ChatMessage[]) => [
            ...prev,
            { id: crypto.randomUUID(), role: "user", content: text },
            {
              id: resp.execution_id,
              role: "assistant",
              content: "Processing...",
              mode,
              isStreaming: true,
            },
          ]);
        } catch (err) {
          console.error("Failed to submit chat:", err);
        }
      }
    },
    [isConnected, sessionId, fileContent, uploadedFile, mode, sendQuery, setMessages],
  );

  const handleFileUpload = useCallback(async (file: File) => {
    try {
      const resp = await uploadFile(file);
      setUploadedFile(resp);
      const text = await file.text();
      setFileContent(text);
    } catch (err) {
      console.error("Failed to upload file:", err);
    }
  }, []);

  const handleNewSession = useCallback(() => {
    setSessionId(null);
    setMessages([]);
    setUploadedFile(null);
    setFileContent("");
  }, [setMessages]);

  return (
    <AppShell
      activeSessionId={sessionId}
      onSelectSession={setSessionId}
      onNewSession={handleNewSession}
      isConnected={isConnected}
    >
      <div className="flex h-full flex-col">
        {/* Toolbar */}
        <div className="flex items-center gap-3 border-b px-4 py-2">
          <ModeSelector value={mode} onChange={setMode} />
          {activeProvider && (
            <ProviderBadge
              provider={activeProvider.display_name}
              model={activeProvider.default_model}
              status="connected"
            />
          )}
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

        {/* Message thread */}
        <ScrollArea className="flex-1">
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center gap-4 py-20 text-center text-muted-foreground">
                <p className="text-lg font-medium">Welcome to RLM Studio</p>
                <p className="text-sm">
                  Upload a document and ask questions. The RLM engine will recursively
                  explore the content using code generation to find answers efficiently.
                </p>
              </div>
            )}
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isStreaming && <TypingIndicator mode={mode} />}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input area */}
        <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isStreaming} />
      </div>
    </AppShell>
  );
}
