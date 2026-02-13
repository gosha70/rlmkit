"use client";

import { useState, useCallback } from "react";
import useSWR from "swr";
import { getSessions, getHealth, deleteSession as apiDeleteSession, type SessionSummary } from "@/lib/api";
import { Sidebar } from "./sidebar";
import { ThemeToggle } from "./theme-toggle";
import { StatusIndicator } from "./status-indicator";

interface AppShellProps {
  children: React.ReactNode;
  activeSessionId?: string | null;
  onSelectSession?: (id: string) => void;
  onNewSession?: () => void;
}

export function AppShell({
  children,
  activeSessionId = null,
  onSelectSession,
  onNewSession,
}: AppShellProps) {
  const { data: health } = useSWR("health", getHealth, {
    refreshInterval: 30_000,
    errorRetryCount: 2,
  });
  const isConnected = health?.status === "ok";
  const [collapsed, setCollapsed] = useState(false);

  const { data: sessions = [], mutate: mutateSessions } = useSWR<SessionSummary[]>(
    "sessions",
    () => getSessions(),
  );

  const handleDeleteSession = useCallback(
    async (id: string) => {
      await apiDeleteSession(id);
      await mutateSessions();
    },
    [mutateSessions],
  );

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((c) => !c)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewSession={onNewSession}
        onDeleteSession={handleDeleteSession}
        onSelectSession={onSelectSession}
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 shrink-0 items-center justify-between border-b bg-card px-4">
          <h1 className="text-lg font-semibold">RLM Studio</h1>
          <div className="flex items-center gap-3">
            <StatusIndicator status={isConnected ? "connected" : "offline"} />
            <ThemeToggle />
          </div>
        </header>

        <main className="h-0 flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
