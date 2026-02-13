"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  LayoutDashboard,
  Activity,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Plus,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { SessionSummary } from "@/lib/api";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  sessions?: SessionSummary[];
  activeSessionId?: string | null;
  onNewSession?: () => void;
  onDeleteSession?: (id: string) => void;
  onSelectSession?: (id: string) => void;
}

const NAV_ITEMS = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/traces", label: "Traces", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar({
  collapsed,
  onToggle,
  sessions = [],
  activeSessionId,
  onNewSession,
  onDeleteSession,
  onSelectSession,
}: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r bg-card transition-all duration-200",
        collapsed ? "w-16" : "w-60",
      )}
    >
      <div className="flex h-14 items-center justify-between border-b px-3">
        {!collapsed && <span className="text-sm font-semibold">Navigation</span>}
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="ml-auto"
        >
          {collapsed ? <PanelLeft className="h-4 w-4" aria-hidden="true" /> : <PanelLeftClose className="h-4 w-4" aria-hidden="true" />}
        </Button>
      </div>

      <nav aria-label="Main navigation" className="flex flex-col gap-1 px-2 pt-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground",
                collapsed && "justify-center px-0",
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
              {!collapsed && <span>{label}</span>}
            </Link>
          );
        })}
      </nav>

      {!collapsed && pathname === "/" && (
        <>
          <Separator className="my-3" />
          <div className="flex items-center justify-between px-4">
            <span className="text-xs font-semibold text-muted-foreground">Sessions</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onNewSession}
              aria-label="New session"
            >
              <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
          </div>
          <ScrollArea className="flex-1 px-2 py-1">
            <div className="flex flex-col gap-0.5">
              {sessions.map((s) => (
                <div
                  key={s.id}
                  className={cn(
                    "group flex items-center justify-between rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-accent cursor-pointer",
                    s.id === activeSessionId && "bg-accent text-accent-foreground",
                  )}
                >
                  <button
                    type="button"
                    className="flex-1 truncate text-left"
                    onClick={() => onSelectSession?.(s.id)}
                  >
                    {s.name}
                  </button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession?.(s.id);
                    }}
                    aria-label={`Delete session ${s.name}`}
                  >
                    <Trash2 className="h-3 w-3" aria-hidden="true" />
                  </Button>
                </div>
              ))}
              {sessions.length === 0 && (
                <p className="px-3 py-2 text-xs text-muted-foreground">No sessions yet</p>
              )}
            </div>
          </ScrollArea>
        </>
      )}
    </aside>
  );
}
