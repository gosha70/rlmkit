"use client";

import { File, X, Hash } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FileAttachmentProps {
  name: string;
  sizeBytes: number;
  tokenCount: number;
  onRemove: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function FileAttachment({ name, sizeBytes, tokenCount, onRemove }: FileAttachmentProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-muted/50 px-4 py-2">
      <File className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
      <div className="flex flex-1 flex-col">
        <span className="text-sm font-medium">{name}</span>
        <span className="text-xs text-muted-foreground">
          {formatBytes(sizeBytes)}
          {tokenCount > 0 && (
            <span className="ml-2 inline-flex items-center gap-0.5">
              <Hash className="h-3 w-3" aria-hidden="true" />
              ~{tokenCount.toLocaleString()} tokens
            </span>
          )}
        </span>
      </div>
      <Button variant="ghost" size="icon" onClick={onRemove} aria-label={`Remove file ${name}`} className="h-6 w-6">
        <X className="h-3 w-3" aria-hidden="true" />
      </Button>
    </div>
  );
}
