"use client";

import { useRef, useState } from "react";
import { Send, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (text: string) => void;
  onFileUpload?: (file: File) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, onFileUpload, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    // Auto-resize
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && onFileUpload) onFileUpload(file);
  };

  return (
    <div
      className="flex items-end gap-2 border-t bg-card p-4"
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
    >
      {onFileUpload && (
        <>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            aria-label="Upload file"
          >
            <Upload className="h-4 w-4" />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.docx,.txt,.md,.py,.json,.csv"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file && onFileUpload) onFileUpload(file);
              e.target.value = "";
            }}
          />
        </>
      )}

      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleInput}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about your document..."
        disabled={disabled}
        rows={1}
        className={cn(
          "flex-1 resize-none rounded-lg border bg-background px-4 py-3 text-sm",
          "placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring",
          "disabled:cursor-not-allowed disabled:opacity-50",
        )}
      />

      <Button
        onClick={handleSend}
        disabled={disabled || !value.trim()}
        size="icon"
        aria-label="Send message"
      >
        <Send className="h-4 w-4" />
      </Button>
    </div>
  );
}
