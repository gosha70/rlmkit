import { cn } from "@/lib/utils";

interface CodeBlockProps {
  code: string;
  language?: string;
  output?: string | null;
  className?: string;
}

export function CodeBlock({ code, language = "python", output, className }: CodeBlockProps) {
  return (
    <div className={cn("rounded-md border overflow-hidden", className)}>
      <div className="flex items-center justify-between bg-muted px-3 py-1.5">
        <span className="text-xs text-muted-foreground">{language}</span>
      </div>
      <pre className="overflow-x-auto p-3 text-sm">
        <code className="font-mono">{code}</code>
      </pre>
      {output != null && (
        <>
          <div className="border-t bg-muted px-3 py-1.5">
            <span className="text-xs text-muted-foreground">Output</span>
          </div>
          <pre className="overflow-x-auto p-3 text-sm bg-muted/30">
            <code className="font-mono text-muted-foreground">{output}</code>
          </pre>
        </>
      )}
    </div>
  );
}
