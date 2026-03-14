"use client";

interface JudgeScoresProps {
  dimensions: Record<string, number>;
  overallScore: number;
}

const DIMENSION_LABELS: Record<string, string> = {
  relevance: "Rel",
  correctness: "Cor",
  completeness: "Com",
  coherence: "Coh",
  conciseness: "Con",
};

export function JudgeScores({ dimensions, overallScore }: JudgeScoresProps) {
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      {Object.entries(dimensions).map(([key, value]) => (
        <div key={key} className="flex items-center gap-1">
          <span className="font-medium">{DIMENSION_LABELS[key] || key}:</span>
          <div className="relative h-1.5 w-8 rounded-full bg-muted">
            <div
              className="absolute h-full rounded-full bg-violet-500"
              style={{ width: `${(value / 5) * 100}%` }}
            />
          </div>
          <span>{value}</span>
        </div>
      ))}
      <span className="ml-1 font-semibold text-foreground">
        {overallScore.toFixed(1)}/5
      </span>
    </div>
  );
}
