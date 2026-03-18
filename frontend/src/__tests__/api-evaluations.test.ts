/**
 * Tests for evaluation API functions in lib/api.ts.
 *
 * Covers: submitThumbRating, removeThumbRating, submitBestPick,
 * getSessionEvaluations, getEvaluationSummary, triggerJudge.
 */

import { vi, describe, test, expect, beforeEach, afterAll } from "vitest";
import {
  submitThumbRating,
  removeThumbRating,
  submitBestPick,
  getSessionEvaluations,
  getEvaluationSummary,
  triggerJudge,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Mock fetch
// ---------------------------------------------------------------------------

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function mockJsonResponse(body: unknown, status = 200) {
  mockFetch.mockResolvedValueOnce({
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  });
}

function mockErrorResponse(status: number, body = "error") {
  mockFetch.mockResolvedValueOnce({
    ok: false,
    status,
    text: () => Promise.resolve(body),
  });
}

beforeEach(() => {
  mockFetch.mockReset();
});

afterAll(() => {
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// submitThumbRating
// ---------------------------------------------------------------------------

describe("submitThumbRating", () => {
  const req = {
    execution_id: "exec-1",
    session_id: "sess-1",
    chat_provider_id: "cp-1",
    rating: "up" as const,
  };

  test("sends POST to /api/evaluations/thumb with JSON body", async () => {
    mockJsonResponse({ status: "ok" });
    await submitThumbRating(req);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/thumb",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(req),
      })
    );
  });

  test("returns parsed response", async () => {
    mockJsonResponse({ status: "created" });
    const result = await submitThumbRating(req);
    expect(result).toEqual({ status: "created" });
  });

  test("throws on non-OK response", async () => {
    mockErrorResponse(400, "bad request");
    await expect(submitThumbRating(req)).rejects.toThrow("API error 400");
  });
});

// ---------------------------------------------------------------------------
// removeThumbRating
// ---------------------------------------------------------------------------

describe("removeThumbRating", () => {
  test("sends DELETE to /api/evaluations/thumb/:executionId", async () => {
    mockJsonResponse({ status: "deleted" });
    await removeThumbRating("exec-42");
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/thumb/exec-42",
      expect.objectContaining({ method: "DELETE" })
    );
  });

  test("returns parsed response", async () => {
    mockJsonResponse({ status: "deleted" });
    const result = await removeThumbRating("exec-42");
    expect(result).toEqual({ status: "deleted" });
  });

  test("throws on 404", async () => {
    mockErrorResponse(404, "not found");
    await expect(removeThumbRating("exec-99")).rejects.toThrow("API error 404");
  });
});

// ---------------------------------------------------------------------------
// submitBestPick
// ---------------------------------------------------------------------------

describe("submitBestPick", () => {
  const req = { session_id: "sess-1", winner_execution_id: "exec-2" };

  test("sends POST to /api/evaluations/best-pick", async () => {
    mockJsonResponse({ status: "ok" });
    await submitBestPick(req);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/best-pick",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(req),
      })
    );
  });

  test("returns parsed response", async () => {
    mockJsonResponse({ status: "created" });
    const result = await submitBestPick(req);
    expect(result).toEqual({ status: "created" });
  });

  test("throws on server error", async () => {
    mockErrorResponse(500, "internal error");
    await expect(submitBestPick(req)).rejects.toThrow("API error 500");
  });
});

// ---------------------------------------------------------------------------
// getSessionEvaluations
// ---------------------------------------------------------------------------

describe("getSessionEvaluations", () => {
  const mockEvals = {
    thumb_ratings: [{ id: "t-1", execution_id: "exec-1", rating: "up" }],
    best_picks: [],
    judge_scores: [],
    judge_pairwise: [],
  };

  test("sends GET to /api/evaluations/:sessionId", async () => {
    mockJsonResponse(mockEvals);
    await getSessionEvaluations("sess-5");
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/sess-5",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      })
    );
  });

  test("returns evaluation data", async () => {
    mockJsonResponse(mockEvals);
    const result = await getSessionEvaluations("sess-5");
    expect(result.thumb_ratings).toHaveLength(1);
    expect(result.thumb_ratings[0].rating).toBe("up");
  });

  test("throws on non-OK response", async () => {
    mockErrorResponse(403, "forbidden");
    await expect(getSessionEvaluations("sess-5")).rejects.toThrow("API error 403");
  });
});

// ---------------------------------------------------------------------------
// getEvaluationSummary
// ---------------------------------------------------------------------------

describe("getEvaluationSummary", () => {
  const mockSummary = {
    session_id: "sess-5",
    by_chat_provider: {
      "cp-1": { thumb_score: 1.0, best_picks: 3, pick_rate: 0.75, judge_avg_score: 4.2, combined_score: 0.85 },
    },
    recommendation: "cp-1",
    recommendation_reason: "Highest combined score",
  };

  test("sends GET to /api/evaluations/:sessionId/summary", async () => {
    mockJsonResponse(mockSummary);
    await getEvaluationSummary("sess-5");
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/sess-5/summary",
      expect.objectContaining({
        headers: { "Content-Type": "application/json" },
      })
    );
  });

  test("returns summary with provider scores and recommendation", async () => {
    mockJsonResponse(mockSummary);
    const result = await getEvaluationSummary("sess-5");
    expect(result.recommendation).toBe("cp-1");
    expect(result.by_chat_provider["cp-1"].combined_score).toBe(0.85);
  });

  test("throws on non-OK response", async () => {
    mockErrorResponse(404, "not found");
    await expect(getEvaluationSummary("sess-5")).rejects.toThrow("API error 404");
  });
});

// ---------------------------------------------------------------------------
// triggerJudge
// ---------------------------------------------------------------------------

describe("triggerJudge", () => {
  const req = {
    session_id: "sess-1",
    execution_ids: ["exec-1", "exec-2"],
    mode: "both" as const,
  };

  const mockResult = {
    pointwise: [
      { execution_id: "exec-1", dimensions: { relevance: 4.0 }, overall_score: 4.0, reasoning: "good" },
    ],
    pairwise: [],
  };

  test("sends POST to /api/evaluations/judge", async () => {
    mockJsonResponse(mockResult);
    await triggerJudge(req);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/judge",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify(req),
      })
    );
  });

  test("returns pointwise and pairwise results", async () => {
    mockJsonResponse(mockResult);
    const result = await triggerJudge(req);
    expect(result.pointwise).toHaveLength(1);
    expect(result.pointwise[0].overall_score).toBe(4.0);
    expect(result.pairwise).toHaveLength(0);
  });

  test("works without optional mode parameter", async () => {
    const minReq = { session_id: "sess-1", execution_ids: ["exec-1"] };
    mockJsonResponse(mockResult);
    await triggerJudge(minReq);
    expect(mockFetch).toHaveBeenCalledWith(
      "/api/evaluations/judge",
      expect.objectContaining({
        body: JSON.stringify(minReq),
      })
    );
  });

  test("throws on server error", async () => {
    mockErrorResponse(500, "judge failed");
    await expect(triggerJudge(req)).rejects.toThrow("API error 500");
  });
});
