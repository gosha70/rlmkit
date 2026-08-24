/**
 * Proxy for POST /api/evaluations/judge with extended timeout.
 *
 * Why this exists:
 *   Next.js rewrites (next.config.ts) time out after 30 s by default.
 *   The judge service makes 4-6 LLM calls (pointwise × providers +
 *   pairwise × 2 for position-swap debiasing) which routinely take 60-120 s
 *   with local models.  A route handler lets us set an explicit timeout.
 *
 * Timeout configuration:
 *   Set JUDGE_TIMEOUT_SECONDS in frontend/.env.local to control how long
 *   the proxy waits for the backend before aborting.  Default: 180 s.
 *
 * Backend address:
 *   This handler runs on the *server*, so it needs the address the Next
 *   process can reach — not the one the browser uses. Under Docker Compose
 *   those differ (backend:8000 vs localhost:8000). INTERNAL_API_URL carries
 *   the server-side address and is read at runtime; NEXT_PUBLIC_* cannot be
 *   used for this because Next inlines it at build time, which is what made
 *   the composed frontend proxy to itself (ECONNREFUSED 127.0.0.1:8000).
 */

import { NextRequest, NextResponse } from "next/server";

const JUDGE_TIMEOUT_MS =
  parseInt(process.env.JUDGE_TIMEOUT_SECONDS ?? "180", 10) * 1_000;

export async function POST(req: NextRequest): Promise<NextResponse> {
  const backend =
    process.env.INTERNAL_API_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000";
  const body = await req.text();

  try {
    const upstream = await fetch(`${backend}/api/evaluations/judge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: AbortSignal.timeout(JUDGE_TIMEOUT_MS),
    });

    const data = await upstream.text();
    return new NextResponse(data, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ detail: `Judge proxy error: ${message}` }, { status: 502 });
  }
}
