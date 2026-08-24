/**
 * Regression tests for the server-side backend address used by the judge
 * route handler (`src/app/api/evaluations/judge/route.ts`).
 *
 * Why this exists: the handler runs on the *server*, which under Docker
 * Compose reaches the backend at a different address than the browser does
 * (`http://backend:8000` vs `http://localhost:8000`). It previously read
 * `NEXT_PUBLIC_API_URL`, which Next inlines at build time — so the compose
 * runtime override was silently ignored and the container proxied to itself
 * (`ECONNREFUSED 127.0.0.1:8000`). `INTERNAL_API_URL` is a server-only var
 * read at runtime and must take precedence.
 */

import { describe, test, expect, vi, beforeEach, afterEach } from "vitest";

const ORIGINAL_ENV = { ...process.env };

/** Import the route fresh so module-level env reads are re-evaluated. */
async function loadRoute() {
  vi.resetModules();
  return import("@/app/api/evaluations/judge/route");
}

/** Capture the URL the handler fetches, without hitting the network. */
function stubFetch(): { calls: string[] } {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string | URL) => {
      calls.push(String(url));
      return new Response(JSON.stringify({ pointwise: [], pairwise: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
  return { calls };
}

function request(): Request {
  return new Request("http://localhost:3000/api/evaluations/judge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: "s1" }),
  });
}

describe("judge route — server-side backend address", () => {
  beforeEach(() => {
    process.env = { ...ORIGINAL_ENV };
    delete process.env.INTERNAL_API_URL;
    delete process.env.NEXT_PUBLIC_API_URL;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    process.env = { ...ORIGINAL_ENV };
  });

  test("INTERNAL_API_URL wins over NEXT_PUBLIC_API_URL (the compose case)", async () => {
    process.env.INTERNAL_API_URL = "http://backend:8000";
    process.env.NEXT_PUBLIC_API_URL = "http://localhost:8000";
    const { calls } = stubFetch();
    const { POST } = await loadRoute();

    await POST(request() as never);

    expect(calls).toHaveLength(1);
    expect(calls[0]).toBe("http://backend:8000/api/evaluations/judge");
    expect(calls[0]).not.toContain("localhost");
  });

  test("falls back to NEXT_PUBLIC_API_URL when no internal address is set", async () => {
    process.env.NEXT_PUBLIC_API_URL = "http://example.test:9000";
    const { calls } = stubFetch();
    const { POST } = await loadRoute();

    await POST(request() as never);

    expect(calls[0]).toBe("http://example.test:9000/api/evaluations/judge");
  });

  test("defaults to localhost:8000 when neither is set (plain dev)", async () => {
    const { calls } = stubFetch();
    const { POST } = await loadRoute();

    await POST(request() as never);

    expect(calls[0]).toBe("http://localhost:8000/api/evaluations/judge");
  });

  test("an unreachable backend surfaces as a 502, not a crash", async () => {
    process.env.INTERNAL_API_URL = "http://backend:8000";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("connect ECONNREFUSED 127.0.0.1:8000");
      }),
    );
    const { POST } = await loadRoute();

    const resp = await POST(request() as never);
    expect(resp.status).toBe(502);
    const body = await resp.json();
    expect(body.detail).toContain("Judge proxy error");
  });
});
