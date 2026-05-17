import type { NextConfig } from "next";

// Two build modes:
//
// * Dev (`next dev`, `next build`): the Next server runs and proxies
//   `/api/*`, `/health`, `/ws/*` to the backend on :8000 via rewrites.
//   This is the path used by `npm run dev`, frontend CI, and any
//   developer who wants the UI on a separate port.
//
// * Bundle (`npm run build:bundle`, BUNDLE=1): produces a fully static
//   export under `frontend/out/`, intended to be copied into
//   `src/rlmkit/_ui/` and served by FastAPI at `/studio` in the same
//   process as the API. No dev server, no rewrites — every request
//   is same-origin, so the rewrites are unnecessary.
//
// `basePath: "/studio"` makes Next emit asset URLs and client-side
// router paths under `/studio`, matching the FastAPI mount.
// `trailingSlash: true` produces one `index.html` per route directory,
// which is what FastAPI's `StaticFiles(html=True)` resolves cleanly.
const isBundle = process.env.BUNDLE === "1";

const nextConfig: NextConfig = isBundle
  ? {
      output: "export",
      basePath: "/studio",
      trailingSlash: true,
    }
  : {
      async rewrites() {
        const backend = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
        return [
          { source: "/api/:path*", destination: `${backend}/api/:path*` },
          { source: "/health", destination: `${backend}/health` },
          { source: "/ws/:path*", destination: `${backend}/ws/:path*` },
        ];
      },
    };

export default nextConfig;
