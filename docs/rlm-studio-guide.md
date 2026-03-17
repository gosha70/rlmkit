# RLM Studio Guide

RLM Studio is a web application for experimenting with, tuning, and monitoring RLMKit. It lets you compare how different LLM providers and execution modes (Direct, RLM, RAG) handle the same queries — side by side, with full cost and performance metrics.

## Prerequisites

- Python 3.10+ with `uv`
- Node.js 22+
- At least one LLM provider API key (OpenAI, Anthropic) or a local model (Ollama, LM Studio)

## Starting the Application

```bash
# Terminal 1: Backend API
uv run uvicorn src.rlmkit.server.app:app --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

The frontend runs on `http://localhost:3001`. The backend runs on `http://localhost:8000`. The frontend proxies API calls to the backend automatically (configured in `next.config.ts`).

## Pages Overview

RLM Studio has four main pages, accessible from the sidebar:

| Page | Purpose |
|------|---------|
| **Chat** | Send queries to one or more Chat Providers in parallel |
| **Dashboard** | View aggregated metrics and charts per session |
| **Traces** | Inspect individual execution traces step by step |
| **Settings** | Configure providers, Chat Providers, budgets, profiles, prompts, and theme |

---

## Settings

Settings is where you configure everything before running experiments. It has six tabs.

### Providers

Providers are the raw LLM connections. RLM Studio supports OpenAI, Anthropic, Ollama (local), and LM Studio (local) out of the box. For each provider you can:

- **Select a model** from the provider's model catalog
- **Enter an API key** (or rely on environment variables like `OPENAI_API_KEY`)
- **Set a custom endpoint** (for local models or proxies)
- **Configure runtime settings** — temperature, top_p, max output tokens, timeout
- **Enable/disable** the provider for chat
- **Test the connection** to verify the key and endpoint work

Providers that detect an environment variable show "API key set" automatically. You only need to configure them manually if you want to override the key or change the default model.

### Chat Providers and Profiles

A **Chat Provider** binds a specific LLM (provider + model) to a **Profile**. The Profile controls execution mode, runtime settings, and budget limits. Editing a Profile immediately affects all Chat Providers that reference it — no caching, no duplication.

| | Chat Provider | Profile |
|---|---|---|
| **Purpose** | A runnable configuration you select on the Chat page | A reusable settings template referenced by Chat Providers |
| **Bound to an LLM?** | Yes — specific provider + model (e.g., Anthropic / claude-sonnet-4-5) | No — provider-agnostic |
| **Used in Chat?** | Yes — select one or more, each executes independently | Indirectly — through the Chat Providers that reference it |
| **Controls** | LLM selection, RAG-specific config | Execution mode, runtime settings (temp, top_p, max tokens, timeout), budget limits |
| **Editable fields** | Name, LLM provider, model, profile, RAG config | Strategy, runtime settings, budget, system prompts, description |

**In practice:** Pick or create a Profile (e.g., "RLM deep" — temperature 0.4, 32 steps, 4096 max tokens). Then create Chat Providers that pair that Profile with specific LLMs. Change the Profile's temperature and every Chat Provider referencing it picks up the new value immediately.

### Chat Providers

A **Chat Provider** is a named, runnable configuration that pairs a specific LLM with a Profile. You select Chat Providers on the Chat page to execute queries.

Each Chat Provider specifies:

- **LLM Provider + Model** — e.g., Anthropic / claude-sonnet-4-5
- **Profile** — controls execution mode (Direct / RLM / RAG), runtime settings, and budget limits
- **RAG config** (optional) — chunk size, overlap, top_k, embedding model (only shown when the profile's strategy is RAG)

Create multiple Chat Providers with different LLM + Profile combinations, then select them on the Chat page to compare results side by side.

**Example setup for comparing modes:**

| Chat Provider Name | Provider | Profile | Notes |
|-------------------|----------|---------|-------|
| GPT-4o Direct | OpenAI / gpt-4o | Accurate | Baseline — full context, direct mode |
| GPT-4o RLM | OpenAI / gpt-4o | RLM deep | Same model, recursive exploration |
| Claude Sonnet Direct | Anthropic / claude-sonnet-4-5 | Accurate | Cross-provider comparison |
| Claude Sonnet RLM | Anthropic / claude-sonnet-4-5 | RLM deep | Cross-provider + cross-mode |

**Example setup for tuning RLM parameters:** Create custom profiles with different step budgets, then assign them to Chat Providers using the same LLM.

| Chat Provider Name | Profile | Steps | Temperature | Notes |
|-------------------|---------|-------|-------------|-------|
| RLM Conservative | Custom: RLM-8 | 8 | 0.3 | Fewer steps, lower randomness |
| RLM Balanced | RLM deep (built-in) | 32 | 0.4 | Built-in default |
| RLM Aggressive | Custom: RLM-64 | 64 | 0.7 | More exploration budget |

### Budget

Global budget limits that apply to all executions:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| Max Steps | 16 | Maximum RLM loop iterations |
| Max Tokens | 50,000 | Total token limit across all steps |
| Max Cost (USD) | $2.00 | Hard cost cap per execution |
| Max Time (seconds) | 30 | Wall-clock timeout |
| Max Recursion Depth | 5 | Nested subcall depth limit |

Use budget controls to prevent runaway costs during experimentation. Start with conservative limits and increase as you understand your workload's requirements.

### Profiles

Profiles are saved presets of global defaults — runtime settings, budget limits, and system prompts bundled into a reusable template. Profiles are **not bound to any LLM provider**. They set the baseline that Chat Providers and the programmatic API inherit.

Each profile includes:

- **Name** and **description**
- **Strategy** (the default execution mode: Direct, RLM, etc.)
- **Runtime settings** — temperature, top_p, max output tokens, timeout
- **Budget limits** — max steps, tokens, cost, time, recursion depth
- **System prompts** — per-mode prompt overrides

RLM Studio ships with four **built-in profiles** (marked with a lock icon, cannot be deleted):

| Profile | Mode | Temp | Max Tokens | Steps | Use Case |
|---------|------|------|------------|-------|----------|
| **Fast & cheap** | Direct | 0.5 | 1,000 | 8 | Quick, low-cost responses with conservative token limits |
| **Accurate** | Direct | 0.2 | 4,096 | 16 | High-quality, precise answers with low temperature |
| **RLM deep** | RLM | 0.4 | 4,096 | 32 | Deep recursive reasoning with high step budget |
| **RAG retrieval** | RAG | 0.3 | 4,096 | 8 | Retrieval-augmented generation with moderate token budget |

**Actions on profiles:**

- **Activate** (play button) — apply the profile's settings as the new global defaults
- **Clone** (copy button) — create an editable copy of any profile (including built-ins)
- **Edit** (pencil button) — modify strategy, runtime settings, and budget inline (custom profiles only)
- **Delete** (trash button) — remove a custom profile. Profiles referenced by Chat Providers cannot be deleted — reassign the Chat Providers first.

### Prompts

Customize the system prompts used for each execution mode:

- **Direct** — the system prompt sent with full-context queries
- **RLM** — the system prompt that instructs the LLM how to use the sandbox tools (`peek()`, `grep()`, `chunk()`, `select()`)
- **RAG** — the system prompt for retrieval-augmented queries

You can also apply **prompt templates** — predefined prompt sets that you can load and customize.

### Appearance

Switch between **Light**, **Dark**, and **System** themes.

---

## Chat

The Chat page is where you run experiments.

### Workflow

1. **Select Chat Providers** — Use the toolbar dropdown to select one or more Chat Providers. Each selected provider will receive the same query in parallel.

2. **Upload a document** (optional) — Click the upload button to attach a PDF, DOCX, TXT, or other supported file. The file is processed server-side and its content becomes available to the LLM.

3. **Ask a question** — Type your query and press Enter (or click Send).

4. **Compare responses** — Each Chat Provider's response appears in its own column. Below each response you see per-execution metrics:
   - **Tokens** — total tokens consumed
   - **Cost** — USD cost for this execution
   - **Latency** — wall-clock time

5. **Continue the conversation** — Conversation history is maintained per Chat Provider. Each follow-up question includes the relevant conversation history for that provider, enabling multi-turn dialogues.

### Side-by-Side Comparison

The column layout scales with the number of selected Chat Providers. For example, selecting three Chat Providers produces a three-column grid where you can visually compare:

- Answer quality and completeness
- Token efficiency (RLM often uses fewer tokens for large content)
- Cost differences between providers and modes
- Latency trade-offs (Direct is faster per call; RLM may be faster overall for large content)

### Sessions

Sessions persist across page reloads. The sidebar lists all sessions. You can:

- Click a session to reload its conversation history
- Create a new session (+ button)
- Delete sessions you no longer need

---

## Dashboard

The Dashboard provides aggregated metrics for a selected session.

### Summary Cards

- **Total Tokens** — sum of all tokens used in the session
- **Total Cost** — sum of all execution costs
- **Avg Latency** — mean execution time
- **Token Savings** — percentage saved by RLM compared to Direct mode

### Charts

- **Comparison Chart** — RLM vs Direct mode metrics side by side
- **Cost by Provider** — pie chart showing cost distribution across LLM providers
- **Provider Performance** — bar chart comparing tokens, latency, and cost per provider
- **Chat Provider Performance** — same breakdown but grouped by Chat Provider (so you can distinguish two configurations using the same LLM)
- **Cost by Chat Provider** — pie chart by Chat Provider
- **Performance Trend** — timeline of tokens, cost, and latency across executions

### Recent Executions Table

A table listing every execution in the session with mode, provider, tokens, cost, and latency.

---

## Traces

The Traces page lets you inspect individual executions in detail.

### Execution List

A table of all recent executions showing:

- **Query** — the user's question
- **Chat Provider** — which Chat Provider was used
- **Mode** — Direct, RLM, or RAG
- **Status** — complete, running, or error
- **Tokens** — total tokens consumed
- **Cost** — USD cost

Click any row to load its full trace.

### Trace Detail

When you select an execution, the trace detail shows:

**Summary bar** — query text, mode, Chat Provider name, steps used/limit, tokens, cost, and status.

**Three views** (tabs):

1. **Timeline** — chronological list of every step the LLM took. Each step shows the action type, token count, cost, and duration.

2. **Tree** — hierarchical view showing recursion depth. Subcalls appear nested under their parent steps.

3. **Code** — shows only the code-generation steps. For each step you see:
   - The Python code the LLM wrote
   - The output that code produced
   - Token and cost metrics

**Step Detail** — click any step in Timeline or Tree to see its full details: the raw code, output, model used, duration, and token breakdown.

---

## Typical Experiment Workflows

### Comparing RLM vs Direct for a Large Document

1. Go to **Settings > Providers** and configure at least one LLM provider (e.g., Anthropic)
2. Go to **Settings > Chat Providers** and create two:
   - "Claude Direct" — Anthropic / claude-sonnet-4-5 / Profile: Accurate
   - "Claude RLM" — Anthropic / claude-sonnet-4-5 / Profile: RLM deep
3. Go to **Chat**, select both Chat Providers
4. Upload a large document
5. Ask a question — both providers respond in parallel
6. Compare: answer quality, tokens used, cost, latency
7. Go to **Dashboard** to see aggregated metrics
8. Go to **Traces** to inspect the RLM execution steps

### Tuning RLM Step Budget

1. Go to **Settings > Profiles** and create three custom profiles with strategy "RLM" and different step budgets: 8, 16, 32
2. Create three Chat Providers with the same LLM, each referencing a different profile
3. Run the same query against all three
4. Check **Traces** for each — does 32 steps find better answers than 8?
5. Check **Dashboard** — what's the cost/quality trade-off?

### Cross-Provider Benchmarking

1. Create Chat Providers for OpenAI/gpt-4o and Anthropic/claude-sonnet-4-5, both using the "Accurate" profile
2. Run identical queries against both
3. Compare response quality, token usage, and cost in the column layout
4. Use **Dashboard > Cost by Chat Provider** to see cumulative cost differences

### Finding the Right Temperature

1. Go to **Settings > Profiles** and create three custom profiles with the same strategy but temperatures 0.2, 0.5, and 0.9
2. Create Chat Providers with the same LLM, each referencing a different profile
3. Run the same factual question against all three
4. Lower temperatures produce more consistent, deterministic answers
5. Higher temperatures produce more creative but potentially less accurate answers

---

## Environment Configuration

### Backend Port

The backend defaults to port 8000. To change it:

```bash
uv run uvicorn src.rlmkit.server.app:app --reload --port 8002
```

Then set the frontend to point at the new port. Create or edit `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8002
```

Restart the frontend dev server for the change to take effect.

### API Keys

Set provider API keys as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Or enter them in **Settings > Providers** in the UI (saved to `.env`).

### Docker / Playwright MCP

When accessing RLM Studio from inside a Docker container (e.g., Playwright MCP for UI testing), use `host.docker.internal` instead of `localhost`:

```
http://host.docker.internal:3001
```

The Next.js frontend proxies all API calls through its own server, so the Docker browser never needs direct access to the backend port.
