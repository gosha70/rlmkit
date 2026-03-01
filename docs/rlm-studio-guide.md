# RLM Studio Guide

RLM Studio is a web application for experimenting with, tuning, and monitoring RLMKit. It lets you compare how different LLM providers and execution modes (Direct, RLM, RAG) handle the same queries — side by side, with full cost and performance metrics.

## Prerequisites

- Python 3.10+ with `uv`
- Node.js 20.9+
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

### Chat Providers

A **Chat Provider** is a named configuration that bundles:

- An **LLM Provider** (e.g., Anthropic with claude-sonnet-4-5)
- An **Execution Mode** (Direct, RLM, or RAG)
- **Runtime settings** (temperature, top_p, max tokens, timeout)
- **Mode-specific settings** (RLM: max steps, timeout; RAG: chunk size, overlap, top_k, embedding model)

Chat Providers are the key abstraction for experiments. You create multiple Chat Providers with different configurations, then select them on the Chat page to compare results side by side.

**Example setup for comparing modes:**

| Chat Provider Name | Provider | Mode | Notes |
|-------------------|----------|------|-------|
| GPT-4o Direct | OpenAI / gpt-4o | Direct | Baseline — full context |
| GPT-4o RLM | OpenAI / gpt-4o | RLM | Same model, recursive exploration |
| Claude Sonnet Direct | Anthropic / claude-sonnet-4-5 | Direct | Cross-provider comparison |
| Claude Sonnet RLM | Anthropic / claude-sonnet-4-5 | RLM | Cross-provider + cross-mode |

**Example setup for tuning RLM parameters:**

| Chat Provider Name | Mode | Max Steps | Temperature | Notes |
|-------------------|------|-----------|-------------|-------|
| RLM Conservative | RLM | 8 | 0.3 | Fewer steps, lower randomness |
| RLM Balanced | RLM | 16 | 0.7 | Default settings |
| RLM Deep | RLM | 32 | 0.4 | More exploration budget |

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

Profiles are saved presets that bundle runtime settings and budget limits into a reusable configuration. Each profile includes:

- **Name** and **description**
- **Strategy** (the execution mode: Direct, RLM, etc.)
- **Runtime settings** — temperature, max output tokens
- **Budget limits** — max steps, tokens, cost, time, recursion depth
- **System prompts** — per-mode prompt overrides

RLM Studio ships with **built-in profiles** (marked with a lock icon):

| Profile | Mode | Temp | Max Tokens | Steps | Use Case |
|---------|------|------|------------|-------|----------|
| **Fast & cheap** | Direct | 0.5 | 1,000 | 8 | Quick, low-cost responses |
| **Accurate** | Direct | 0.2 | 4,096 | 16 | High-quality, precise answers |
| **RLM deep** | RLM | 0.4 | 4,096 | 32 | Deep recursive reasoning for complex problems |

Built-in profiles cannot be deleted. You can create custom profiles and **activate** any profile to apply its settings globally. The active profile's settings become the defaults for new Chat Providers and direct API usage.

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
   - "Claude Direct" — Anthropic / claude-sonnet-4-5 / Direct mode
   - "Claude RLM" — Anthropic / claude-sonnet-4-5 / RLM mode / 16 steps
3. Go to **Chat**, select both Chat Providers
4. Upload a large document
5. Ask a question — both providers respond in parallel
6. Compare: answer quality, tokens used, cost, latency
7. Go to **Dashboard** to see aggregated metrics
8. Go to **Traces** to inspect the RLM execution steps

### Tuning RLM Step Budget

1. Create three Chat Providers with the same LLM but different `max_steps`: 8, 16, 32
2. Run the same query against all three
3. Check **Traces** for each — does 32 steps find better answers than 8?
4. Check **Dashboard** — what's the cost/quality trade-off?

### Cross-Provider Benchmarking

1. Create Chat Providers for OpenAI/gpt-4o (Direct) and Anthropic/claude-sonnet-4-5 (Direct)
2. Run identical queries against both
3. Compare response quality, token usage, and cost in the column layout
4. Use **Dashboard > Cost by Chat Provider** to see cumulative cost differences

### Finding the Right Temperature

1. Create Chat Providers with the same LLM/mode but temperatures 0.2, 0.5, and 0.9
2. Run the same factual question against all three
3. Lower temperatures produce more consistent, deterministic answers
4. Higher temperatures produce more creative but potentially less accurate answers

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
