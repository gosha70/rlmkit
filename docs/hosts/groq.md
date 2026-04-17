# Groq

Fast cloud inference on LPU hardware. OpenAI-compatible API.

## 1. Get an API key

1. Sign in at <https://console.groq.com>.
2. Open **API Keys** → **Create API Key**.
3. Copy the `gsk_…` key — it is only shown once.

## 2. Add to RLM Studio

In **Settings → LLM Providers → New**:

| Field    | Value                                  |
|----------|----------------------------------------|
| Backend  | `groq`                                 |
| Model    | e.g. `llama-3.1-8b-instant`, `llama-3.1-70b-versatile` |
| API key  | `gsk_…`                                |

Base URL defaults to `https://api.groq.com/openai/v1` — leave blank
to use it.

## 3. Test connection

Click **Test Connection** in Settings.

## 4. Common errors

- **401 Invalid API key** — the key is wrong or from a different
  Groq organization.
- **429 rate_limit_exceeded** — Groq enforces per-tier rate limits.
  Wait or upgrade the account.
- **400 model_not_found** — the model id is wrong or deprecated.
  Check <https://console.groq.com/docs/models> for the current list.
