# Anthropic

Cloud-hosted Claude family models via the Anthropic API.

- **Best for:** strong reasoning and long-context tasks; the
  Claude Sonnet and Opus families.
- **You'll need:** an Anthropic account with an API key and
  prepaid credits.
- **Known-good config:** leave base URL blank; model
  `claude-sonnet-4-6` as a sensible default.
- **Most common failure:** empty response — the request set both
  `temperature` and `top_p`. Use one or the other, never both.
  See §5 and §6.

## 1. Get an API key

1. Sign in at <https://console.anthropic.com>.
2. Open **API keys** → **Create key**.
3. Copy the `sk-ant-…` key — it is only shown once.

## 2. Add credits

Anthropic bills prepaid credits. Add a payment method and purchase a
starter credit amount under **Billing**.

## 3. Add to RLM Studio

In **Settings → LLM Providers → New**:

| Field    | Value                                 |
|----------|---------------------------------------|
| Backend  | `anthropic`                           |
| Model    | e.g. `claude-sonnet-4-6`, `claude-opus-4-6` |
| API key  | `sk-ant-…`                            |

Base URL is optional — the Anthropic default is used if blank.

## 4. Test connection

Click **Test Connection** in Settings.

## 5. Heads-up: temperature + top_p

Anthropic's API rejects requests that set **both** `temperature` and
`top_p`. RLM Studio clears `top_p` automatically when you pick a
profile with a custom temperature; if you hand-build a runtime
settings object, set one or the other — not both. Symptom: the model
returns an empty response with no visible error.

## 6. Common errors

- **Empty response** — you almost certainly sent both `temperature`
  and `top_p`. Remove one.
- **401 Invalid x-api-key** — the key is wrong or revoked.
- **429 overloaded_error** — service is under load; retry with
  exponential backoff.
- **400 model_not_found** — the model id is wrong or deprecated.
  Check <https://docs.anthropic.com/en/docs/about-claude/models> for
  current ids.
