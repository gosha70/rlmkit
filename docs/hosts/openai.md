# OpenAI

Cloud-hosted inference via the OpenAI API.

- **Best for:** broad model selection (GPT-4o family), pay-per-use
  cloud; the fastest path when you don't want to run a server.
- **You'll need:** an OpenAI account with an API key and a
  billing method on file.
- **Known-good config:** leave base URL blank (the OpenAI default
  is used); model `gpt-4o-mini` for cheap experimentation,
  `gpt-4o` for stronger results.
- **Most common failure:** 401 Unauthorized — key is wrong,
  revoked, or the account has no billing. See §5.

## 1. Get an API key

1. Sign in at <https://platform.openai.com>.
2. Open **API keys** (sidebar) → **Create new secret key**.
3. Copy the `sk-…` key — it is only shown once.

## 2. Add billing

OpenAI requires a billing source before API calls succeed. Add one at
**Billing → Payment methods**.

## 3. Add to RLM Studio

In **Settings → LLM Providers → New**:

| Field    | Value                       |
|----------|-----------------------------|
| Backend  | `openai`                    |
| Model    | e.g. `gpt-4o`, `gpt-4o-mini`|
| API key  | `sk-…`                      |

Base URL is optional — RLM Studio uses the OpenAI default
(`https://api.openai.com/v1`) if you leave it blank.

## 4. Test connection

Click **Test Connection** in Settings.

## 5. Common errors

- **401 Unauthorized** — the key is wrong, revoked, or missing billing
  on the account.
- **429 Too Many Requests** — you've hit a tier rate limit. Wait, or
  upgrade the organization's tier.
- **404 model_not_found** — the model id is wrong, or your org does
  not have access yet (new models are often gated).
