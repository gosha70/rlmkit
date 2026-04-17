# Ollama

Local LLM runtime with a simple CLI. Runs on Linux and macOS.

- **Best for:** quick local start, quantized models, no API key.
- **You'll need:** Homebrew (macOS) or `curl` (Linux); ~5 GB disk
  for a starter model like `llama3.1:8b`; 16 GB RAM is comfortable.
- **Known-good config:** base URL `http://localhost:11434`, model
  `llama3.1:8b`.
- **Most common failure:** "connection refused" — `ollama serve`
  isn't running. See §6.

> **Windows:** Windows users can usually follow the WSL path below.
> Full Windows-native guidance is out of scope for V1.

## 1. Install

**macOS (Homebrew):**

```bash
brew install ollama
```

**Linux (curl):**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## 2. Start the server

```bash
ollama serve
```

The server listens on `http://localhost:11434` by default.

## 3. Pull a model

```bash
ollama pull llama3.1:8b
```

Browse available models at <https://ollama.com/library>. For first
experiments, `llama3.1:8b` is a good default: 4–5 GB, runs on 16 GB RAM.

## 4. Add to RLM Studio

In **Settings → LLM Providers → New**:

| Field    | Value                    |
|----------|--------------------------|
| Backend  | `ollama`                 |
| Model    | `llama3.1:8b`            |
| Base URL | `http://localhost:11434` |

No API key is required.

## 5. Test connection

Click **Test Connection** in Settings. A green check appears on success.

## 6. Common errors

- **Connection refused** — `ollama serve` is not running. Start it, then
  retry.
- **Model not found** — run `ollama pull <model-name>` first.
- **Out of memory** — the model is too large for available RAM. Try a
  smaller variant (e.g. `llama3.1:8b` instead of `70b`).
