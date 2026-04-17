# LM Studio

Cross-platform desktop app with a built-in local inference server.
Works on Windows, macOS, and Linux.

- **Best for:** GUI-driven local inference; no terminal required.
- **You'll need:** LM Studio installed; a downloaded model loaded
  in the Chat tab before the server can serve it.
- **Known-good config:** base URL `http://localhost:1234/v1`, model
  id from the Local Server tab.
- **Most common failure:** "Model not loaded" — you started the
  server without loading a model first. See §6.

## 1. Install

Download the installer for your platform from
<https://lmstudio.ai>. Launch the app after install.

## 2. Load a model

1. Open the **Search** tab inside LM Studio.
2. Search for a model (e.g. `llama-3.1-8b-instruct`).
3. Click **Download**. Models live in the LM Studio model cache.
4. Open the **Chat** tab and select the downloaded model to load it
   into memory.

## 3. Start the local server

1. Open the **Local Server** tab (the `↔` icon in the left rail).
2. Click **Start Server**. The default base URL is
   `http://localhost:1234/v1`.

## 4. Add to RLM Studio

In **Settings → LLM Providers → New**:

| Field    | Value                         |
|----------|-------------------------------|
| Backend  | `lmstudio`                    |
| Model    | Use the model ID shown in the Local Server tab |
| Base URL | `http://localhost:1234/v1`    |

No API key is required.

## 5. Test connection

Click **Test Connection** in Settings.

## 6. Common errors

- **Model not loaded** — LM Studio's server tab only serves the model
  you've loaded in Chat. Load a model first.
- **CORS error** — enable "CORS" in the Local Server tab options.
- **Port already in use** — LM Studio defaults to 1234. Change it in
  the server tab and update the base URL in RLM Studio to match.
