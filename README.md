# Local llama-server setup

This directory runs dedicated `llama-server` processes:

- Chat API on `127.0.0.1:8080` with the configured `CHAT_MODEL`
- Embeddings API on `127.0.0.1:8081` with the configured `EMBED_MODEL`
- Vision chat API on `127.0.0.1:8082` with the configured `VISION_MODEL` and `VISION_MMPROJ`

## Model locations

Place the files here:

- `~/models/Mistral-Small-3.2-24B-Instruct-2506-BF16.gguf`
- `~/models/Qwen3-Embedding-4B-Q6_K.gguf`
- `~/models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`
- the `VISION_MMPROJ` path configured in `env.sh` or `env.local.sh`

The committed `env.sh` uses `$HOME/models` and repo-local `logs/` by default.
For machine-specific paths or ports, create `env.local.sh` next to `env.sh`; it is ignored by git and loaded automatically after `env.sh`.

For embeddings, `start-embed.sh` also forwards:

- `EMBED_BATCH_SIZE` as `llama-server --batch-size` (logical batch limit)
- `EMBED_UBATCH_SIZE` as `llama-server --ubatch-size` (physical per-input token limit)

With the current defaults, a single embedding input that tokenizes past `EMBED_UBATCH_SIZE` tokens will fail even if the request contains only one text. The client should trim or split oversized inputs before calling `/v1/embeddings`.

## Start manually

```zsh
./start-chat.sh
./start-embed.sh
./start-vision.sh
```

The direct start scripts delegate to `scripts/llama_role_command.sh`, which is the shared source of truth for role-specific `llama-server` arguments. The helper loads configuration in this order:

1. `env.sh`
2. `env.local.sh`
3. the role-specific tune file under `bench-results/tuned/`, if present
4. an optional caller-provided port override

## Juggle full-context models

Use the juggler when you need all three APIs to be callable but cannot keep chat and vision resident at the same time. The juggler keeps embeddings always on when possible, and switches between chat and vision on demand while preserving each role's individual settings and full context window.

```zsh
./juggle-models.sh
```

Dry-run the resolved ports and backend commands without starting models:

```zsh
./juggle-models.sh --dry-run
```

Validate configured files without starting the proxy:

```zsh
./juggle-models.sh --check
```

Default public endpoints:

- Chat: `http://127.0.0.1:8080/v1` when port `8080` is free; otherwise `http://127.0.0.1:18080/v1`
- Embeddings: `http://127.0.0.1:8081/v1`
- Vision: `http://127.0.0.1:8082/v1`

Default supervised backend ports:

- Chat backend: `18180`
- Embedding backend: `18181`
- Vision backend: `18182`

If HKJC FastAPI or Podman is already occupying host port `8080`, the juggler falls chat back to `18080`. In that case use:

```zsh
LLAMA_SERVER_CHAT_BASE_URL_DOCKER=http://host.docker.internal:18080/v1
```

Useful juggler overrides:

- `JUGGLE_CHAT_PUBLIC_FALLBACK_PORT=18080`
- `JUGGLE_CHAT_BACKEND_PORT=18180`
- `JUGGLE_EMBED_BACKEND_PORT=18181`
- `JUGGLE_VISION_BACKEND_PORT=18182`
- `JUGGLE_SWITCH_TIMEOUT_SECONDS=600`
- `JUGGLE_STARTUP_TIMEOUT_SECONDS=900`
- `JUGGLE_REQUEST_TIMEOUT_SECONDS=3600`

## Quick test

Chat:

```zsh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-30b-a3b-thinking-2507",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

Embeddings:

```zsh
curl http://127.0.0.1:8081/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-Embedding-4B-Q6_K.gguf",
    "input": "hello world"
  }'
```

Vision chat:

```zsh
curl http://127.0.0.1:8082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-vl-3b-instruct",
    "messages": [
      {"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]}
    ]
  }'
```

## launchd

Load the agents after the model files exist:

```zsh
launchctl load ~/Library/LaunchAgents/com.example.llama-chat.plist
launchctl load ~/Library/LaunchAgents/com.example.llama-embed.plist
```

Restart after config changes:

```zsh
launchctl unload ~/Library/LaunchAgents/com.example.llama-chat.plist
launchctl unload ~/Library/LaunchAgents/com.example.llama-embed.plist
launchctl load ~/Library/LaunchAgents/com.example.llama-chat.plist
launchctl load ~/Library/LaunchAgents/com.example.llama-embed.plist
```

Logs:

- direct launch logs are printed by the foreground `start-*.sh` process
- juggler-supervised backend logs use `./logs/chat-18180.log`, `./logs/embed-18181.log`, and `./logs/vision-18182.log`

## Note on full-context juggling

The configured chat and vision context windows are intentionally large: `CHAT_CTX_SIZE=128000` and `VISION_CTX_SIZE=256000`. Running both heavy models at the same time is usually not practical on this laptop once KV cache and runtime overhead are included.

The juggler preserves those role-specific settings and trades latency for memory headroom: it keeps embeddings available, then starts either chat or vision on demand and stops the other heavy model before switching.
