# Local llama-server setup

This directory runs dedicated `llama-server` processes through cross-platform Python entrypoints:

- Chat API on `127.0.0.1:8080` with the configured `CHAT_MODEL`
- Embeddings API on `127.0.0.1:8081` with the configured `EMBED_MODEL`
- Vision chat API on `127.0.0.1:8082` with the configured `VISION_MODEL` and `VISION_MMPROJ`

The checked-in `.sh` commands now work on macOS/Linux without requiring `zsh`, and matching `.cmd` launchers are available for Windows.

## Prerequisites

- Python 3
- `llama-server` installed either on `PATH` or at `LLAMA_SERVER_BIN`
- the configured GGUF model files

## Model locations

Place the files here:

- `~/models/Mistral-Small-3.2-24B-Instruct-2506-BF16.gguf`
- `~/models/Qwen3-Embedding-4B-Q6_K.gguf`
- `~/models/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`
- the `VISION_MMPROJ` path configured by your local overrides

Defaults are mirrored from `env.sh`: model files under `$HOME/models` and logs under `./logs`.

For machine-specific paths, ports, or binary locations, create one of these ignored local override files in the repo root:

- `env.local.env` for portable `KEY=value` overrides
- `env.local.json` for structured overrides
- `env.local.sh` for legacy simple `export KEY=value` overrides

Configuration is applied in this order:

1. built-in defaults mirrored from `env.sh`
2. current process environment
3. `env.local.env`
4. `env.local.json`
5. `env.local.sh`
6. the role-specific tune file under `bench-results/tuned/`, if present

For embeddings, `start-embed.sh` also forwards:

- `EMBED_BATCH_SIZE` as `llama-server --batch-size` (logical batch limit)
- `EMBED_UBATCH_SIZE` as `llama-server --ubatch-size` (physical per-input token limit)

With the current defaults, a single embedding input that tokenizes past `EMBED_UBATCH_SIZE` tokens will fail even if the request contains only one text. The client should trim or split oversized inputs before calling `/v1/embeddings`.

## Start manually

macOS/Linux:

```sh
./start-chat.sh
./start-embed.sh
./start-vision.sh
```

Windows:

```powershell
.\start-chat.cmd
.\start-embed.cmd
.\start-vision.cmd
```

The direct launchers delegate to `scripts/llama_role_command.py`, which is the shared source of truth for role-specific `llama-server` arguments.

## Juggle full-context models

Use the juggler when you need all three APIs to be callable but cannot keep chat and vision resident at the same time. The juggler keeps embeddings always on when possible, and switches between chat and vision on demand while preserving each role's individual settings and full context window.

macOS/Linux:

```sh
./juggle-models.sh
```

Windows:

```powershell
.\juggle-models.cmd
```

Dry-run the resolved ports and backend commands without starting models:

```sh
./juggle-models.sh --dry-run
```

```powershell
.\juggle-models.cmd --dry-run
```

Validate configured files without starting the proxy:

```sh
./juggle-models.sh --check
```

```powershell
.\juggle-models.cmd --check
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

```sh
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

## Share as a nearby service

Use the service gateway when another nearby laptop should call this llama setup through one OpenAI-compatible base URL. This is intended for a trusted direct/private link; it does not configure networking, TLS, or API-key auth.

macOS/Linux:

```sh
./start-service.sh
```

Windows:

```powershell
.\start-service.cmd
```

The gateway listens on `0.0.0.0:8088` by default and keeps supervised llama backends on `127.0.0.1`. It prints local IPv4 client URL candidates and prefers link-local or bridge-style addresses when they are available.

On the other laptop, use the printed direct-link URL:

```sh
export OPENAI_BASE_URL=http://<printed-ip>:8088/v1
export OPENAI_API_KEY=local
```

Gateway routing:

- `POST /v1/embeddings` uses the embedding model.
- `POST /v1/chat/completions` uses vision when the request model matches `VISION_ALIAS` or the messages include image content; otherwise it uses the chat model.
- `GET /v1/models` returns the combined chat, embedding, and vision model list.

Useful service commands:

```sh
./start-service.sh --dry-run
./start-service.sh --check
./start-service.sh --port 8090
./start-service.sh --bind 127.0.0.1
```

```powershell
.\start-service.cmd --dry-run
.\start-service.cmd --check
.\start-service.cmd --port 8090
.\start-service.cmd --bind 127.0.0.1
```

Useful service overrides:

- `SERVICE_GATEWAY_PORT=8088`
- `SERVICE_GATEWAY_BIND=0.0.0.0`

## Benchmark and auto-tune

Auto-tune writes per-role overrides under `bench-results/tuned/`:

```sh
./auto-tune.sh chat
./auto-tune.sh embed
./auto-tune.sh vision
```

```powershell
.\auto-tune.cmd chat
.\auto-tune.cmd embed
.\auto-tune.cmd vision
```

Run the generic benchmark harness with:

```sh
./benchmark.sh
./benchmark.sh model1.gguf model2.gguf
```

```powershell
.\benchmark.cmd
.\benchmark.cmd model1.gguf model2.gguf
```

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

## macOS launchd

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
