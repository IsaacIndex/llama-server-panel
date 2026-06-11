# Llama Server Panel

Small cross-platform launcher and GUI for running local `llama-server` roles:

- chat completions on `127.0.0.1:8080`
- embeddings on `127.0.0.1:8081`
- vision chat on `127.0.0.1:8082`
- optional juggler or single gateway for machines that cannot keep every model resident at once

The project is intentionally script-driven. Python owns configuration, process launch, model juggling, and the lightweight Tk GUI. The top-level `.sh` and `.cmd` files are convenience launchers for macOS/Linux and Windows.

## Prerequisites

- Python 3.10 or newer
- `llama-server` installed on `PATH`, or `LLAMA_SERVER_BIN` set in a local override file
- GGUF model files matching your local configuration
- Tk support in your Python installation when using the GUI

Model files are not included in this repository.

## Configuration

Default paths are repo-local:

- models: `./models`
- logs: `./logs`
- llama binary: `llama-server`

Create an ignored local override when your machine uses different paths, ports, models, or binary locations:

- `env.local.env` for portable `KEY=value` overrides
- `env.local.json` for structured overrides
- `env.local.sh` for legacy simple `export KEY=value` overrides
- `env.local.gui.json` for overrides saved by the GUI

Start from the example file:

```sh
cp .env.example env.local.env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example env.local.env
```

Configuration is applied in this order:

1. built-in defaults
2. current process environment
3. `env.local.env`
4. `env.local.json`
5. `env.local.sh`
6. `env.local.gui.json`
7. the role-specific tune file under `bench-results/tuned/`, if present

Relative model paths are resolved under `MODEL_DIR`. For example, `VISION_MMPROJ=mmproj-model.gguf` resolves to `./models/mmproj-model.gguf` with the default `MODEL_DIR`.

## Run

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

## GUI

Run the local GUI to select paths, import GGUF models, assign role models, start roles, and run the juggler.

macOS/Linux:

```sh
./start-gui.sh
```

Windows:

```powershell
.\start-gui.cmd
```

The GUI uses Python's standard Tk toolkit and the same runtime helpers as the CLI launchers. It saves machine-local choices to ignored `env.local.gui.json`.

## Juggler

Use the juggler when all three APIs should be callable but the machine cannot keep chat and vision resident at the same time. Embeddings stay available when possible; chat and vision are switched on demand.

macOS/Linux:

```sh
./juggle-models.sh
```

Windows:

```powershell
.\juggle-models.cmd
```

Dry-run the resolved ports and backend commands:

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

- chat: `http://127.0.0.1:8080/v1`, or `http://127.0.0.1:18080/v1` when port `8080` is already in use
- embeddings: `http://127.0.0.1:8081/v1`
- vision: `http://127.0.0.1:8082/v1`

Default supervised backend ports:

- chat backend: `18180`
- embedding backend: `18181`
- vision backend: `18182`

Useful juggler overrides:

- `JUGGLE_CHAT_PUBLIC_FALLBACK_PORT=18080`
- `JUGGLE_CHAT_BACKEND_PORT=18180`
- `JUGGLE_EMBED_BACKEND_PORT=18181`
- `JUGGLE_VISION_BACKEND_PORT=18182`
- `JUGGLE_SWITCH_TIMEOUT_SECONDS=600`
- `JUGGLE_STARTUP_TIMEOUT_SECONDS=900`
- `JUGGLE_REQUEST_TIMEOUT_SECONDS=3600`

## Service Gateway

The service gateway exposes one OpenAI-compatible base URL that routes chat, embedding, and vision requests to the right local role.

macOS/Linux:

```sh
./start-service.sh
```

Windows:

```powershell
.\start-service.cmd
```

The gateway listens on `0.0.0.0:8088` by default and keeps supervised backends on `127.0.0.1`. It does not configure TLS or API-key enforcement. Use it only on a trusted private network, or bind it to localhost:

```sh
./start-service.sh --bind 127.0.0.1
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
- `SERVICE_GATEWAY_BIND=127.0.0.1`

## Benchmark and Auto-Tune

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

## Quick API Checks

Chat:

```sh
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

```sh
curl http://127.0.0.1:8081/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-Embedding-4B-Q6_K.gguf",
    "input": "hello world"
  }'
```

Vision chat:

```sh
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

## Test

Run the local validation used by CI:

```sh
python -m compileall scripts
python -m unittest discover -s tests
python scripts/llama_role_command.py --help
python scripts/model_juggler.py --help
```

## Build Executables

Install build dependencies, then build the executable archive for the current platform:

```sh
python -m pip install -r requirements-build.txt
python scripts/build_release.py
```

The build writes ignored output under:

- `build/`
- `dist/`
- `release/`

The generated zip contains the GUI executable, `.env.example`, README, license placeholder, and security notes. The source launchers remain available from the repository checkout.

## Release

GitHub Actions builds downloadable GUI executables for macOS and Windows.

Release from a clean `main` branch:

```sh
git status --short
git tag v0.1.0
git push origin v0.1.0
```

The `Release` workflow runs on `v*` tags, builds macOS and Windows archives, writes SHA256 checksums, and attaches the files to a GitHub release.

Use the manual `workflow_dispatch` trigger when you only want to test packaging artifacts without publishing a release.

## Security Notes

- Do not commit `env.local.*`, `logs/`, `models/`, `bench-results/`, `build/`, `dist/`, or `release/`.
- Treat model paths, local logs, benchmark outputs, and GUI overrides as machine-local data.
- The service gateway does not enforce authentication. Do not expose it to untrusted networks.
- Store real external service keys in your shell environment or ignored local override files.

## License

No open-source license has been selected yet. Add a real license before public release. MIT is a simple option for permissive source distribution; Apache-2.0 is a stronger option when an explicit patent grant matters.
