# Local llama-server setup

This directory runs dedicated `llama-server` processes:

- Chat API on `127.0.0.1:8080` with `Qwen3-30B-A3B-Thinking-2507-UD-IQ3_XXS.gguf`
- Embeddings API on `127.0.0.1:8081` with `nomic-embed-text-v1.5.Q8_0.gguf`
- Vision chat API on `127.0.0.1:8083` for multimodal requests

## Model locations

Place the files here:

- `~/models/Qwen3-30B-A3B-Thinking-2507-UD-IQ3_XXS.gguf`
- `~/models/nomic-embed-text-v1.5.Q8_0.gguf`
- `~/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf`

The committed `env.sh` uses `$HOME/models` and repo-local `logs/` by default.
For machine-specific paths or ports, create `env.local.sh` next to `env.sh`; it is ignored by git and loaded automatically after `env.sh`.

## Start manually

```zsh
./start-chat.sh
./start-embed.sh
./start-vision.sh
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
    "model": "nomic-embed-text-v1.5.Q8_0",
    "input": "hello world"
  }'
```

Vision chat:

```zsh
curl http://127.0.0.1:8083/v1/chat/completions \
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

- `./logs/chat.log`
- `./logs/chat.err.log`
- `./logs/embed.log`
- `./logs/embed.err.log`
- `./logs/vision.log`
- `./logs/vision.err.log`

## Note on chat model sizing

This MacBook Pro has an Apple M1 Pro with 16 GB unified memory. The configured chat model is the largest practical Qwen3 thinking MoE option for this machine: `Qwen3-30B-A3B-Thinking-2507` at the 12.9 GB `UD-IQ3_XXS` GGUF quantization. The 4-bit files are 16.4 GB or larger before KV cache and runtime overhead, so they are not a practical default on this machine.

The chat server uses one slot, a 4k context, Qwen's recommended thinking-mode sampling defaults, q8 KV cache, disabled prompt cache, disabled warmup, and most MoE layers on CPU to keep memory pressure under control.
