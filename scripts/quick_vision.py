import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def encode_image(path: str) -> tuple[str, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image file not found: {path}")

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    mime_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return image_b64, mime_type


def extract_text(response_json: dict) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in response: {json.dumps(response_json, indent=2)}")

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(part for part in text_parts if part).strip()

    return str(content).strip()


def call_openrouter(prompt: str, image_path: str, model: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    image_b64, mime_type = encode_image(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                ],
            }
        ],
    }

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "quick_vision.py",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    return extract_text(response_json)


def prompt_for_input(label: str) -> str:
    value = input(label).strip()
    if not value:
        raise ValueError(f"Missing required input: {label.strip()}")
    return value


def main():
    try:
        if len(sys.argv) >= 3:
            prompt = sys.argv[1]
            image_path = sys.argv[2]
            model = sys.argv[3] if len(sys.argv) >= 4 else "openai/gpt-4.1-mini"
        else:
            print("Interactive mode")
            print("Leave model blank to use openai/gpt-4.1-mini")
            prompt = prompt_for_input("Prompt: ")
            image_path = prompt_for_input("Image path: ")
            model = input("Model [openai/gpt-4.1-mini]: ").strip() or "openai/gpt-4.1-mini"

        image_path = str(Path(image_path).expanduser())
        print(call_openrouter(prompt, image_path, model))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print('Usage: python quick_vision.py "describe this image" /path/to/image.png [model]', file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
