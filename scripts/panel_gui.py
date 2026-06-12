#!/usr/bin/env python3
"""Lightweight cross-platform GUI for managing local llama-server models."""

from __future__ import annotations

import json
import base64
import mimetypes
import os
import queue
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from llama_runtime import (
    GUI_OVERRIDE_FILE,
    PanelError,
    build_role_argv,
    load_config,
    popen_session_kwargs,
    port_in_use,
    repo_dir,
    terminate_process,
    tune_file_exists,
    validate_role_files,
)
from model_juggler import (
    GATEWAY_DEFAULT_BIND,
    GATEWAY_DEFAULT_PORT,
    JugglerState,
    build_runtimes,
    check_gateway_port,
    make_gateway_handler,
    make_handler,
    parse_int_env,
)
from update_checker import UpdateCheckResult, UpdateInstallResult, apply_update, check_for_updates


ROLES = ("chat", "embed", "vision")
ROLE_LABELS = {
    "chat": "Chat",
    "embed": "Embedding",
    "vision": "Vision",
}
ROLE_PREFIX = {
    "chat": "CHAT",
    "embed": "EMBED",
    "vision": "VISION",
}
ROLE_ASSIGN_KEYS = {
    "chat": "CHAT_MODEL",
    "embed": "EMBED_MODEL",
    "vision": "VISION_MODEL",
}
CONFIG_KEYS = (
    "LLAMA_SERVER_BIN",
    "MODEL_DIR",
    "LOG_DIR",
    "CHAT_MODEL",
    "CHAT_ALIAS",
    "CHAT_PORT",
    "CHAT_CTX_SIZE",
    "CHAT_THREADS",
    "EMBED_MODEL",
    "EMBED_PORT",
    "EMBED_CTX_SIZE",
    "EMBED_THREADS",
    "EMBED_BATCH_SIZE",
    "EMBED_UBATCH_SIZE",
    "VISION_MODEL",
    "VISION_MMPROJ",
    "VISION_ALIAS",
    "VISION_PORT",
    "VISION_CTX_SIZE",
    "VISION_THREADS",
)
PATH_KEYS = {
    "LLAMA_SERVER_BIN",
    "MODEL_DIR",
    "LOG_DIR",
    "CHAT_MODEL",
    "EMBED_MODEL",
    "VISION_MODEL",
    "VISION_MMPROJ",
}
API_TIMEOUT_SECONDS = 3600
LOG_TAIL_BYTES = 64 * 1024
MAX_TEST_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_TEST_IMAGE_MIME_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}
START_PENDING_STATUSES = {"Loading", "Starting"}
IBM_BLUE = "#0f62fe"
IBM_BLUE_HOVER = "#0050e6"
IBM_BLUE_PRESSED = "#002d9c"
INK = "#161616"
INK_MUTED = "#525252"
INK_SUBTLE = "#6f6f6f"
CANVAS = "#ffffff"
SURFACE_1 = "#f4f4f4"
SURFACE_2 = "#e0e0e0"
SURFACE_3 = "#c6c6c6"
INVERSE_CANVAS = "#161616"
INVERSE_SURFACE = "#262626"
INVERSE_INK = "#ffffff"
HAIRLINE = "#e0e0e0"
SUCCESS = "#24a148"
WARNING = "#f1c21b"
ERROR = "#da1e28"
DISPLAY_FONT = ("IBM Plex Sans", 20, "normal")
HEADLINE_FONT = ("IBM Plex Sans", 12, "bold")
TITLE_FONT = ("IBM Plex Sans", 11, "bold")
BODY_FONT = ("IBM Plex Sans", 10, "normal")
BODY_EMPHASIS_FONT = ("IBM Plex Sans", 10, "bold")
CAPTION_FONT = ("IBM Plex Sans", 10, "normal")
MONO_FONT = ("Menlo", 9, "normal")
HEADER_COPY_WRAP = 720
FIELD_LABEL_WIDTH = 12


@dataclass
class JugglerHandle:
    state: JugglerState
    servers: list[ThreadingHTTPServer] = field(default_factory=list)
    threads: list[threading.Thread] = field(default_factory=list)

    def stop(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()
        self.state.shutdown()


def gui_override_path(panel_dir: Path) -> Path:
    return panel_dir / GUI_OVERRIDE_FILE


def start_status_for_auto_tune(auto_tune_enabled: bool, roles: Iterable[str], panel_dir: Path) -> str:
    if auto_tune_enabled and any(not tune_file_exists(role, panel_dir) for role in roles):
        return "Loading"
    return "Starting"


def load_gui_overrides(panel_dir: Path) -> Dict[str, str]:
    path = gui_override_path(panel_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise PanelError(f"{GUI_OVERRIDE_FILE} must contain a JSON object.")
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def save_gui_overrides(panel_dir: Path, overrides: Mapping[str, str]) -> Path:
    path = gui_override_path(panel_dir)
    payload = {key: value for key, value in overrides.items() if value != ""}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _relative_to(candidate: Path, parent: Path) -> Optional[Path]:
    try:
        return candidate.relative_to(parent)
    except ValueError:
        return None


def compact_path_value(path_text: str, *, base_dir: Optional[Path] = None, home: Optional[Path] = None) -> str:
    value = path_text.strip()
    if not value:
        return ""

    candidate = Path(os.path.expanduser(value))
    if not candidate.is_absolute():
        return value

    resolved = candidate.resolve()
    if base_dir is not None:
        relative = _relative_to(resolved, Path(os.path.expanduser(str(base_dir))).resolve())
        if relative is not None:
            text = relative.as_posix()
            return text or "."

    home_dir = Path(os.path.expanduser(str(home or Path.home()))).resolve()
    relative_home = _relative_to(resolved, home_dir)
    if relative_home is not None:
        return "~" if not relative_home.parts else f"~/{relative_home.as_posix()}"

    return str(resolved)


def model_config_value(path_text: str, model_dir: Path) -> str:
    return compact_path_value(path_text, base_dir=model_dir)


def model_dir_from_value(path_text: str, *, panel_dir: Path) -> Path:
    value = path_text.strip()
    if not value:
        return (panel_dir / "models").resolve()
    candidate = Path(os.path.expanduser(value))
    if not candidate.is_absolute():
        candidate = panel_dir / candidate
    return candidate.resolve()


def build_gui_overrides(values: Mapping[str, str], *, panel_dir: Path) -> Dict[str, str]:
    model_dir = model_dir_from_value(values.get("MODEL_DIR", ""), panel_dir=panel_dir)
    overrides: Dict[str, str] = {}
    for key in CONFIG_KEYS:
        value = values.get(key, "").strip()
        if not value:
            continue
        if key in {"CHAT_MODEL", "EMBED_MODEL", "VISION_MODEL", "VISION_MMPROJ"}:
            overrides[key] = model_config_value(value, model_dir)
        elif key == "LOG_DIR":
            overrides[key] = compact_path_value(value, base_dir=panel_dir)
        elif key in PATH_KEYS:
            overrides[key] = compact_path_value(value)
        else:
            overrides[key] = value
    return overrides


def import_model_file(source: Path, model_dir: Path, *, overwrite: bool = False) -> Path:
    source = Path(os.path.expanduser(str(source))).resolve()
    model_dir = Path(os.path.expanduser(str(model_dir))).resolve()
    if not source.is_file():
        raise PanelError(f"Model file not found: {source}")
    if source.suffix.lower() != ".gguf":
        raise PanelError("Only .gguf files are supported by this importer.")

    model_dir.mkdir(parents=True, exist_ok=True)
    destination = model_dir / source.name
    if source == destination:
        return destination
    if destination.exists() and not overwrite:
        raise PanelError(f"Model already exists in model directory: {destination.name}")
    shutil.copy2(source, destination)
    return destination


def discover_models(model_dir: Path) -> list[Path]:
    model_dir = Path(os.path.expanduser(str(model_dir))).resolve()
    if not model_dir.is_dir():
        return []
    return sorted(model_dir.glob("*.gguf"), key=lambda path: path.name.lower())


def image_data_url(image_path: Path, *, max_bytes: int = MAX_TEST_IMAGE_BYTES) -> str:
    path = Path(os.path.expanduser(str(image_path))).resolve()
    if not path.is_file():
        raise PanelError(f"Image file not found: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise PanelError(f"Image file is too large for the GUI tester: {path} ({size} bytes, max {max_bytes} bytes)")
    mime_type = mimetypes.guess_type(str(path))[0]
    if mime_type not in SUPPORTED_TEST_IMAGE_MIME_TYPES:
        supported = ", ".join(sorted(SUPPORTED_TEST_IMAGE_MIME_TYPES))
        raise PanelError(f"Unsupported image type for {path.name}. Supported types: {supported}")
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_chat_payload(prompt: str, model: str, *, image_path: Optional[Path] = None) -> Dict[str, object]:
    text = prompt.strip()
    if not text:
        raise PanelError("Enter a chat or vision prompt first.")
    if not model.strip():
        raise PanelError("Missing chat or vision model alias.")

    content: object = text
    if image_path is not None:
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
        ]

    return {
        "model": model.strip(),
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def build_embedding_payload(text: str, model: str) -> Dict[str, object]:
    value = text.strip()
    if not value:
        raise PanelError("Enter text to embed first.")
    if not model.strip():
        raise PanelError("Missing embedding model.")
    return {"model": model.strip(), "input": value}


def chat_model_id_for_role(config: Mapping[str, str], role: str) -> str:
    require_role(role)
    if role not in {"chat", "vision"}:
        raise PanelError(f"{ROLE_LABELS[role]} does not support chat completions.")
    key = f"{ROLE_PREFIX[role]}_ALIAS"
    value = config.get(key, "").strip()
    if not value:
        raise PanelError(f"Missing {key}; set it in local configuration before testing {ROLE_LABELS[role]}.")
    return value


def embedding_model_id_for_config(config: Mapping[str, str]) -> str:
    value = config.get("EMBED_MODEL", "").strip()
    if not value:
        raise PanelError("Missing EMBED_MODEL; set it in local configuration before testing embeddings.")
    return Path(value).name


def extract_chat_text(response_json: Mapping[str, object]) -> str:
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(response_json, indent=2)
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return str(choice)
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return json.dumps(choice, indent=2)
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, Mapping) and item.get("type") == "text"]
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def summarize_embedding_response(response_json: Mapping[str, object]) -> str:
    data = response_json.get("data")
    if not isinstance(data, list) or not data:
        return json.dumps(response_json, indent=2)
    first = data[0]
    if not isinstance(first, Mapping):
        return str(first)
    embedding = first.get("embedding")
    if not isinstance(embedding, list):
        return json.dumps(first, indent=2)
    preview = ", ".join(f"{float(value):.4f}" for value in embedding[:8] if isinstance(value, (int, float)))
    suffix = ", ..." if len(embedding) > 8 else ""
    return f"Embedding dimensions: {len(embedding)}\nFirst values: [{preview}{suffix}]"


def post_json(url: str, payload: Mapping[str, object], *, timeout: int = API_TIMEOUT_SECONDS) -> Dict[str, object]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise PanelError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise PanelError(f"Request failed for {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PanelError(f"Invalid JSON response from {url}: {exc}") from exc
    if not isinstance(decoded, dict):
        raise PanelError(f"Unexpected JSON response from {url}: {decoded!r}")
    return decoded


def default_assign_key_for_role(role: str) -> str:
    try:
        return ROLE_ASSIGN_KEYS[role]
    except KeyError as exc:
        raise PanelError(f"Unsupported role for model assignment: {role}") from exc


def require_role(role: str) -> None:
    if role not in ROLES:
        raise PanelError(f"Unsupported role: {role}")


def role_log_path(config: Mapping[str, str], role: str) -> Path:
    require_role(role)
    return Path(config["LOG_DIR"]) / f"{role}-gui.log"


def tail_file_text(path: Path, *, max_bytes: int = LOG_TAIL_BYTES) -> str:
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return f"No log yet: {path}\n"

    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(-max_bytes, os.SEEK_END)
            size_label = f"{max_bytes} bytes" if max_bytes < 1024 else f"{max_bytes // 1024} KiB"
            prefix = f"... showing last {size_label} of {path}\n\n"
        else:
            prefix = ""
        return prefix + fh.read().decode("utf-8", errors="replace")


def process_running(proc: Optional[subprocess.Popen[bytes]]) -> bool:
    return proc is not None and proc.poll() is None


def configure_carbon_style(root, style) -> None:
    root.configure(bg=CANVAS)
    root.option_add("*Font", BODY_FONT)
    root.option_add("*selectBackground", IBM_BLUE)
    root.option_add("*selectForeground", INVERSE_INK)

    style.configure(".", background=CANVAS, foreground=INK, font=BODY_FONT, borderwidth=0, relief="flat")
    style.configure("TFrame", background=CANVAS)
    style.configure("Canvas.TFrame", background=CANVAS)
    style.configure("Surface.TFrame", background=SURFACE_1)
    style.configure("Header.TFrame", background=CANVAS)
    style.configure("Toolbar.TFrame", background=CANVAS)

    style.configure("TLabel", background=CANVAS, foreground=INK, font=BODY_FONT)
    style.configure("Muted.TLabel", background=CANVAS, foreground=INK_MUTED, font=BODY_FONT)
    style.configure("Field.TLabel", background=SURFACE_1, foreground=INK_MUTED, font=BODY_FONT)
    style.configure("Eyebrow.TLabel", background=CANVAS, foreground=INK_MUTED, font=BODY_EMPHASIS_FONT)
    style.configure("Display.TLabel", background=CANVAS, foreground=INK, font=DISPLAY_FONT)
    style.configure("Headline.TLabel", background=CANVAS, foreground=INK, font=HEADLINE_FONT)
    style.configure("Section.TLabel", background=SURFACE_1, foreground=INK, font=TITLE_FONT)
    style.configure("Status.TLabel", background=CANVAS, foreground=IBM_BLUE, font=BODY_EMPHASIS_FONT)
    style.configure("Inverse.TLabel", background=INVERSE_CANVAS, foreground=INVERSE_INK, font=BODY_FONT)
    style.configure("InverseMuted.TLabel", background=INVERSE_CANVAS, foreground="#c6c6c6", font=CAPTION_FONT)

    style.configure("TLabelframe", background=SURFACE_1, bordercolor=HAIRLINE, borderwidth=1, relief="solid", padding=10)
    style.configure("TLabelframe.Label", background=SURFACE_1, foreground=INK, font=BODY_EMPHASIS_FONT)
    style.configure("Panel.TLabelframe", background=SURFACE_1, bordercolor=HAIRLINE, borderwidth=1, relief="solid", padding=10)
    style.configure("Panel.TLabelframe.Label", background=SURFACE_1, foreground=INK, font=BODY_EMPHASIS_FONT)

    style.configure("TButton", background=SURFACE_1, foreground=INK, bordercolor=HAIRLINE, focusthickness=1, focuscolor=IBM_BLUE, padding=(10, 5), relief="flat")
    style.map(
        "TButton",
        background=[("pressed", SURFACE_3), ("active", SURFACE_2), ("disabled", CANVAS)],
        foreground=[("disabled", INK_SUBTLE)],
        bordercolor=[("focus", IBM_BLUE)],
    )
    style.configure("Secondary.TButton", background=CANVAS, foreground=INK, bordercolor=HAIRLINE, padding=(10, 5), relief="flat")
    style.map(
        "Secondary.TButton",
        background=[("pressed", SURFACE_2), ("active", SURFACE_1), ("disabled", CANVAS)],
        foreground=[("disabled", INK_SUBTLE)],
        bordercolor=[("focus", IBM_BLUE)],
    )
    style.configure("Segment.TButton", background=CANVAS, foreground=INK, bordercolor=HAIRLINE, padding=(12, 6), relief="flat")
    style.map(
        "Segment.TButton",
        background=[("pressed", SURFACE_2), ("active", SURFACE_1), ("disabled", CANVAS)],
        foreground=[("disabled", INK_SUBTLE)],
        bordercolor=[("focus", IBM_BLUE)],
    )
    style.configure("SelectedSegment.TButton", background=IBM_BLUE, foreground=INVERSE_INK, bordercolor=IBM_BLUE, padding=(12, 6), relief="flat")
    style.map(
        "SelectedSegment.TButton",
        background=[("pressed", IBM_BLUE_PRESSED), ("active", IBM_BLUE_HOVER), ("disabled", IBM_BLUE)],
        foreground=[("disabled", INVERSE_INK)],
    )
    style.configure("Accent.TButton", background=IBM_BLUE, foreground=INVERSE_INK, bordercolor=IBM_BLUE, padding=(12, 6), relief="flat")
    style.map(
        "Accent.TButton",
        background=[("pressed", IBM_BLUE_PRESSED), ("active", IBM_BLUE_HOVER), ("disabled", SURFACE_2)],
        foreground=[("disabled", INK_SUBTLE)],
    )
    style.configure("Danger.TButton", background=ERROR, foreground=INVERSE_INK, bordercolor=ERROR, padding=(10, 5), relief="flat")
    style.map("Danger.TButton", background=[("pressed", "#750e13"), ("active", "#ba1b23")])

    style.configure("TCheckbutton", background=CANVAS, foreground=INK, font=BODY_FONT, padding=(2, 2))
    style.configure("TEntry", fieldbackground=SURFACE_1, foreground=INK, bordercolor=HAIRLINE, lightcolor=HAIRLINE, darkcolor=HAIRLINE, padding=(8, 4), relief="flat")
    style.map("TEntry", bordercolor=[("focus", IBM_BLUE)], fieldbackground=[("disabled", SURFACE_2)])
    style.configure("TCombobox", fieldbackground=SURFACE_1, background=SURFACE_1, foreground=INK, bordercolor=HAIRLINE, arrowcolor=IBM_BLUE, padding=(6, 4), relief="flat")
    style.map("TCombobox", bordercolor=[("focus", IBM_BLUE)], fieldbackground=[("readonly", SURFACE_1)])
    style.configure("TNotebook", background=CANVAS, borderwidth=0, tabmargins=(0, 0, 0, 0))
    style.configure("TNotebook.Tab", background=SURFACE_1, foreground=INK_MUTED, padding=(14, 7), borderwidth=0, font=BODY_FONT)
    style.map("TNotebook.Tab", background=[("selected", CANVAS), ("active", SURFACE_2)], foreground=[("selected", INK), ("active", INK)])
    style.configure("TSeparator", background=HAIRLINE)
    style.configure("Vertical.TScrollbar", background=SURFACE_1, troughcolor=CANVAS, bordercolor=CANVAS, arrowcolor=IBM_BLUE)
    style.configure("Horizontal.TScrollbar", background=SURFACE_1, troughcolor=CANVAS, bordercolor=CANVAS, arrowcolor=IBM_BLUE)


def run_gui() -> int:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("Python tkinter is required for the GUI. Install a Python build that includes Tk.", file=sys.stderr)
        return 1

    class PanelApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.panel_dir = repo_dir()
            self.values: Dict[str, tk.StringVar] = {}
            self.role_processes: Dict[str, subprocess.Popen[bytes]] = {}
            self.role_start_buttons: Dict[str, ttk.Button] = {}
            self.role_stop_buttons: Dict[str, ttk.Button] = {}
            self.log_texts: Dict[str, tk.Text] = {}
            self.juggler_handle: Optional[JugglerHandle] = None
            self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
            self.update_check_running = False

            self.root.title("Llama Server Panel")
            self.root.geometry("1240x820")
            self.root.minsize(1040, 680)

            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            configure_carbon_style(self.root, style)

            self.auto_tune = tk.BooleanVar(value=True)
            self.juggler_mode = tk.StringVar(value="gateway")
            self.gateway_bind = tk.StringVar(value=GATEWAY_DEFAULT_BIND)
            self.gateway_port = tk.StringVar(value=str(GATEWAY_DEFAULT_PORT))
            self.selected_assign_key = tk.StringVar(value="CHAT_MODEL")
            self.test_image_path = tk.StringVar(value="")
            self.status_vars = {role: tk.StringVar(value="Stopped") for role in ROLES}
            self.juggler_status = tk.StringVar(value="Stopped")

            self._build_layout(ttk, tk, filedialog, messagebox)
            self.reload_config()
            self.refresh_model_list()
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.root.after(400, self.poll_queue)
            self.root.after(1500, self.refresh_status)
            self.root.after(1000, self.refresh_logs)

        def _build_layout(self, ttk, tk, filedialog, messagebox) -> None:
            main = ttk.Frame(self.root, padding=0, style="Canvas.TFrame")
            main.pack(fill=tk.BOTH, expand=True)
            main.columnconfigure(0, weight=1, uniform="main")
            main.columnconfigure(1, weight=2, uniform="main")
            main.rowconfigure(2, weight=4)
            main.rowconfigure(3, weight=1)

            header = ttk.Frame(main, padding=(20, 14, 20, 12), style="Header.TFrame")
            header.grid(row=0, column=0, columnspan=2, sticky="ew")
            header.columnconfigure(0, weight=1)
            ttk.Label(header, text="LOCAL MODEL OPERATIONS", style="Eyebrow.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(header, text="Llama Server Panel", style="Display.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
            ttk.Label(
                header,
                text="Run chat, embedding, vision, and gateway roles from one local control surface.",
                style="Muted.TLabel",
                wraplength=HEADER_COPY_WRAP,
                justify="left",
            ).grid(row=2, column=0, sticky="w", pady=(4, 0))
            actions = ttk.Frame(header, style="Toolbar.TFrame")
            actions.grid(row=1, column=1, rowspan=2, sticky="ne")
            self.check_updates_button = ttk.Button(actions, text="Check Updates", command=self.check_updates, style="Secondary.TButton")
            self.check_updates_button.grid(row=0, column=0, padx=(0, 8))
            ttk.Button(actions, text="Save Config", command=self.save_config, style="Accent.TButton").grid(row=0, column=1, padx=(0, 8))
            ttk.Button(actions, text="Reload", command=self.reload_config, style="Secondary.TButton").grid(row=0, column=2)

            general = ttk.LabelFrame(main, text="Paths", padding=10, style="Panel.TLabelframe")
            general.grid(row=1, column=0, columnspan=2, sticky="ew", padx=20, pady=(0, 10))
            general.columnconfigure(1, weight=1)
            self._entry_row(general, ttk, "llama-server", "LLAMA_SERVER_BIN", 0, browse_file=True)
            self._entry_row(general, ttk, "Model dir", "MODEL_DIR", 1, browse_dir=True)
            self._entry_row(general, ttk, "Log dir", "LOG_DIR", 2, browse_dir=True)

            library = ttk.LabelFrame(main, text="Model Library", padding=10, style="Panel.TLabelframe")
            library.grid(row=2, column=0, sticky="nsew", padx=(20, 6), pady=(0, 10))
            library.columnconfigure(0, weight=1)
            library.rowconfigure(0, weight=1)
            self.model_list = tk.Listbox(
                library,
                activestyle="none",
                exportselection=False,
                bg=CANVAS,
                fg=INK,
                selectbackground=IBM_BLUE,
                selectforeground=INVERSE_INK,
                highlightthickness=1,
                highlightbackground=HAIRLINE,
                highlightcolor=IBM_BLUE,
                borderwidth=0,
                relief="flat",
                font=BODY_FONT,
            )
            self.model_list.grid(row=0, column=0, sticky="nsew")
            model_scroll = ttk.Scrollbar(library, orient=tk.VERTICAL, command=self.model_list.yview)
            model_scroll.grid(row=0, column=1, sticky="ns")
            self.model_list.configure(yscrollcommand=model_scroll.set)

            library_actions = ttk.Frame(library, style="Surface.TFrame")
            library_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
            library_actions.columnconfigure(0, weight=1)
            ttk.Button(library_actions, text="Import GGUF", command=self.import_model, style="Secondary.TButton").grid(row=0, column=0, sticky="w")
            ttk.Button(library_actions, text="Refresh", command=self.refresh_model_list, style="Secondary.TButton").grid(row=0, column=1, sticky="e")
            ttk.Combobox(
                library_actions,
                textvariable=self.selected_assign_key,
                values=("CHAT_MODEL", "EMBED_MODEL", "VISION_MODEL", "VISION_MMPROJ"),
                state="readonly",
                width=18,
            ).grid(row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 6))
            ttk.Button(library_actions, text="Assign", command=self.assign_selected_model, style="Secondary.TButton").grid(row=1, column=1, sticky="e", pady=(6, 0))

            right = ttk.Frame(main, style="Canvas.TFrame")
            right.grid(row=2, column=1, sticky="nsew", padx=(6, 20), pady=(0, 10))
            right.columnconfigure(0, weight=1)
            right.rowconfigure(0, weight=1)

            main_tabs = ttk.Notebook(right)
            main_tabs.grid(row=0, column=0, sticky="nsew")

            role_tab, role_frame = self._scrollable_tab_page(main_tabs, ttk, tk, padding=(10, 10, 10, 8))
            role_frame.columnconfigure(0, weight=1)
            main_tabs.add(role_tab, text="Roles")

            role_selector = ttk.Frame(role_frame, style="Canvas.TFrame")
            role_selector.grid(row=0, column=0, sticky="w", pady=(0, 8))
            self.active_role = tk.StringVar(value="chat")
            self.role_selector_buttons: Dict[str, ttk.Button] = {}
            self.role_pages: Dict[str, ttk.Frame] = {}
            for role in ROLES:
                button = ttk.Button(
                    role_selector,
                    text=ROLE_LABELS[role],
                    command=lambda selected=role: self.set_active_role(selected),
                    style="Segment.TButton",
                )
                button.grid(row=0, column=len(self.role_selector_buttons), sticky="w", padx=(0, 6))
                self.role_selector_buttons[role] = button

                role_page = ttk.LabelFrame(role_frame, text=f"{ROLE_LABELS[role]} Role", padding=10, style="Panel.TLabelframe")
                role_page.grid(row=1, column=0, sticky="nsew")
                role_page.columnconfigure(1, weight=1)
                self._role_block(role_page, ttk, role, 0)
                self.role_pages[role] = role_page
            self.set_active_role("chat")

            juggler = ttk.LabelFrame(role_frame, text="Juggler", padding=10, style="Panel.TLabelframe")
            juggler.grid(row=2, column=0, sticky="ew", pady=(10, 0))
            juggler.columnconfigure(1, weight=1)
            juggler.columnconfigure(3, weight=1)
            ttk.Label(juggler, text="Mode", style="Field.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Combobox(
                juggler,
                textvariable=self.juggler_mode,
                values=("gateway", "role proxies"),
                state="readonly",
                width=14,
            ).grid(row=0, column=1, sticky="ew", padx=(8, 16))
            ttk.Label(juggler, text="Bind", style="Field.TLabel").grid(row=0, column=2, sticky="w")
            ttk.Entry(juggler, textvariable=self.gateway_bind, width=14).grid(row=0, column=3, sticky="ew", padx=(8, 16))
            ttk.Label(juggler, text="Port", style="Field.TLabel").grid(row=0, column=4, sticky="w")
            ttk.Entry(juggler, textvariable=self.gateway_port, width=8).grid(row=0, column=5, sticky="w", padx=(8, 16))
            ttk.Label(juggler, textvariable=self.juggler_status, style="Status.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
            ttk.Checkbutton(juggler, text="Auto-tune", variable=self.auto_tune).grid(row=1, column=2, sticky="w", pady=(8, 0))
            juggler_actions = ttk.Frame(juggler, style="Surface.TFrame")
            juggler_actions.grid(row=1, column=3, columnspan=3, sticky="e", pady=(8, 0))
            ttk.Button(juggler_actions, text="Start", command=self.start_juggler, style="Accent.TButton").grid(row=0, column=0, padx=(0, 6))
            ttk.Button(juggler_actions, text="Stop", command=self.stop_juggler, style="Secondary.TButton").grid(row=0, column=1, padx=(0, 6))
            ttk.Button(juggler_actions, text="Check", command=self.check_juggler, style="Secondary.TButton").grid(row=0, column=2)

            tester_tab, tester = self._scrollable_tab_page(main_tabs, ttk, tk, padding=(10, 10, 10, 8))
            main_tabs.add(tester_tab, text="API Tester")
            self._api_tester_block(tester, ttk, tk)

            logs_tab, logs = self._scrollable_tab_page(main_tabs, ttk, tk, padding=(10, 10, 10, 8))
            main_tabs.add(logs_tab, text="Logs")
            self._build_log_panel(logs, ttk, tk)

            output_frame = ttk.LabelFrame(main, text="Output", padding=10, style="Panel.TLabelframe")
            output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=20, pady=(0, 16))
            output_frame.columnconfigure(0, weight=1)
            output_frame.rowconfigure(0, weight=1)
            self.output = tk.Text(
                output_frame,
                height=5,
                wrap="none",
                bg=INVERSE_CANVAS,
                fg=INVERSE_INK,
                insertbackground=INVERSE_INK,
                selectbackground=IBM_BLUE,
                selectforeground=INVERSE_INK,
                borderwidth=0,
                highlightthickness=0,
                padx=12,
                pady=8,
                font=MONO_FONT,
            )
            self.output.grid(row=0, column=0, sticky="nsew")
            output_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
            output_scroll.grid(row=0, column=1, sticky="ns")
            output_x_scroll = ttk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=self.output.xview)
            output_x_scroll.grid(row=1, column=0, sticky="ew")
            self.output.configure(yscrollcommand=output_scroll.set, xscrollcommand=output_x_scroll.set)

        def _scrollable_tab_page(self, notebook, ttk, tk, *, padding):
            outer = ttk.Frame(notebook, style="Canvas.TFrame")
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(0, weight=1)

            canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, bg=CANVAS)
            canvas.grid(row=0, column=0, sticky="nsew")
            y_scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
            y_scroll.grid(row=0, column=1, sticky="ns")
            canvas.configure(yscrollcommand=y_scroll.set)

            inner = ttk.Frame(canvas, padding=padding, style="Canvas.TFrame")
            window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

            def refresh_scroll_region(_event=None) -> None:
                canvas.configure(scrollregion=canvas.bbox("all"))

            def fit_inner_width(event) -> None:
                canvas.itemconfigure(window_id, width=event.width)
                refresh_scroll_region()

            def on_mousewheel(event) -> str:
                if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
                    canvas.yview_scroll(-1, "units")
                else:
                    canvas.yview_scroll(1, "units")
                return "break"

            def bind_mousewheel(_event=None) -> None:
                canvas.bind_all("<MouseWheel>", on_mousewheel)
                canvas.bind_all("<Button-4>", on_mousewheel)
                canvas.bind_all("<Button-5>", on_mousewheel)

            def unbind_mousewheel(_event=None) -> None:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")

            inner.bind("<Configure>", refresh_scroll_region)
            canvas.bind("<Configure>", fit_inner_width)
            for widget in (outer, canvas, inner):
                widget.bind("<Enter>", bind_mousewheel)
            outer.bind("<Leave>", unbind_mousewheel)
            return outer, inner

        def _var(self, key: str) -> "tk.StringVar":
            if key not in self.values:
                import tkinter as tk

                self.values[key] = tk.StringVar()
            return self.values[key]

        def _entry_row(self, parent, ttk, label: str, key: str, row: int, *, browse_file: bool = False, browse_dir: bool = False) -> None:
            ttk.Label(parent, text=label, style="Field.TLabel", width=FIELD_LABEL_WIDTH).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(parent, textvariable=self._var(key)).grid(row=row, column=1, sticky="ew", padx=(8, 6), pady=3)
            if browse_file:
                ttk.Button(parent, text="Browse", command=lambda: self.browse_file(key), style="Secondary.TButton").grid(row=row, column=2, sticky="e", pady=3)
            elif browse_dir:
                ttk.Button(parent, text="Browse", command=lambda: self.browse_dir(key), style="Secondary.TButton").grid(row=row, column=2, sticky="e", pady=3)

        def _role_block(self, parent, ttk, role: str, start_row: int) -> int:
            prefix = ROLE_PREFIX[role]
            parent.columnconfigure(0, minsize=120)
            parent.columnconfigure(1, weight=1)
            ttk.Separator(parent).grid(row=start_row, column=0, columnspan=5, sticky="ew", pady=(4, 8))
            ttk.Label(parent, text=ROLE_LABELS[role], style="Headline.TLabel").grid(row=start_row + 1, column=0, sticky="w")
            ttk.Label(parent, textvariable=self.status_vars[role], style="Status.TLabel").grid(row=start_row + 1, column=1, sticky="w")
            start_button = ttk.Button(parent, text="Start", command=lambda r=role: self.start_role(r), style="Accent.TButton")
            start_button.grid(row=start_row + 1, column=2, padx=(6, 0))
            stop_button = ttk.Button(parent, text="Stop", command=lambda r=role: self.stop_role(r), style="Secondary.TButton")
            stop_button.grid(row=start_row + 1, column=3, padx=(6, 0))
            self.role_start_buttons[role] = start_button
            self.role_stop_buttons[role] = stop_button
            ttk.Button(parent, text="Check", command=lambda r=role: self.check_role(r), style="Secondary.TButton").grid(row=start_row + 1, column=4, padx=(6, 0))
            self.update_role_controls(role)

            row = start_row + 2
            self._entry_row(parent, ttk, "Model", f"{prefix}_MODEL", row, browse_file=True)
            row += 1
            if role == "vision":
                self._entry_row(parent, ttk, "MMProj", "VISION_MMPROJ", row, browse_file=True)
                row += 1

            compact = ttk.Frame(parent, padding=(0, 4, 0, 0), style="Canvas.TFrame")
            compact.grid(row=row, column=0, columnspan=5, sticky="ew", pady=(0, 4))
            compact.columnconfigure(1, weight=1)
            compact.columnconfigure(3, weight=1)
            compact.columnconfigure(5, weight=1)
            for idx, key in enumerate((f"{prefix}_PORT", f"{prefix}_CTX_SIZE", f"{prefix}_THREADS")):
                ttk.Label(compact, text=key.replace(f"{prefix}_", "").title(), style="Muted.TLabel").grid(row=0, column=idx * 2, sticky="w", padx=(0 if idx == 0 else 12, 4))
                ttk.Entry(compact, textvariable=self._var(key), width=10).grid(row=0, column=idx * 2 + 1, sticky="ew")
            return row + 1

        def _api_tester_block(self, parent, ttk, tk) -> None:
            parent.columnconfigure(0, weight=1)

            ttk.Label(parent, text="Chat / vision prompt", style="Headline.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
            self.chat_test_input = tk.Text(
                parent,
                height=4,
                wrap="word",
                bg=SURFACE_1,
                fg=INK,
                insertbackground=INK,
                selectbackground=IBM_BLUE,
                selectforeground=INVERSE_INK,
                borderwidth=0,
                highlightthickness=1,
                highlightbackground=HAIRLINE,
                highlightcolor=IBM_BLUE,
                padx=10,
                pady=8,
                font=BODY_FONT,
            )
            self.chat_test_input.insert("1.0", "Describe this image, or answer this text-only prompt.")
            self.chat_test_input.grid(row=1, column=0, sticky="ew", pady=(0, 6))

            image_row = ttk.Frame(parent, style="Canvas.TFrame")
            image_row.grid(row=2, column=0, sticky="ew", pady=(0, 6))
            image_row.columnconfigure(1, weight=1)
            ttk.Label(image_row, text="Image", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
            ttk.Entry(image_row, textvariable=self.test_image_path).grid(row=0, column=1, sticky="ew", padx=(0, 8))
            ttk.Button(image_row, text="Browse", command=self.browse_test_image).grid(row=0, column=2, sticky="e")
            ttk.Button(image_row, text="Clear", command=lambda: self.test_image_path.set("")).grid(row=0, column=3, sticky="e", padx=(8, 0))

            self.chat_test_button = ttk.Button(parent, text="Send Chat / Vision", command=self.run_chat_vision_test, style="Accent.TButton")
            self.chat_test_button.grid(row=3, column=0, sticky="e", pady=(0, 8))

            ttk.Separator(parent).grid(row=4, column=0, sticky="ew", pady=(0, 6))
            ttk.Label(parent, text="Embedding text", style="Headline.TLabel").grid(row=5, column=0, sticky="w", pady=(0, 4))
            self.embedding_test_input = tk.Text(
                parent,
                height=3,
                wrap="word",
                bg=SURFACE_1,
                fg=INK,
                insertbackground=INK,
                selectbackground=IBM_BLUE,
                selectforeground=INVERSE_INK,
                borderwidth=0,
                highlightthickness=1,
                highlightbackground=HAIRLINE,
                highlightcolor=IBM_BLUE,
                padx=10,
                pady=8,
                font=BODY_FONT,
            )
            self.embedding_test_input.insert("1.0", "Text to embed")
            self.embedding_test_input.grid(row=6, column=0, sticky="ew", pady=(0, 6))
            self.embedding_test_button = ttk.Button(parent, text="Embed Text", command=self.run_embedding_test, style="Accent.TButton")
            self.embedding_test_button.grid(row=7, column=0, sticky="e")

        def browse_test_image(self) -> None:
            path = filedialog.askopenfilename(
                title="Select image",
                filetypes=(
                    ("Images", "*.png *.jpg *.jpeg *.webp *.gif"),
                    ("All files", "*.*"),
                ),
                initialdir=str(Path.home()),
            )
            if path:
                self.test_image_path.set(path)

        def _text_value(self, widget) -> str:
            return widget.get("1.0", "end").strip()

        def _role_url(self, config: Mapping[str, str], role: str, path: str) -> str:
            prefix = ROLE_PREFIX[role]
            return f"http://{config['LLAMA_HOST']}:{config[f'{prefix}_PORT']}{path}"

        def run_chat_vision_test(self) -> None:
            prompt = self._text_value(self.chat_test_input)
            image_text = self.test_image_path.get().strip()
            self.chat_test_button.configure(state="disabled")
            self.append_output("Sending chat / vision test request...\n")
            threading.Thread(target=self._chat_vision_test_worker, args=(prompt, image_text), daemon=True).start()

        def _chat_vision_test_worker(self, prompt: str, image_text: str) -> None:
            try:
                config = load_config(self.panel_dir, apply_tune=False)
                image_path = Path(image_text) if image_text else None
                role = "vision" if image_path is not None else "chat"
                payload = build_chat_payload(prompt, chat_model_id_for_role(config, role), image_path=image_path)
                response = post_json(self._role_url(config, role, "/v1/chat/completions"), payload)
                self.queue.put(("api_test_result", (f"{ROLE_LABELS[role]} response", extract_chat_text(response), "chat")))
            except Exception as exc:
                self.queue.put(("api_test_error", ("Chat / vision test failed", str(exc), "chat")))

        def run_embedding_test(self) -> None:
            text = self._text_value(self.embedding_test_input)
            self.embedding_test_button.configure(state="disabled")
            self.append_output("Sending embedding test request...\n")
            threading.Thread(target=self._embedding_test_worker, args=(text,), daemon=True).start()

        def _embedding_test_worker(self, text: str) -> None:
            try:
                config = load_config(self.panel_dir, apply_tune=False)
                payload = build_embedding_payload(text, embedding_model_id_for_config(config))
                response = post_json(self._role_url(config, "embed", "/v1/embeddings"), payload)
                self.queue.put(("api_test_result", ("Embedding response", summarize_embedding_response(response), "embed")))
            except Exception as exc:
                self.queue.put(("api_test_error", ("Embedding test failed", str(exc), "embed")))

        def _build_log_panel(self, parent, ttk, tk) -> None:
            parent.columnconfigure(0, weight=1)
            parent.rowconfigure(0, weight=1)
            log_frame = ttk.LabelFrame(parent, text="Llama Server Logs", padding=10, style="Panel.TLabelframe")
            log_frame.grid(row=0, column=0, sticky="nsew")
            log_frame.columnconfigure(0, weight=1)
            log_frame.rowconfigure(0, weight=1)

            log_tabs = ttk.Notebook(log_frame)
            log_tabs.grid(row=0, column=0, sticky="nsew")
            for role in ROLES:
                tab = ttk.Frame(log_tabs, style="Canvas.TFrame")
                tab.columnconfigure(0, weight=1)
                tab.rowconfigure(0, weight=1)
                text = tk.Text(
                    tab,
                    height=7,
                    wrap="none",
                    bg=INVERSE_CANVAS,
                    fg=INVERSE_INK,
                    insertbackground=INVERSE_INK,
                    selectbackground=IBM_BLUE,
                    selectforeground=INVERSE_INK,
                    borderwidth=0,
                    highlightthickness=0,
                    padx=12,
                    pady=8,
                    font=MONO_FONT,
                )
                text.grid(row=0, column=0, sticky="nsew")
                y_scroll = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=text.yview)
                y_scroll.grid(row=0, column=1, sticky="ns")
                x_scroll = ttk.Scrollbar(tab, orient=tk.HORIZONTAL, command=text.xview)
                x_scroll.grid(row=1, column=0, sticky="ew")
                text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set, state="disabled")
                self.log_texts[role] = text
                log_tabs.add(tab, text=ROLE_LABELS[role])

            ttk.Button(log_frame, text="Refresh Logs", command=self.refresh_logs_once).grid(row=1, column=0, sticky="e", pady=(6, 0))

        def set_active_role(self, role: str) -> None:
            require_role(role)
            self.active_role.set(role)
            self.selected_assign_key.set(default_assign_key_for_role(role))
            for candidate, page in self.role_pages.items():
                if candidate == role:
                    page.grid()
                else:
                    page.grid_remove()
            for candidate, button in self.role_selector_buttons.items():
                button.configure(style="SelectedSegment.TButton" if candidate == role else "Segment.TButton")

        def browse_file(self, key: str) -> None:
            path = filedialog.askopenfilename(title=f"Select {key}", initialdir=self.values.get("MODEL_DIR", self._var("MODEL_DIR")).get() or str(Path.home()))
            if path:
                self._var(key).set(path)

        def browse_dir(self, key: str) -> None:
            path = filedialog.askdirectory(title=f"Select {key}", initialdir=self._var(key).get() or str(Path.home()))
            if path:
                self._var(key).set(path)
                if key == "MODEL_DIR":
                    self.refresh_model_list()

        def reload_config(self) -> None:
            try:
                config = load_config(self.panel_dir, apply_tune=False)
            except PanelError as exc:
                messagebox.showerror("Configuration error", str(exc))
                return
            for key in CONFIG_KEYS:
                self._var(key).set(config.get(key, ""))
            self.append_output(f"Loaded config from {self.panel_dir}\n")

        def current_values(self) -> Dict[str, str]:
            return {key: self._var(key).get() for key in CONFIG_KEYS}

        def save_config(self) -> bool:
            try:
                overrides = build_gui_overrides(self.current_values(), panel_dir=self.panel_dir)
                path = save_gui_overrides(self.panel_dir, overrides)
                self.append_output(f"Saved GUI overrides to {path}\n")
                self.refresh_model_list()
                return True
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc))
                return False

        def refresh_model_list(self) -> None:
            self.model_list.delete(0, "end")
            model_dir = model_dir_from_value(self._var("MODEL_DIR").get(), panel_dir=self.panel_dir)
            for path in discover_models(model_dir):
                self.model_list.insert("end", path.name)

        def selected_model_path(self) -> Optional[Path]:
            selection = self.model_list.curselection()
            if not selection:
                return None
            name = self.model_list.get(selection[0])
            return model_dir_from_value(self._var("MODEL_DIR").get(), panel_dir=self.panel_dir) / name

        def import_model(self) -> None:
            source_text = filedialog.askopenfilename(
                title="Import GGUF",
                filetypes=(("GGUF models", "*.gguf"), ("All files", "*.*")),
                initialdir=str(Path.home()),
            )
            if not source_text:
                return
            try:
                destination = import_model_file(Path(source_text), Path(self._var("MODEL_DIR").get()), overwrite=False)
                self.append_output(f"Imported {destination.name} into {destination.parent}\n")
                self.refresh_model_list()
                self.select_model_name(destination.name)
            except PanelError as exc:
                messagebox.showerror("Import failed", str(exc))

        def select_model_name(self, name: str) -> None:
            for index in range(self.model_list.size()):
                if self.model_list.get(index) == name:
                    self.model_list.selection_clear(0, "end")
                    self.model_list.selection_set(index)
                    self.model_list.see(index)
                    return

        def assign_selected_model(self) -> None:
            path = self.selected_model_path()
            if path is None:
                messagebox.showinfo("No model selected", "Select a model first.")
                return
            key = self.selected_assign_key.get()
            self._var(key).set(str(path))
            if self.save_config():
                self.append_output(f"Assigned {path.name} to {key}\n")

        def check_role(self, role: str) -> None:
            if not self.save_config():
                return
            try:
                config = load_config(self.panel_dir, role=role)
                validate_role_files(role, config)
                prefix = ROLE_PREFIX[role]
                host = config["LLAMA_HOST"]
                port = int(config[f"{prefix}_PORT"])
                if port_in_use(host, port):
                    self.append_output(f"{ROLE_LABELS[role]} check: files ok, port {host}:{port} is already active\n")
                else:
                    self.append_output(f"{ROLE_LABELS[role]} check passed on {host}:{port}\n")
            except Exception as exc:
                messagebox.showerror(f"{ROLE_LABELS[role]} check failed", str(exc))

        def start_role(self, role: str) -> None:
            require_role(role)
            if not self.save_config():
                return
            if process_running(self.role_processes.get(role)):
                self.append_output(f"{ROLE_LABELS[role]} is already running from this GUI\n")
                return
            self.status_vars[role].set(start_status_for_auto_tune(self.auto_tune.get(), (role,), self.panel_dir))
            self.update_role_controls(role)
            threading.Thread(target=self._start_role_worker, args=(role,), daemon=True).start()

        def _start_role_worker(self, role: str) -> None:
            try:
                config = load_config(self.panel_dir, role=role)
                validate_role_files(role, config)
                prefix = ROLE_PREFIX[role]
                host = config["LLAMA_HOST"]
                port = int(config[f"{prefix}_PORT"])
                if port_in_use(host, port):
                    raise PanelError(f"Port {host}:{port} is already in use.")
                argv = build_role_argv(role, panel_dir=self.panel_dir, auto_tune=self.auto_tune.get())
                log_path = role_log_path(config, role)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_fh = open(log_path, "ab", buffering=0)
                try:
                    proc = subprocess.Popen(
                        argv,
                        cwd=str(self.panel_dir),
                        stdout=log_fh,
                        stderr=subprocess.STDOUT,
                        **popen_session_kwargs(),
                    )
                finally:
                    log_fh.close()
                self.queue.put(("role_started", (role, proc, log_path)))
            except Exception as exc:
                self.queue.put(("role_error", (role, str(exc))))

        def stop_role(self, role: str) -> None:
            require_role(role)
            proc = self.role_processes.get(role)
            if not process_running(proc):
                self.role_processes.pop(role, None)
                self.status_vars[role].set("Stopped")
                self.update_role_controls(role)
                return
            try:
                terminate_process(proc)
                self.append_output(f"Stopped {ROLE_LABELS[role]}\n")
            except Exception as exc:
                messagebox.showerror(f"Stop {ROLE_LABELS[role]} failed", str(exc))
            finally:
                self.role_processes.pop(role, None)
            self.status_vars[role].set("Stopped")
            self.update_role_controls(role)

        def check_juggler(self) -> None:
            if not self.save_config():
                return
            try:
                gateway = self.juggler_mode.get() == "gateway"
                roles = build_runtimes(
                    dry_run=True,
                    backend_host="127.0.0.1" if gateway else None,
                    expose_public_ports=not gateway,
                )
                state = JugglerState(
                    roles,
                    auto_tune=self.auto_tune.get(),
                    switch_timeout=parse_int_env("JUGGLE_SWITCH_TIMEOUT_SECONDS", 600),
                    startup_timeout=parse_int_env("JUGGLE_STARTUP_TIMEOUT_SECONDS", 900),
                    request_timeout=parse_int_env("JUGGLE_REQUEST_TIMEOUT_SECONDS", 3600),
                )
                state.validate_files()
                if gateway:
                    check_gateway_port(self.gateway_bind.get(), int(self.gateway_port.get()))
                    self.append_output(f"Gateway check passed on {self.gateway_bind.get()}:{self.gateway_port.get()}\n")
                else:
                    self.append_output("Role proxy juggler check passed\n")
            except Exception as exc:
                messagebox.showerror("Juggler check failed", str(exc))

        def start_juggler(self) -> None:
            if not self.save_config():
                return
            if self.juggler_handle is not None or self.juggler_status.get() in START_PENDING_STATUSES:
                self.append_output("Juggler is already running from this GUI\n")
                return
            self.juggler_status.set(start_status_for_auto_tune(self.auto_tune.get(), ROLES, self.panel_dir))
            threading.Thread(target=self._start_juggler_worker, daemon=True).start()

        def _start_juggler_worker(self) -> None:
            try:
                gateway = self.juggler_mode.get() == "gateway"
                roles = build_runtimes(
                    dry_run=False,
                    backend_host="127.0.0.1" if gateway else None,
                    expose_public_ports=not gateway,
                )
                state = JugglerState(
                    roles,
                    auto_tune=self.auto_tune.get(),
                    switch_timeout=parse_int_env("JUGGLE_SWITCH_TIMEOUT_SECONDS", 600),
                    startup_timeout=parse_int_env("JUGGLE_STARTUP_TIMEOUT_SECONDS", 900),
                    request_timeout=parse_int_env("JUGGLE_REQUEST_TIMEOUT_SECONDS", 3600),
                )
                state.validate_files()
                servers: list[ThreadingHTTPServer] = []
                threads: list[threading.Thread] = []

                if gateway:
                    bind = self.gateway_bind.get()
                    port = int(self.gateway_port.get())
                    check_gateway_port(bind, port)
                    state.start_embed_baseline()
                    server = ThreadingHTTPServer((bind, port), make_gateway_handler(state))
                    servers.append(server)
                    thread = threading.Thread(target=server.serve_forever, name="gateway-proxy", daemon=True)
                    thread.start()
                    threads.append(thread)
                    message = f"Gateway ready at http://{bind}:{port}/v1"
                else:
                    state.start_embed_baseline()
                    for role in ROLES:
                        runtime = roles[role]
                        if runtime.external:
                            continue
                        server = ThreadingHTTPServer((runtime.host, runtime.public_port), make_handler(role, state))
                        servers.append(server)
                        thread = threading.Thread(target=server.serve_forever, name=f"{role}-proxy", daemon=True)
                        thread.start()
                        threads.append(thread)
                    message = "Role proxies ready: " + ", ".join(
                        f"{role}=http://{roles[role].host}:{roles[role].public_port}/v1" for role in ROLES
                    )

                self.queue.put(("juggler_started", (JugglerHandle(state=state, servers=servers, threads=threads), message)))
            except Exception as exc:
                self.queue.put(("juggler_error", str(exc)))

        def stop_juggler(self) -> None:
            if self.juggler_handle is None:
                self.juggler_status.set("Stopped")
                return
            try:
                self.juggler_handle.stop()
                self.append_output("Stopped juggler\n")
            except Exception as exc:
                messagebox.showerror("Stop juggler failed", str(exc))
            finally:
                self.juggler_handle = None
                self.juggler_status.set("Stopped")

        def check_updates(self) -> None:
            if self.update_check_running:
                self.append_output("Update check is already running\n")
                return
            self.update_check_running = True
            self.check_updates_button.configure(state="disabled")
            self.append_output("Checking GitHub releases for updates...\n")
            threading.Thread(target=self._check_updates_worker, daemon=True).start()

        def _check_updates_worker(self) -> None:
            try:
                result = check_for_updates(self.panel_dir)
                self.queue.put(("update_check_result", result))
            except Exception as exc:
                self.queue.put(("update_check_error", str(exc)))

        def _apply_update_worker(self, result: UpdateCheckResult) -> None:
            try:
                install_result = apply_update(result, self.panel_dir)
                self.queue.put(("update_install_result", install_result))
            except Exception as exc:
                self.queue.put(("update_install_error", str(exc)))

        def refresh_status(self) -> None:
            try:
                config = load_config(self.panel_dir, apply_tune=False)
                for role in ROLES:
                    if self.status_vars[role].get() in START_PENDING_STATUSES:
                        self.update_role_controls(role)
                        continue
                    proc = self.role_processes.get(role)
                    if process_running(proc):
                        self.status_vars[role].set("Running")
                        self.update_role_controls(role)
                        continue
                    if proc is not None:
                        self.role_processes.pop(role, None)
                    prefix = ROLE_PREFIX[role]
                    host = config["LLAMA_HOST"]
                    port = int(config[f"{prefix}_PORT"])
                    self.status_vars[role].set("Port active" if port_in_use(host, port) else "Stopped")
                    self.update_role_controls(role)
                if self.juggler_handle is not None:
                    self.juggler_status.set("Running")
            finally:
                self.root.after(2000, self.refresh_status)

        def refresh_logs_once(self) -> None:
            try:
                config = load_config(self.panel_dir, apply_tune=False)
            except Exception as exc:
                for text in self.log_texts.values():
                    self.replace_log_text(text, f"Could not load log configuration: {exc}\n")
                return
            for role, text in self.log_texts.items():
                self.replace_log_text(text, tail_file_text(role_log_path(config, role)))

        def refresh_logs(self) -> None:
            try:
                self.refresh_logs_once()
            finally:
                self.root.after(2000, self.refresh_logs)

        def poll_queue(self) -> None:
            while True:
                try:
                    kind, payload = self.queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "role_started":
                    role, proc, log_path = payload
                    self.role_processes[role] = proc
                    self.status_vars[role].set("Running")
                    self.update_role_controls(role)
                    self.append_output(f"Started {ROLE_LABELS[role]} with log {log_path}\n")
                    self.refresh_logs_once()
                elif kind == "role_error":
                    role, message = payload
                    self.status_vars[role].set("Stopped")
                    self.update_role_controls(role)
                    messagebox.showerror(f"{ROLE_LABELS[role]} start failed", message)
                elif kind == "juggler_started":
                    handle, message = payload
                    self.juggler_handle = handle
                    self.juggler_status.set("Running")
                    self.append_output(f"{message}\n")
                elif kind == "juggler_error":
                    self.juggler_status.set("Stopped")
                    messagebox.showerror("Juggler start failed", str(payload))
                elif kind == "update_check_result":
                    result = payload
                    if isinstance(result, UpdateCheckResult):
                        self.append_output(f"{result.message}\n")
                        if result.update_available:
                            self.append_output(f"Downloading and installing {result.latest.tag_name}...\n")
                            threading.Thread(target=self._apply_update_worker, args=(result,), daemon=True).start()
                            continue
                        else:
                            messagebox.showinfo("Updates", result.message)
                    self.update_check_running = False
                    self.check_updates_button.configure(state="normal")
                elif kind == "update_check_error":
                    self.update_check_running = False
                    self.check_updates_button.configure(state="normal")
                    message = str(payload)
                    self.append_output(f"Update check failed: {message}\n")
                    messagebox.showerror("Update check failed", message)
                elif kind == "update_install_result":
                    self.update_check_running = False
                    self.check_updates_button.configure(state="normal")
                    result = payload
                    if isinstance(result, UpdateInstallResult):
                        self.append_output(f"{result.message}\n")
                        messagebox.showinfo("Update installing", result.message)
                    self.root.after(500, self.exit_for_update)
                elif kind == "update_install_error":
                    self.update_check_running = False
                    self.check_updates_button.configure(state="normal")
                    message = str(payload)
                    self.append_output(f"Update install failed: {message}\n")
                    messagebox.showerror("Update install failed", message)
                elif kind == "api_test_result":
                    title, message, test_type = payload
                    if test_type == "chat":
                        self.chat_test_button.configure(state="normal")
                    elif test_type == "embed":
                        self.embedding_test_button.configure(state="normal")
                    self.append_output(f"{title}:\n{message}\n")
                elif kind == "api_test_error":
                    title, message, test_type = payload
                    if test_type == "chat":
                        self.chat_test_button.configure(state="normal")
                    elif test_type == "embed":
                        self.embedding_test_button.configure(state="normal")
                    self.append_output(f"{title}: {message}\n")
                    messagebox.showerror(title, message)
            self.root.after(300, self.poll_queue)

        def append_output(self, text: str) -> None:
            self.output.insert("end", text)
            self.output.see("end")

        def replace_log_text(self, widget, text: str) -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("end", text)
            widget.see("end")
            widget.configure(state="disabled")

        def update_role_controls(self, role: str) -> None:
            start_button = self.role_start_buttons.get(role)
            stop_button = self.role_stop_buttons.get(role)
            if start_button is None or stop_button is None:
                return
            status = self.status_vars[role].get()
            running = process_running(self.role_processes.get(role))
            start_button.configure(state="disabled" if running or status in START_PENDING_STATUSES else "normal")
            stop_button.configure(state="normal" if running else "disabled")

        def exit_for_update(self) -> None:
            for role in list(self.role_processes):
                self.stop_role(role)
            self.stop_juggler()
            self.root.destroy()

        def on_close(self) -> None:
            running = [role for role, proc in self.role_processes.items() if process_running(proc)]
            if self.juggler_handle is not None:
                running.append("juggler")
            if running and not messagebox.askyesno("Stop running processes", f"Stop {', '.join(running)} and close?"):
                return
            for role in list(self.role_processes):
                self.stop_role(role)
            self.stop_juggler()
            self.root.destroy()

    root = tk.Tk()
    PanelApp(root)
    root.mainloop()
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    _ = argv
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
