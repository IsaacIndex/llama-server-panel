#!/usr/bin/env python3
"""Lightweight cross-platform GUI for managing local llama-server models."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
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


def process_running(proc: Optional[subprocess.Popen[bytes]]) -> bool:
    return proc is not None and proc.poll() is None


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
            self.juggler_handle: Optional[JugglerHandle] = None
            self.queue: queue.Queue[tuple[str, object]] = queue.Queue()

            self.root.title("Llama Server Panel")
            self.root.geometry("1180x760")
            self.root.minsize(980, 640)

            style = ttk.Style()
            if "clam" in style.theme_names():
                style.theme_use("clam")
            style.configure("Accent.TButton", padding=(12, 6))
            style.configure("Danger.TButton", padding=(12, 6))
            style.configure("Status.TLabel", font=("TkDefaultFont", 9, "bold"))

            self.auto_tune = tk.BooleanVar(value=True)
            self.juggler_mode = tk.StringVar(value="gateway")
            self.gateway_bind = tk.StringVar(value=GATEWAY_DEFAULT_BIND)
            self.gateway_port = tk.StringVar(value=str(GATEWAY_DEFAULT_PORT))
            self.selected_assign_key = tk.StringVar(value="CHAT_MODEL")
            self.status_vars = {role: tk.StringVar(value="Stopped") for role in ROLES}
            self.juggler_status = tk.StringVar(value="Stopped")

            self._build_layout(ttk, tk, filedialog, messagebox)
            self.reload_config()
            self.refresh_model_list()
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.root.after(400, self.poll_queue)
            self.root.after(1500, self.refresh_status)

        def _build_layout(self, ttk, tk, filedialog, messagebox) -> None:
            main = ttk.Frame(self.root, padding=12)
            main.pack(fill=tk.BOTH, expand=True)
            main.columnconfigure(0, weight=1, uniform="main")
            main.columnconfigure(1, weight=2, uniform="main")
            main.rowconfigure(2, weight=1)

            header = ttk.Frame(main)
            header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            header.columnconfigure(1, weight=1)
            ttk.Label(header, text="Llama Server Panel", font=("TkDefaultFont", 18, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Button(header, text="Save Config", command=self.save_config, style="Accent.TButton").grid(row=0, column=2, padx=(8, 0))
            ttk.Button(header, text="Reload", command=self.reload_config).grid(row=0, column=3, padx=(8, 0))

            general = ttk.LabelFrame(main, text="Paths", padding=10)
            general.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            general.columnconfigure(1, weight=1)
            self._entry_row(general, ttk, "llama-server", "LLAMA_SERVER_BIN", 0, browse_file=True)
            self._entry_row(general, ttk, "Model dir", "MODEL_DIR", 1, browse_dir=True)
            self._entry_row(general, ttk, "Log dir", "LOG_DIR", 2, browse_dir=True)

            library = ttk.LabelFrame(main, text="Model Library", padding=10)
            library.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
            library.columnconfigure(0, weight=1)
            library.rowconfigure(0, weight=1)
            self.model_list = tk.Listbox(library, activestyle="dotbox", exportselection=False)
            self.model_list.grid(row=0, column=0, sticky="nsew")
            model_scroll = ttk.Scrollbar(library, orient=tk.VERTICAL, command=self.model_list.yview)
            model_scroll.grid(row=0, column=1, sticky="ns")
            self.model_list.configure(yscrollcommand=model_scroll.set)

            library_actions = ttk.Frame(library)
            library_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            library_actions.columnconfigure(1, weight=1)
            ttk.Button(library_actions, text="Import GGUF", command=self.import_model).grid(row=0, column=0, sticky="w")
            ttk.Combobox(
                library_actions,
                textvariable=self.selected_assign_key,
                values=("CHAT_MODEL", "EMBED_MODEL", "VISION_MODEL", "VISION_MMPROJ"),
                state="readonly",
                width=18,
            ).grid(row=0, column=1, sticky="ew", padx=8)
            ttk.Button(library_actions, text="Assign", command=self.assign_selected_model).grid(row=0, column=2)
            ttk.Button(library_actions, text="Refresh", command=self.refresh_model_list).grid(row=0, column=3, padx=(8, 0))

            right = ttk.Frame(main)
            right.grid(row=2, column=1, sticky="nsew")
            right.columnconfigure(0, weight=1)
            right.rowconfigure(0, weight=1)

            role_frame = ttk.LabelFrame(right, text="Roles", padding=10)
            role_frame.grid(row=0, column=0, sticky="nsew")
            role_frame.columnconfigure(0, weight=1)
            role_frame.rowconfigure(0, weight=1)

            role_tabs = ttk.Notebook(role_frame)
            role_tabs.grid(row=0, column=0, sticky="nsew")
            for role in ROLES:
                role_page = ttk.Frame(role_tabs, padding=4)
                role_page.columnconfigure(1, weight=1)
                role_tabs.add(role_page, text=ROLE_LABELS[role])
                self._role_block(role_page, ttk, role, 0)

            juggler = ttk.LabelFrame(right, text="Juggler", padding=10)
            juggler.grid(row=1, column=0, sticky="ew", pady=(10, 0))
            juggler.columnconfigure(3, weight=1)
            ttk.Label(juggler, text="Mode").grid(row=0, column=0, sticky="w")
            ttk.Combobox(
                juggler,
                textvariable=self.juggler_mode,
                values=("gateway", "role proxies"),
                state="readonly",
                width=14,
            ).grid(row=0, column=1, sticky="w", padx=(8, 16))
            ttk.Label(juggler, text="Bind").grid(row=0, column=2, sticky="w")
            ttk.Entry(juggler, textvariable=self.gateway_bind, width=14).grid(row=0, column=3, sticky="ew", padx=(8, 16))
            ttk.Label(juggler, text="Port").grid(row=0, column=4, sticky="w")
            ttk.Entry(juggler, textvariable=self.gateway_port, width=8).grid(row=0, column=5, sticky="w", padx=(8, 16))
            ttk.Checkbutton(juggler, text="Auto-tune", variable=self.auto_tune).grid(row=0, column=6, sticky="w")
            ttk.Label(juggler, textvariable=self.juggler_status, style="Status.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
            ttk.Button(juggler, text="Start", command=self.start_juggler, style="Accent.TButton").grid(row=1, column=4, sticky="e", pady=(10, 0))
            ttk.Button(juggler, text="Stop", command=self.stop_juggler).grid(row=1, column=5, sticky="e", pady=(10, 0))
            ttk.Button(juggler, text="Check", command=self.check_juggler).grid(row=1, column=6, sticky="e", pady=(10, 0))

            output_frame = ttk.LabelFrame(main, text="Output", padding=10)
            output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
            output_frame.columnconfigure(0, weight=1)
            output_frame.rowconfigure(0, weight=1)
            self.output = tk.Text(output_frame, height=9, wrap="word", bg="#101418", fg="#d8e3e7", insertbackground="#d8e3e7")
            self.output.grid(row=0, column=0, sticky="nsew")
            output_scroll = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
            output_scroll.grid(row=0, column=1, sticky="ns")
            self.output.configure(yscrollcommand=output_scroll.set)

        def _var(self, key: str) -> "tk.StringVar":
            if key not in self.values:
                import tkinter as tk

                self.values[key] = tk.StringVar()
            return self.values[key]

        def _entry_row(self, parent, ttk, label: str, key: str, row: int, *, browse_file: bool = False, browse_dir: bool = False) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(parent, textvariable=self._var(key)).grid(row=row, column=1, sticky="ew", padx=8, pady=3)
            if browse_file:
                ttk.Button(parent, text="Browse", command=lambda: self.browse_file(key)).grid(row=row, column=2, sticky="e", pady=3)
            elif browse_dir:
                ttk.Button(parent, text="Browse", command=lambda: self.browse_dir(key)).grid(row=row, column=2, sticky="e", pady=3)

        def _role_block(self, parent, ttk, role: str, start_row: int) -> int:
            prefix = ROLE_PREFIX[role]
            ttk.Separator(parent).grid(row=start_row, column=0, columnspan=5, sticky="ew", pady=(6, 8))
            ttk.Label(parent, text=ROLE_LABELS[role], font=("TkDefaultFont", 11, "bold")).grid(row=start_row + 1, column=0, sticky="w")
            ttk.Label(parent, textvariable=self.status_vars[role], style="Status.TLabel").grid(row=start_row + 1, column=1, sticky="w")
            ttk.Button(parent, text="Start", command=lambda r=role: self.start_role(r), style="Accent.TButton").grid(row=start_row + 1, column=2, padx=(8, 0))
            ttk.Button(parent, text="Stop", command=lambda r=role: self.stop_role(r)).grid(row=start_row + 1, column=3, padx=(8, 0))
            ttk.Button(parent, text="Check", command=lambda r=role: self.check_role(r)).grid(row=start_row + 1, column=4, padx=(8, 0))

            row = start_row + 2
            self._entry_row(parent, ttk, "Model", f"{prefix}_MODEL", row, browse_file=True)
            row += 1
            if role == "vision":
                self._entry_row(parent, ttk, "MMProj", "VISION_MMPROJ", row, browse_file=True)
                row += 1
            if role in {"chat", "vision"}:
                self._entry_row(parent, ttk, "Alias", f"{prefix}_ALIAS", row)
                row += 1

            compact = ttk.Frame(parent)
            compact.grid(row=row, column=0, columnspan=5, sticky="ew", pady=(0, 6))
            for idx, key in enumerate((f"{prefix}_PORT", f"{prefix}_CTX_SIZE", f"{prefix}_THREADS")):
                ttk.Label(compact, text=key.replace(f"{prefix}_", "").title()).grid(row=0, column=idx * 2, sticky="w", padx=(0 if idx == 0 else 12, 4))
                ttk.Entry(compact, textvariable=self._var(key), width=10).grid(row=0, column=idx * 2 + 1, sticky="w")
            return row + 1

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
            if not self.save_config():
                return
            if process_running(self.role_processes.get(role)):
                self.append_output(f"{ROLE_LABELS[role]} is already running from this GUI\n")
                return
            self.status_vars[role].set("Starting")
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
                log_dir = Path(config["LOG_DIR"])
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{role}-gui.log"
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
            proc = self.role_processes.get(role)
            if not process_running(proc):
                self.status_vars[role].set("Stopped")
                return
            try:
                terminate_process(proc)
                self.append_output(f"Stopped {ROLE_LABELS[role]}\n")
            except Exception as exc:
                messagebox.showerror(f"Stop {ROLE_LABELS[role]} failed", str(exc))
            self.status_vars[role].set("Stopped")

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
            if self.juggler_handle is not None:
                self.append_output("Juggler is already running from this GUI\n")
                return
            self.juggler_status.set("Starting")
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

        def refresh_status(self) -> None:
            try:
                config = load_config(self.panel_dir, apply_tune=False)
                for role in ROLES:
                    proc = self.role_processes.get(role)
                    if process_running(proc):
                        self.status_vars[role].set("Running")
                        continue
                    prefix = ROLE_PREFIX[role]
                    host = config["LLAMA_HOST"]
                    port = int(config[f"{prefix}_PORT"])
                    self.status_vars[role].set("Port active" if port_in_use(host, port) else "Stopped")
                if self.juggler_handle is not None:
                    self.juggler_status.set("Running")
            finally:
                self.root.after(2000, self.refresh_status)

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
                    self.append_output(f"Started {ROLE_LABELS[role]} with log {log_path}\n")
                elif kind == "role_error":
                    role, message = payload
                    self.status_vars[role].set("Stopped")
                    messagebox.showerror(f"{ROLE_LABELS[role]} start failed", message)
                elif kind == "juggler_started":
                    handle, message = payload
                    self.juggler_handle = handle
                    self.juggler_status.set("Running")
                    self.append_output(f"{message}\n")
                elif kind == "juggler_error":
                    self.juggler_status.set("Stopped")
                    messagebox.showerror("Juggler start failed", str(payload))
            self.root.after(300, self.poll_queue)

        def append_output(self, text: str) -> None:
            self.output.insert("end", text)
            self.output.see("end")

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
