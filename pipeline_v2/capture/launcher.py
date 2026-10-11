"""Small local control window for hands-free Fortnite evaluation collection."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import tkinter as tk
from tkinter import messagebox

from pipeline_v2.capture.collection import CollectionConfig, Collector
from pipeline_v2.jsonio import write_json
from pipeline_v2.config import REPO_ROOT


def publish_status(path: Path, snapshot: dict) -> bool:
    """A diagnostic snapshot must never interrupt the control window's refresh."""
    try:
        write_json(path, snapshot)
    except OSError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically collect Fortnite evaluation footage"
    )
    parser.add_argument(
        "--out", type=Path, default=Path.home() / "Videos" / "Salience" / "Sessions"
    )
    parser.add_argument(
        "--test-capture",
        action="store_true",
        help="Explicitly run the recorder for a controlled performance test",
    )
    args = parser.parse_args()
    if os.name != "nt":
        raise SystemExit("Desktop Duplication requires Windows")
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except AttributeError:
        pass
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    mutex = kernel.CreateMutexW(None, False, "Local\\SalienceEvaluationRecorder")
    duplicate = ctypes.get_last_error() == 183
    root = tk.Tk()
    root.title("Salience — recording setup")
    root.geometry("610x340")
    root.resizable(False, False)
    root.configure(bg="#10151d")
    if duplicate:
        root.withdraw()
        messagebox.showinfo(
            "Salience",
            "The recording control is already open. Use that window to stop recording.",
        )
        root.destroy()
        kernel.CloseHandle(mutex)
        return 0
    if not mutex:
        root.destroy()
        raise OSError("Could not acquire the recorder lock")
    if not args.test_capture:
        root.withdraw()
        messagebox.showwarning(
            "Recording paused after gameplay problems",
            "This capture setup caused FPS drops and game crashes during the September 7 session. "
            "Normal recording is paused until a controlled performance test passes. "
            "Your saved footage is available in Videos\\Salience\\Sessions.",
        )
        root.destroy()
        kernel.CloseHandle(mutex)
        return 0
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        root.withdraw()
        messagebox.showerror(
            "Salience", "FFmpeg/FFprobe is missing from PATH. Recording did not start."
        )
        root.destroy()
        kernel.CloseHandle(mutex)
        return 1

    status = tk.StringVar(value="Preparing recorder…")
    detail = tk.StringVar(value="")
    counts = tk.StringVar(value="")
    tk.Label(
        root,
        text="SALIENCE  /  EVALUATION RECORDING",
        bg="#10151d",
        fg="#91a0b4",
        font=("Segoe UI", 11),
    ).pack(anchor="w", padx=26, pady=(24, 12))
    title = tk.Label(
        root,
        textvariable=status,
        bg="#10151d",
        fg="#85b8ff",
        font=("Segoe UI", 19, "bold"),
        wraplength=555,
        justify="left",
    )
    title.pack(anchor="w", padx=26)
    tk.Label(
        root,
        textvariable=detail,
        bg="#10151d",
        fg="#e0e6ed",
        font=("Segoe UI", 10),
        wraplength=550,
        justify="left",
    ).pack(anchor="w", padx=26, pady=(12, 0))
    tk.Label(
        root,
        text="Play normally. No save keys or rating prompts.\nVideo only · no uploads · 2 hours / 15 GB maximum",
        bg="#10151d",
        fg="#91a0b4",
        font=("Segoe UI", 10),
        justify="left",
    ).pack(anchor="w", padx=26, pady=12)
    tk.Label(
        root, textvariable=counts, bg="#10151d", fg="#e0e6ed", font=("Segoe UI", 10)
    ).pack(anchor="w", padx=26)
    buttons = tk.Frame(root, bg="#10151d")
    buttons.pack(anchor="w", padx=26, pady=20)
    local = REPO_ROOT / ".local-data"
    local.mkdir(exist_ok=True)
    collector = Collector(CollectionConfig(args.out))
    profiles = local / "capture-settings.json"
    if profiles.exists():
        try:
            for profile in json.loads(profiles.read_text())["verified_encoders"]:
                if profile["encoder"] in ("h264_qsv", "h264_nvenc"):
                    collector.encoder_for[
                        (int(profile["adapter"]), int(profile["output"]))
                    ] = profile["encoder"]
        except (ValueError, KeyError, TypeError):
            pass
    updates = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=collector.watch,
        args=(stop_event, updates.put),
        name="Salience collection",
    )
    closing = False

    def stop():
        status.set("Finishing current chunk…")
        stop_button.config(state="disabled")
        stop_event.set()

    def close():
        nonlocal closing
        closing = True
        stop()

    stop_button = tk.Button(
        buttons, text="Stop recording", command=stop, padx=14, pady=5
    )
    stop_button.pack(side="left")
    tk.Button(
        buttons,
        text="Open recordings",
        command=lambda: os.startfile(collector.directory),
        padx=14,
        pady=5,
    ).pack(side="left", padx=12)
    root.protocol("WM_DELETE_WINDOW", close)

    def refresh():
        latest = None
        while True:
            try:
                latest = updates.get_nowait()
            except queue.Empty:
                break
        if latest:
            if not stop_event.is_set() or latest["done"]:
                status.set(latest["status"])
            detail.set(latest["detail"])
            counts.set(
                f"{latest['clips']} saved chunks  ·  {latest['recorded_seconds'] / 60:.1f} recorded minutes  ·  {latest['bytes'] / 1024**3:.2f} GB"
            )
            title.config(
                fg="#ff8080" if latest["status"] == "Recording Fortnite" else "#85b8ff"
            )
            if latest["done"]:
                stop_button.config(state="disabled")
            publish_status(
                local / "capture-status.json", {"pid": os.getpid(), **latest}
            )
        if closing and not worker.is_alive():
            root.destroy()
            return
        root.after(250, refresh)

    try:
        worker.start()
        refresh()
        root.mainloop()
    finally:
        stop_event.set()
        worker.join(timeout=15)
        kernel.CloseHandle(mutex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
