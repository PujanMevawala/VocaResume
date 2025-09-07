"""Development auto-reload runner for VocaResume.

This script mimics nodemon-like behavior for Streamlit using watchdog.
It watches source directories for file changes and restarts the Streamlit
process automatically.

Usage:
  1. Install requirements (includes watchdog)
  2. Run: python run_dev.py
  3. Open the printed Streamlit URL in your browser.

Configuration:
  - By default watches: current directory excluding .venv, __pycache__, .git
  - You can modify WATCH_PATTERNS for custom glob patterns.

Press Ctrl+C to stop.
"""
from __future__ import annotations
import subprocess
import sys
import time
import os
from threading import Event, Thread
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import PatternMatchingEventHandler
except ImportError:
    print("watchdog not installed. Run: pip install watchdog")
    sys.exit(1)

APP_ENTRY = "app.py"
OPEN_BROWSER_DEFAULT = False  # change with --open flag
WATCH_PATTERNS = ["*.py", "*.css", "*.toml"]
IGNORE_DIRS = {".git", "__pycache__", ".venv", ".mypy_cache"}
DEBOUNCE_SECONDS = 0.5

class RestartHandler(PatternMatchingEventHandler):
    def __init__(self, trigger_cb, *args, **kwargs):
        super().__init__(patterns=WATCH_PATTERNS, ignore_directories=False, case_sensitive=False)
        self.trigger_cb = trigger_cb
        self._last = 0.0

    def on_any_event(self, event):  # type: ignore
        if any(part in IGNORE_DIRS for part in Path(event.src_path).parts):
            return
        now = time.time()
        if now - self._last < DEBOUNCE_SECONDS:
            return
        self._last = now
        print(f"[watch] Change detected: {event.src_path}")
        self.trigger_cb()

class StreamlitRunner:
    def __init__(self, app_entry: str):
        self.app_entry = app_entry
        self.proc: subprocess.Popen | None = None
        self.restart_event = Event()
        self.stop_event = Event()

    def start(self):
        print("[runner] Starting Streamlit (headless, no auto-open)...")
        env = os.environ.copy()
        # Prevent Streamlit from opening a new browser tab each restart
        env["BROWSER"] = "none"
        cmd = [
            sys.executable, "-m", "streamlit", "run", self.app_entry,
            "--server.headless", "true"
        ]
        if not OPEN_BROWSER_DEFAULT:
            # Streamlit respects BROWSER=none; just ensure verbose note
            pass
        self.proc = subprocess.Popen(cmd, env=env)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            print("[runner] Stopping Streamlit...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def restart_loop(self):
        while not self.stop_event.is_set():
            self.restart_event.wait()
            if self.stop_event.is_set():
                break
            self.restart_event.clear()
            self.stop()
            self.start()

    def trigger_restart(self):
        print("[runner] Scheduling restart...")
        self.restart_event.set()

    def shutdown(self):
        self.stop_event.set()
        self.restart_event.set()
        self.stop()


def main():
    global OPEN_BROWSER_DEFAULT
    if '--open' in sys.argv:
        OPEN_BROWSER_DEFAULT = True
        # If user wants initial open, we allow system browser. Remove BROWSER override after first start.
        print('[dev] Browser auto-open enabled for first launch.')
    if not Path(APP_ENTRY).exists():
        print(f"Entry file {APP_ENTRY} not found.")
        sys.exit(1)

    runner = StreamlitRunner(APP_ENTRY)
    runner.start()

    observer = Observer()
    handler = RestartHandler(runner.trigger_restart)
    observer.schedule(handler, path=".", recursive=True)
    observer.start()

    t = Thread(target=runner.restart_loop, daemon=True)
    t.start()

    print("[dev] Watching for changes. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[dev] Shutting down...")
    finally:
        runner.shutdown()
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
