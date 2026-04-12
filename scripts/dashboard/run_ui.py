from __future__ import annotations

import argparse
import contextlib
import socket
import shutil
import subprocess
import sys
import time
import webbrowser

from cos435_citylearn.paths import REPO_ROOT


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    backend_command = [sys.executable, "scripts/dashboard/run_backend.py", "--reload"]
    frontend_command = ["npm", "run", "dev"]
    frontend_root = REPO_ROOT / "apps" / "dashboard"

    if not (frontend_root / "package.json").exists():
        raise FileNotFoundError("apps/dashboard/package.json is missing")
    if shutil.which("npm") is None:
        raise RuntimeError("npm is required for the dashboard frontend")
    if not (frontend_root / "node_modules").exists():
        raise RuntimeError("dashboard dependencies are missing; run `make dashboard-install`")

    backend = subprocess.Popen(backend_command, cwd=REPO_ROOT)
    frontend = None

    try:
        frontend = subprocess.Popen(frontend_command, cwd=frontend_root)
    except Exception:
        backend.terminate()
        backend.wait(timeout=10)
        raise

    if args.open_browser:
        frontend_ready = _wait_for_port("127.0.0.1", 5173)
        backend_ready = _wait_for_port("127.0.0.1", 8001)
        if frontend_ready and backend_ready:
            webbrowser.open("http://127.0.0.1:5173/dashboard/")

    try:
        backend.wait()
    finally:
        if frontend is not None:
            frontend.terminate()
        backend.terminate()


if __name__ == "__main__":
    main()
