"""
sandbox.py - one Docker container per agent, driven from Windows via WSL.

Isolation model:
  - container runs as uid 1000 (non-root), --cap-drop ALL, no-new-privileges
  - CPU / memory / pids limits, every command wrapped in `timeout`
  - the ONLY host path it sees is its own workspace folder (runs/<id>/<agent>/),
    plus optional read-only mounts (the verifier reads solvers' workspaces)
  - no API keys or project files inside; container is deleted afterwards
  - full internet (by design for this lab: data downloads, pip installs)
Every command is appended to <workspace>/../commands.log for auditing.
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
from pathlib import Path, PureWindowsPath

IMAGE = "agent-lab-sandbox:latest"
DISTRO = "Ubuntu-24.04"
MAX_OUTPUT = 6000  # chars returned to the model per command (head + tail)

_log_lock = threading.Lock()


def to_wsl_path(p: Path) -> str:
    """C:\\Users\\x\\proj -> /mnt/c/Users/x/proj"""
    pw = PureWindowsPath(p.resolve())
    return f"/mnt/{pw.drive[0].lower()}/" + "/".join(pw.parts[1:])


def wsl(*args: str, input: str | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["wsl", "-d", DISTRO, "--exec", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", input=input, timeout=timeout,
    )


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    """Keep the head and tail of long output; the middle is rarely what the model needs."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} chars omitted] ...\n{text[-half:]}"


class Sandbox:
    def __init__(self, workspace: Path, name: str, cpus: float = 4, memory: str = "3g",
                 readonly_mounts: dict[Path, str] | None = None):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.name = f"agentlab-{name}-{uuid.uuid4().hex[:6]}"
        self.cpus, self.memory = cpus, memory
        self.readonly_mounts = readonly_mounts or {}
        self.audit = workspace.parent / "commands.log"

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "Sandbox":
        args = [
            "docker", "run", "-d", "--name", self.name,
            "--cpus", str(self.cpus), "--memory", self.memory, "--pids-limit", "512",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "-v", f"{to_wsl_path(self.workspace)}:/workspace",
        ]
        for host, target in self.readonly_mounts.items():
            args += ["-v", f"{to_wsl_path(host)}:{target}:ro"]
        r = wsl(*args, IMAGE, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr.strip()}")
        return self

    def stop(self) -> None:
        wsl("docker", "rm", "-f", self.name, timeout=60)

    def __enter__(self) -> "Sandbox":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- execution ---------------------------------------------------------------
    def exec(self, command: str, timeout_s: int = 120) -> dict:
        """Run a bash command in the container. Returns exit code, clipped output, duration."""
        t0 = time.perf_counter()
        try:
            r = wsl("docker", "exec", "-i", self.name, "timeout", "--kill-after=5", str(timeout_s), "bash", "-lc", command,
                    timeout=timeout_s + 30)
            code, out = r.returncode, (r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else ""))
        except subprocess.TimeoutExpired:
            code, out = 124, "host-side timeout"
        if code == 124:
            out += f"\n[killed: exceeded {timeout_s}s timeout]"
        seconds = round(time.perf_counter() - t0, 1)
        with _log_lock, self.audit.open("a", encoding="utf-8") as f:
            f.write(f"--- {time.strftime('%H:%M:%S')} {self.workspace.name} exit={code} {seconds}s\n$ {command[:2000]}\n")
        return {"exit_code": code, "output": clip(out.strip()), "seconds": seconds}

    # Files are exchanged through the bind-mounted workspace on the host side.
    def host_path(self, rel: str) -> Path:
        p = (self.workspace / rel.lstrip("/").removeprefix("workspace/")).resolve()
        if not p.is_relative_to(self.workspace.resolve()):
            raise ValueError(f"path escapes the workspace: {rel}")
        return p
