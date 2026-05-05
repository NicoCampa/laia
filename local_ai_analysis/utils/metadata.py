from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover - optional at runtime
    psutil = None


@dataclass(frozen=True)
class ModelFileInfo:
    path: str | None
    exists: bool
    file_name: str | None
    size_bytes: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model_file(path: str | None, file_name: str | None = None) -> ModelFileInfo:
    if not path:
        return ModelFileInfo(None, False, file_name, None, None)

    candidate = Path(path).expanduser()
    if candidate.exists() and candidate.is_file():
        return ModelFileInfo(
            path=str(candidate),
            exists=True,
            file_name=file_name or candidate.name,
            size_bytes=candidate.stat().st_size,
            sha256=sha256_file(candidate),
        )
    return ModelFileInfo(str(candidate), False, file_name or candidate.name, None, None)


def command_exists(binary: str) -> bool:
    return shutil.which(binary) is not None or Path(binary).exists()


def run_version_command(binary: str, args: list[str] | None = None) -> str | None:
    if not command_exists(binary):
        return None
    command = [binary, *(args or ["--version"])]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    text = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    return text[:2000] or None


def collect_hardware_metadata() -> dict[str, Any]:
    vm = psutil.virtual_memory() if psutil else None
    metadata: dict[str, Any] = {
        "os_name": platform.system(),
        "os_version": platform.platform(),
        "machine": platform.machine(),
        "python_version": sys.version.split()[0],
        "cpu_model": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_total_bytes": int(vm.total) if vm else None,
        "gpu_name": None,
        "gpu_memory_bytes": None,
        "accelerator": None,
        "extra": {},
    }

    if platform.system() == "Darwin":
        metadata["accelerator"] = "Apple Silicon" if platform.machine() == "arm64" else "CPU"
        metadata["gpu_name"] = _mac_gpu_name()
    elif shutil.which("nvidia-smi"):
        gpu = _nvidia_gpu_info()
        metadata.update(gpu)
        metadata["accelerator"] = "NVIDIA GPU"
    else:
        metadata["accelerator"] = "CPU"

    metadata["hardware_hash"] = stable_hash(
        {
            "os_name": metadata["os_name"],
            "machine": metadata["machine"],
            "cpu_model": metadata["cpu_model"],
            "cpu_count": metadata["cpu_count"],
            "ram_total_bytes": metadata["ram_total_bytes"],
            "gpu_name": metadata["gpu_name"],
            "gpu_memory_bytes": metadata["gpu_memory_bytes"],
            "accelerator": metadata["accelerator"],
        }
    )
    return metadata


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _mac_gpu_name() -> str | None:
    try:
        completed = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None

    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            return stripped.split(":", 1)[1].strip()
    return None


def _nvidia_gpu_info() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return {"gpu_name": None, "gpu_memory_bytes": None}

    first = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    if "," not in first:
        return {"gpu_name": None, "gpu_memory_bytes": None}
    name, memory_mib = [part.strip() for part in first.split(",", 1)]
    try:
        memory_bytes = int(memory_mib) * 1024 * 1024
    except ValueError:
        memory_bytes = None
    return {"gpu_name": name, "gpu_memory_bytes": memory_bytes}
