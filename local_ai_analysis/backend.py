from __future__ import annotations

from typing import Any

from local_ai_analysis.config import BackendSettings


def backend_profile_payload(settings: BackendSettings) -> dict[str, Any]:
    return {
        "backend_name": settings.name,
        "backend_type": settings.backend_type,
        "backend_version": settings.version,
        "backend_commit": settings.commit,
        "command": settings.command,
        "extra": settings.model_extra or {},
    }
