from __future__ import annotations
from typing import Any, Callable


ProgressCallback = Callable[[str, dict[str, Any]], None]


def maybe_reset_runtime(
    *,
    client: Any,
    settings: Any,
    model: str,
    task: str,
    variant_name: str,
    completed_samples: int,
    total_samples: int,
    progress_callback: ProgressCallback | None,
    language: str | None = None,
    reason: str = "call_interval",
    allow_at_end: bool = False,
) -> None:
    restart_every_calls = getattr(settings, "restart_every_calls", None)
    if (
        restart_every_calls is None
        or restart_every_calls <= 0
        or completed_samples <= 0
        or (completed_samples >= total_samples and not allow_at_end)
        or completed_samples % restart_every_calls != 0
    ):
        return
    reset_runtime(
        client=client,
        settings=settings,
        model=model,
        task=task,
        variant_name=variant_name,
        completed_samples=completed_samples,
        progress_callback=progress_callback,
        language=language,
        reason=reason,
    )


def reset_runtime(
    *,
    client: Any,
    settings: Any,
    model: str,
    task: str,
    variant_name: str,
    completed_samples: int,
    progress_callback: ProgressCallback | None,
    language: str | None = None,
    reason: str,
) -> None:
    reset_error = None
    cooldown_seconds = max(0.0, float(getattr(settings, "restart_cooldown_seconds", 0.0) or 0.0))
    try:
        instance_id = client.reset_model_runtime(
            model,
            request_extra=getattr(settings, "request_extra", None),
            cooldown_seconds=cooldown_seconds,
        )
    except Exception as exc:
        instance_id = None
        reset_error = str(exc)

    if progress_callback:
        progress_callback(
            "runtime_cache_reset",
            {
                "task": task,
                "variant": variant_name,
                "language": language,
                "provider": getattr(settings, "provider", None),
                "model": model,
                "instance_id": instance_id,
                "completed_samples": completed_samples,
                "reason": reason,
                "cooldown_seconds": cooldown_seconds,
                "error": reset_error,
            },
        )
