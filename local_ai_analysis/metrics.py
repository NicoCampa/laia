from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricResult:
    metric_name: str
    metric_value: float | None
    unit: str | None
    raw: dict[str, Any]
