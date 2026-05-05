from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from local_ai_analysis import __version__
from local_ai_analysis.db import LocalAIAnalysisDB
from local_ai_analysis.export import leaderboard_payload


def create_app(db_path: str | Path = "results/local_ai_analysis.duckdb") -> FastAPI:
    app = FastAPI(
        title="Local AI Analysis API",
        version=__version__,
        description="Reproducible Global MMLU Lite benchmark API for local AI servers.",
    )
    app.state.db_path = str(db_path)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/leaderboard")
    def leaderboard(
        family: str | None = None,
        quantization: str | None = None,
        backend: str | None = None,
        hardware: str | None = None,
        min_quality: float | None = Query(default=None, ge=0, le=1),
        max_runtime_seconds: float | None = Query(default=None, ge=0),
    ) -> dict[str, Any]:
        payload = leaderboard_payload(app.state.db_path)
        rows = payload["leaderboard"]
        rows = [
            row
            for row in rows
            if _matches(row, "family", family)
            and _matches(row, "quantization", quantization)
            and _matches(row, "backend_name", backend)
            and _matches(row, "hardware_accelerator", hardware)
            and (
                min_quality is None
                or (
                    row.get("global_mmlu_lite_pass_at_1") is not None
                    and row["global_mmlu_lite_pass_at_1"] >= min_quality
                )
            )
            and (
                max_runtime_seconds is None
                or (
                    row.get("benchmark_runtime_seconds") is not None
                    and row["benchmark_runtime_seconds"] <= max_runtime_seconds
                )
            )
        ]
        payload["leaderboard"] = rows
        return payload

    @app.get("/api/variants/{variant_id}/raw")
    def raw_variant(variant_id: str) -> dict[str, Any]:
        db = LocalAIAnalysisDB(app.state.db_path)
        try:
            db.init_schema()
            payload = db.raw_variant_metadata(variant_id)
        finally:
            db.close()
        if payload is None:
            raise HTTPException(status_code=404, detail="Variant not found")
        return payload

    return app


def _matches(row: dict[str, Any], key: str, expected: str | None) -> bool:
    return expected is None or str(row.get(key)) == expected


app = create_app()
