#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

if sys.platform == "darwin":
    # Ensure Homebrew's library path is searched for libcairo
    dyld_path = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if "/opt/homebrew/lib" not in dyld_path.split(":"):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{dyld_path}:{'/opt/homebrew/lib'}" if dyld_path else "/opt/homebrew/lib"

try:
    import cairosvg
except Exception:
    cairosvg = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "public" / "results.json"
WORLD_MAP_PATH = ROOT / "web" / "src" / "worldMapPaths.ts"
OUTPUT_DIR = ROOT / "docs" / "images" / "linkedin"
LAIA_LOGO_PATH = ROOT / "web" / "public" / "laia_primary_logo_transparent.png"

BG = "#ffffff"
INK = "#111111"
TEXT = "#353535"
MUTED = "#6b6b6b"
LINE = "#d8d8d8"
SOFT = "#f4f4f4"

LAIA_KEYS = [
    ("Knowledge", "global_mmlu_lite_pass_at_1"),
    ("Instructions", "ifbench_prompt_level_loose"),
    ("Tools", "bfcl_v4_selected_accuracy"),
    ("Coding", "mbpp_pass_at_1"),
    ("Grounding", "rgb_all_rate"),
]

KNOWN_MODEL_FILE_SIZES = {
    "lfm2.5-350m@bf16": 711_500_000,
    "lfm2.5-350m@q8_0": 379_200_000,
    "lfm2.5-350m@q4_k_m": 229_300_000,
    "nemotron-3-nano:4b": 3_006_477_107,
    "qwen3.5-0.8b@bf16": 1_700_000_000,
    "qwen3.5-0.8b@q8_0": 1_200_000_000,
    "qwen3.5-0.8b@q4_k_m": 934_900_000,
}

PROVIDERS = {
    "openai": {"label": "OpenAI", "color": "#111111"},
    "alibaba": {"label": "Alibaba", "color": "#6d4cff"},
    "google": {"label": "Google", "color": "#63b35d"},
    "mistral": {"label": "Mistral AI", "color": "#ff8a2a"},
    "meta": {"label": "Meta", "color": "#4f8dff"},
    "nvidia": {"label": "NVIDIA", "color": "#76b900"},
    "liquid": {"label": "Liquid AI", "color": "#f2c94c"},
    "ibm": {"label": "IBM", "color": "#6f84a3"},
    "ai2": {"label": "AI2", "color": "#b06df5"},
    "tii": {"label": "TII", "color": "#e56f52"},
    "microsoft": {"label": "Microsoft", "color": "#2f80ed"},
    "huggingface": {"label": "Hugging Face", "color": "#b88700"},
    "local": {"label": "Local", "color": "#7a7a74"},
}

ORIGINS = [
    {"id": "ai2", "city": "Seattle", "country": "United States", "lat": 47.6062, "lon": -122.3321},
    {"id": "microsoft", "city": "Redmond", "country": "United States", "lat": 47.674, "lon": -122.1215},
    {"id": "nvidia", "city": "Santa Clara", "country": "United States", "lat": 37.3541, "lon": -121.9552},
    {"id": "meta", "city": "Menlo Park", "country": "United States", "lat": 37.453, "lon": -122.1817},
    {"id": "google", "city": "Mountain View", "country": "United States", "lat": 37.3861, "lon": -122.0839},
    {"id": "liquid", "city": "Cambridge, MA", "country": "United States", "lat": 42.3736, "lon": -71.1097},
    {"id": "ibm", "city": "Armonk", "country": "United States", "lat": 41.1265, "lon": -73.714},
    {"id": "huggingface", "city": "New York City", "country": "United States", "lat": 40.7128, "lon": -74.006},
    {"id": "mistral", "city": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
    {"id": "tii", "city": "Abu Dhabi", "country": "United Arab Emirates", "lat": 24.4539, "lon": 54.3773},
    {"id": "alibaba", "city": "Hangzhou", "country": "China", "lat": 30.2741, "lon": 120.1551},
]


@dataclass
class PlotRow:
    raw: dict[str, Any]

    @property
    def provider_key(self) -> str:
        return provider_key(self.raw)

    @property
    def provider(self) -> dict[str, str]:
        return PROVIDERS.get(self.provider_key, PROVIDERS["local"])

    @property
    def score(self) -> float:
        return score(self.raw)

    @property
    def name(self) -> str:
        return short_model_name(self.raw)

    @property
    def size_gb(self) -> float | None:
        return model_size_gb(self.raw)


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("metadata_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def api_model(row: dict[str, Any]) -> str | None:
    meta = metadata(row)
    return row.get("api_model") or meta.get("variant_config", {}).get("api_model") or meta.get("api_model")


def title_case(value: str) -> str:
    words = []
    for part in str(value or "").split():
        if re.match(r"^\d+(?:\.\d+)?b$", part, re.I):
            words.append(part.upper())
        else:
            words.append(part[:1].upper() + part[1:])
    return " ".join(words)


def format_billion(value: float) -> str:
    if value < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if value < 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def format_model_name(value: str | None) -> str:
    last = str(value or "Model").split("/")[-1]
    no_quant = last.split("@")[0]
    if re.search(r"nemotron[-_\s]*3[-_\s]*nano", last, re.I):
        return "Nemotron 3 Nano 4B"

    lfm_match = re.search(r"\blfm\s*(\d+(?:\.\d+)?)\s*[-_\s]*(\d+(?:\.\d+)?)([bm])\b", no_quant, re.I)
    if lfm_match:
        size = float(lfm_match.group(2)) / (1000 if lfm_match.group(3).lower() == "m" else 1)
        return f"LFM {lfm_match.group(1)} {format_billion(size)}B"

    clean = re.sub(
        r"@(?:q\d+(?:[_-][a-z0-9]+)*|bf16|fp16|fp32|f16|f32|4bit|8bit|16bit)\b",
        " ",
        last,
        flags=re.I,
    )
    clean = re.sub(
        r"\b(?:ollama|lm studio|omlx|all languages|smoke|reasoning|none|server|mlx|gguf|bf16|fp16|"
        r"q\d+(?:[_-][a-z0-9]+)*|it|instruct|chat)\b",
        " ",
        clean,
        flags=re.I,
    )
    clean = re.sub(r"[:_-]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    qwen_match = re.search(r"\bqwen\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b", clean, re.I)
    if qwen_match:
        return f"Qwen {qwen_match.group(1)} {qwen_match.group(2)}B"

    gemma_match = re.search(r"\bgemma\s*(\d+(?:\.\d+)?)\s*e\s*(\d+(?:\.\d+)?)\s*b\b", clean, re.I)
    if gemma_match:
        return f"Gemma {gemma_match.group(1)} E{gemma_match.group(2)}B"

    falcon_match = re.search(r"\bfalcon\s*h\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)([bm])\b", clean, re.I)
    if falcon_match:
        size = float(falcon_match.group(2)) / (1000 if falcon_match.group(3).lower() == "m" else 1)
        return f"Falcon H{falcon_match.group(1)} {format_billion(size)}B"

    generic_match = re.search(r"\b([a-z]+(?:\s+h)?)(?:\s+)?(\d+(?:\.\d+)?)?(?:\s+)?(\d+(?:\.\d+)?)([bm])\b", clean, re.I)
    if generic_match:
        family = title_case(generic_match.group(1).strip())
        version = f" {generic_match.group(2)}" if generic_match.group(2) else ""
        size = float(generic_match.group(3)) / (1000 if generic_match.group(4).lower() == "m" else 1)
        return f"{family}{version} {format_billion(size)}B"

    return title_case(clean or last)


def display_model_name(row: dict[str, Any]) -> str:
    return format_model_name(api_model(row) or row.get("base_model_name") or row.get("variant_name"))


def short_model_name(row: dict[str, Any]) -> str:
    return re.sub(r"\s+(Instruct|Reasoning)\b", "", display_model_name(row), flags=re.I).strip()


def provider_key(row: dict[str, Any]) -> str:
    source = " ".join(
        str(row.get(key) or "")
        for key in ("family", "base_model_name", "model_repo", "variant_name")
    ).lower()
    source = f"{source} {str(api_model(row) or '').lower()}"
    if "openai" in source or re.search(r"\bgpt-[\w.-]+", source):
        return "openai"
    if "nemotron" in source or "nvidia" in source:
        return "nvidia"
    if "liquid" in source or "lfm" in source:
        return "liquid"
    if "qwen" in source or "alibaba" in source:
        return "alibaba"
    if "gemma" in source or "google" in source:
        return "google"
    if "llama" in source or "meta" in source:
        return "meta"
    if "mistral" in source or "ministral" in source:
        return "mistral"
    if "granite" in source or "ibm" in source:
        return "ibm"
    if "olmo" in source or "allenai" in source:
        return "ai2"
    if "falcon" in source or "tii" in source:
        return "tii"
    if "smollm" in source or "hugging" in source:
        return "huggingface"
    if "phi" in source or "microsoft" in source:
        return "microsoft"
    return "local"


def quantization_label(row: dict[str, Any]) -> str:
    source = " ".join(
        str(row.get(key) or "")
        for key in ("quantization", "precision", "variant_name", "file_name")
    ).lower()
    source = f"{source} {str(api_model(row) or '').lower()}"
    labels = [
        (r"\b(?:fp32|f32|32\s*bit|32b)\b", "32 bit"),
        (r"\b(?:bf16|fp16|f16|16\s*bit|16b)\b", "16 bit"),
        (r"\b(?:q8|q8_0|int8|8\s*bit|8bit)\b", "8 bit"),
        (r"\b(?:q6|6\s*bit|6bit)\b", "6 bit"),
        (r"\b(?:q5|5\s*bit|5bit)\b", "5 bit"),
        (r"\b(?:q4|q4_k_m|4\s*bit|4bit)\b", "4 bit"),
    ]
    for pattern, label in labels:
        if re.search(pattern, source):
            return label
    quantization = row.get("quantization")
    if quantization and str(quantization).upper() != "SERVER":
        return title_case(str(quantization).replace("_", " "))
    return "Closed source"


def display_parameter(row: dict[str, Any]) -> str:
    parameter = numeric(row.get("parameter_size_b"))
    if parameter and parameter > 0:
        return f"{format_billion(parameter)}B"
    match = re.search(r"(\d+(?:\.\d+)?)B$", display_model_name(row), re.I)
    return f"{match.group(1)}B" if match else "n/a"


def file_size_bytes(row: dict[str, Any]) -> float | None:
    top = numeric(row.get("file_size_bytes"))
    if top is not None:
        return top
    meta = metadata(row)
    model_file_size = numeric(meta.get("model_file", {}).get("size_bytes"))
    if model_file_size is not None:
        return model_file_size
    key = api_model(row)
    if key and key in KNOWN_MODEL_FILE_SIZES:
        return float(KNOWN_MODEL_FILE_SIZES[key])
    return None


def estimated_bytes_per_parameter(row: dict[str, Any]) -> float | None:
    source = " ".join(
        str(row.get(key) or "")
        for key in ("quantization", "precision", "variant_name")
    ).lower()
    source = f"{source} {str(api_model(row) or '').lower()}"
    labels = [
        (r"\b(?:bf16|fp16|f16|16\s*bit|16b)\b", 2.0),
        (r"\b(?:q8|int8|8\s*bit|8bit)\b", 1.0),
        (r"\b(?:q6|6\s*bit|6bit)\b", 0.75),
        (r"\b(?:q5|5\s*bit|5bit)\b", 0.625),
        (r"\b(?:q4|4\s*bit|4bit)\b", 0.5),
    ]
    for pattern, value in labels:
        if re.search(pattern, source):
            return value
    return None


def model_size_gb(row: dict[str, Any]) -> float | None:
    bytes_value = file_size_bytes(row)
    if bytes_value is not None:
        return bytes_value / (1024 ** 3)
    parameter_size = numeric(row.get("parameter_size_b"))
    bytes_per_parameter = estimated_bytes_per_parameter(row)
    if parameter_size is None or bytes_per_parameter is None:
        return None
    return (parameter_size * 1_000_000_000 * bytes_per_parameter) / (1024 ** 3)


def format_model_size(row: dict[str, Any]) -> str:
    size = model_size_gb(row)
    if size is None:
        return "n/a"
    prefix = "~" if file_size_bytes(row) is None else ""
    return f"{prefix}{size:.1f} GB" if size < 10 else f"{prefix}{size:.0f} GB"


def row_time(row: dict[str, Any]) -> float:
    for key in ("run_started_at", "created_at", "generated_at", "updated_at", "model_release_date", "started_at"):
        candidate = row.get(key)
        if not candidate:
            continue
        try:
            return datetime.fromisoformat(str(candidate).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return 0.0


def score(row: dict[str, Any]) -> float:
    return numeric(row.get("model_intelligence_score")) or 0.0


def is_synthetic_row(row: dict[str, Any]) -> bool:
    return bool(row.get("metadata_json") and '"synthetic": true' in str(row.get("metadata_json")))


def is_smoke_row(row: dict[str, Any]) -> bool:
    samples = numeric(row.get("benchmark_samples"))
    return bool(re.search(r"\bsmoke\b", str(row.get("variant_name") or ""), re.I) or (samples is not None and samples <= 5))


def is_smollm2_row(row: dict[str, Any]) -> bool:
    source = " ".join(str(row.get(key) or "") for key in ("variant_name", "base_model_name", "model_repo")).lower()
    source = f"{source} {str(api_model(row) or '').lower()}"
    return any(part in source for part in ("smollm2", "smol lm2", "smol lm 2"))


def is_hosted(row: dict[str, Any]) -> bool:
    return provider_key(row) == "openai"


def is_four_bit(row: dict[str, Any]) -> bool:
    return quantization_label(row).lower() == "4 bit"


def comparable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = "|".join(
            [
                display_model_name(row).lower(),
                provider_key(row),
                display_parameter(row).lower(),
                quantization_label(row).lower(),
            ]
        )
        groups[key].append(row)

    merged_rows: list[dict[str, Any]] = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: (row_time(item), score(item)), reverse=True)
        merged = dict(ordered[0])
        for _, key in LAIA_KEYS:
            source = next((item for item in ordered if numeric(item.get(key)) is not None), None)
            merged[key] = numeric(source.get(key)) if source else None
        weighted = 0.0
        covered = 0.0
        for _, key in LAIA_KEYS:
            metric = numeric(merged.get(key))
            if metric is not None:
                covered += 0.2
            weighted += max(0.0, min(1.0, metric or 0.0)) * 0.2
        merged["model_intelligence_score"] = weighted
        merged["model_intelligence_coverage"] = covered
        merged_rows.append(merged)
    return merged_rows


def public_rows(rows: list[dict[str, Any]]) -> list[PlotRow]:
    return [
        PlotRow(row)
        for row in sorted(rows, key=score, reverse=True)
        if score(row) > 0 and (is_hosted(row) or is_four_bit(row))
    ]


def local_rows(rows: list[dict[str, Any]]) -> list[PlotRow]:
    return [
        PlotRow(row)
        for row in sorted(rows, key=score, reverse=True)
        if score(row) > 0 and not is_hosted(row) and is_four_bit(row)
    ]


def efficiency_frontier(rows: list[PlotRow]) -> list[PlotRow]:
    points = [row for row in rows if row.size_gb is not None and row.score > 0]
    frontier: list[PlotRow] = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate or other.size_gb is None:
                continue
            if (
                other.size_gb <= candidate.size_gb
                and other.score >= candidate.score
                and (other.size_gb < candidate.size_gb or other.score > candidate.score)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda row: (row.size_gb or 0.0, -row.score))


def svg_root(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" fill="none">'
        f'<rect width="{width}" height="{height}" fill="{BG}"/>'
        f'<style>'
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif;}"
        ".title{font-size:42px;font-weight:800;fill:#111111;}"
        ".subtitle{font-size:20px;font-weight:500;fill:#6b6b6b;}"
        ".label{font-size:20px;font-weight:700;fill:#111111;}"
        ".meta{font-size:16px;font-weight:600;fill:#6b6b6b;}"
        ".small{font-size:14px;font-weight:600;fill:#6b6b6b;}"
        ".axis{font-size:14px;font-weight:600;fill:#6b6b6b;}"
        "</style>"
        f"{body}</svg>"
    )


def text(x: float, y: float, value: str, klass: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{klass}" text-anchor="{anchor}">{escape(value)}</text>'


def text_colored(
    x: float,
    y: float,
    value: str,
    *,
    size: int,
    weight: int,
    color: str,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'style="font-size:{size}px;font-weight:{weight};fill:{color};'
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Arial,sans-serif\">"
        f"{escape(value)}</text>"
    )


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str | None = None, rx: float = 0.0) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}"{stroke_attr}/>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = LINE, width: float = 1.0, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dash_attr}/>'
    )


def circle(cx: float, cy: float, r: float, fill: str, stroke: str | None = None, width: float = 0.0, opacity: float | None = None) -> str:
    stroke_attr = f' stroke="{stroke}" stroke-width="{width:.1f}"' if stroke else ""
    opacity_attr = f' opacity="{opacity:.3f}"' if opacity is not None else ""
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"{stroke_attr}{opacity_attr}/>'


def image_data_uri(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    ext = img_path.suffix.lower()
    mime_by_ext = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = mime_by_ext.get(ext)
    if not mime:
        return ""
    encoded = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def laia_brand_mark(x: float, y: float, width: float = 112, opacity: float = 0.24) -> str:
    data_uri = image_data_uri(LAIA_LOGO_PATH)
    if not data_uri:
        return ""
    height = width * (490 / 1660)
    return (
        f'<image href="{data_uri}" x="{x:.1f}" y="{y:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}" opacity="{opacity:.3f}"/>'
    )


def provider_badge(x: float, y: float, label: str, color: str) -> str:
    width = max(72, len(label) * 9 + 22)
    body = rect(x, y - 18, width, 28, "#ffffff", LINE, 14)
    body += circle(x + 14, y - 4, 5, color)
    body += text(x + 26, y + 2, label, "small")
    return body


def score_label(value: float) -> str:
    return f"{value * 100:.1f}"


def generate_bar_plot(rows: list[PlotRow], generated_at: str) -> str:
    width, height = 1200, 1600

    def get_image_data_uri(lab_key: str) -> str:
        labs_dir = ROOT / "web" / "public" / "labs"
        for ext in [".svg", ".png", ".PNG", ".jpg", ".jpeg", ".webp"]:
            img_path = labs_dir / f"{lab_key}{ext}"
            if img_path.exists():
                return image_data_uri(img_path)
        return ""

    def get_lab_key(row_dict: dict[str, Any]) -> str:
        source = " ".join(
            str(row_dict.get(key) or "")
            for key in ("family", "base_model_name", "model_repo", "variant_name")
        ).lower()
        source = f"{source} {str(api_model(row_dict) or '').lower()}"
        if "nvidia" in source or "nemotron" in source:
            return "nvidia"
        if "qwen" in source or "alibaba" in source:
            return "qwen"
        if "google" in source or "gemma" in source:
            return "google"
        if "meta" in source or "llama" in source:
            return "meta"
        if "mistral" in source or "ministral" in source:
            return "mistral"
        if "tii" in source or "falcon" in source:
            return "TechnologyInnovationINstitute"
        if "microsoft" in source or "phi" in source:
            return "microsoft"
        if "ibm" in source or "granite" in source:
            return "ibm"
        if "ai2" in source or "olmo" in source:
            return "ai2"
        if "openai" in source:
            return "openai"
        if "liquid" in source or "lfm" in source:
            return "liquidAI"
        if "hugging" in source or "smollm" in source:
            return "huggingface"
        return re.sub(r"[^a-z0-9]+", "", (row_dict.get("family") or row_dict.get("base_model_name") or "ai").lower())

    parts = [laia_brand_mark(width - 158, 58, 106, 0.28)]

    # Spacing and layout configuration
    track_y = 800
    track_h = 480
    bar_w = 146
    col_w = 170
    gap = 8
    logo_size = 64
    max_score = max(row.score for row in rows) or 1.0
    start_x = (width - (6 * col_w + 5 * gap)) / 2

    for i, row in enumerate(rows):
        col_x = start_x + i * (col_w + gap)
        score_val = row.score
        score_pct = score_val * 100
        filled_h = (score_val / max_score) * track_h
        fill_color = row.provider["color"]

        # Center track in column
        track_x = col_x + (col_w - bar_w) / 2

        # Draw filled bar with both top and bottom rounded corners (capsule shape)
        bar_y = track_y + track_h - filled_h
        r_radius = 20
        parts.append(f'<rect x="{track_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{filled_h:.1f}" rx="{r_radius}" ry="{r_radius}" fill="{fill_color}"/>')

        # Score text inside bar (rounded to nearest integer, bold)
        score_text_y = track_y + track_h - 32
        score_display = str(int(round(score_pct)))
        parts.append(f'<text x="{track_x + bar_w/2:.1f}" y="{score_text_y:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="34px" font-weight="bold" fill="#ffffff">{score_display}</text>')

        # Label Area
        label_y = track_y + track_h + 30
        
        # 1. Lab Icon
        lab_key = get_lab_key(row.raw)
        data_uri = get_image_data_uri(lab_key)
        if data_uri:
            icon_x = col_x + (col_w - logo_size) / 2
            parts.append(f'<image href="{data_uri}" x="{icon_x:.1f}" y="{label_y:.1f}" width="{logo_size}" height="{logo_size}"/>')
        else:
            initials = (row.raw.get("family") or "AI")[:2].upper()
            icon_x = col_x + (col_w - logo_size) / 2
            text_x = col_x + col_w / 2
            parts.append(f'<rect x="{icon_x:.1f}" y="{label_y:.1f}" width="{logo_size}" height="{logo_size}" rx="{logo_size/2}" fill="#eeeeec"/>')
            parts.append(f'<text x="{text_x:.1f}" y="{label_y + 38:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="20px" font-weight="bold" fill="#101010">{initials}</text>')

        # 2. Model Name (bold like the website)
        parts.append(f'<text x="{col_x + col_w/2:.1f}" y="{label_y + logo_size + 42:.1f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="24px" font-weight="bold" fill="#101010">{escape(row.name)}</text>')

    return svg_root(width, height, "".join(parts))


def generate_scatter_plot(all_rows: list[PlotRow], frontier: list[PlotRow], generated_at: str) -> str:
    width, height = 1600, 980
    left, right, top, bottom = 140, 90, 180, 110
    plot_w = width - left - right
    plot_h = height - top - bottom

    points = [row for row in all_rows if row.size_gb is not None]
    max_x = math.ceil(max(row.size_gb or 0 for row in points) * 1.08)
    max_x = max(1, max_x)

    def x_for(x: float) -> float:
        return left + (x / max_x) * plot_w

    def y_for(y: float) -> float:
        return top + plot_h - (y * plot_h)

    parts = [
        laia_brand_mark(width - right - 114, 72, 104, 0.28),
        text(left, 82, "Pareto frontier by model size", "title"),
        text(
            left,
            116,
            f"Local 4-bit rows only. A frontier model is not beaten on both size and score. Generated {generated_at}.",
            "subtitle",
        ),
        line(left, 138, width - right, 138, LINE, 1.2),
        rect(left, top, plot_w, plot_h, "#ffffff", LINE, 18),
    ]

    for tick in range(0, max_x + 1):
        x = x_for(tick)
        parts.append(line(x, top, x, top + plot_h, SOFT, 1))
        parts.append(text(x, top + plot_h + 32, f"{tick} GB", "axis", "middle"))

    for tick in [0.2, 0.4, 0.6, 0.8]:
        y = y_for(tick)
        parts.append(line(left, y, left + plot_w, y, SOFT, 1))
        parts.append(text(left - 18, y + 4, f"{tick * 100:.0f}", "axis", "end"))

    parts.append(text(left - 34, top - 14, "LAIA", "small"))
    parts.append(text(left + plot_w - 4, top + plot_h + 70, "Model size (GB)", "small", "end"))

    frontier_coords = [(x_for(row.size_gb or 0.0), y_for(row.score)) for row in frontier]
    if frontier_coords:
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in frontier_coords)
        parts.append(f'<polyline points="{path}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="8 6"/>')

    frontier_ids = {id(row) for row in frontier}
    for row in points:
        x = x_for(row.size_gb or 0.0)
        y = y_for(row.score)
        opacity = 0.18 if id(row) not in frontier_ids else 1.0
        radius = 6 if id(row) not in frontier_ids else 10
        parts.append(circle(x, y, radius, row.provider["color"], "#ffffff", 2, opacity))

    card_offsets = [
        (-92, -138),
        (-86, -120),
        (-46, -154),
        (-26, -128),
        (8, -158),
        (22, -130),
        (18, -110),
        (14, -148),
    ]
    card_w = 188
    card_h = 68
    for index, row in enumerate(frontier):
        dot_x = x_for(row.size_gb or 0.0)
        dot_y = y_for(row.score)
        dx, dy = card_offsets[index % len(card_offsets)]
        card_x = min(max(left + 12, dot_x + dx), left + plot_w - card_w - 12)
        card_y = min(max(top + 12, dot_y + dy), top + plot_h - card_h - 16)
        parts.append(line(card_x + card_w / 2, card_y + card_h, dot_x, dot_y - 12, "#b7b7b7", 1.2))
        parts.append(rect(card_x, card_y, card_w, card_h, "#ffffff", LINE, 14))
        parts.append(rect(card_x, card_y, 8, card_h, row.provider["color"], None, 14))
        parts.append(text(card_x + 20, card_y + 28, row.name, "small"))
        parts.append(text(card_x + 20, card_y + 50, f"{score_label(row.score)} pts  |  {format_model_size(row.raw)}", "small"))

    parts.append(text(left, height - 42, "Frontier rows currently run from LFM 2.5 350M at 0.2 GB up to Qwen 3.5 9B at 7.0 GB.", "small"))
    return svg_root(width, height, "".join(parts))


def read_world_country_path() -> str:
    source = WORLD_MAP_PATH.read_text()
    match = re.search(r'WORLD_COUNTRY_PATH = "(.*)";', source)
    if not match:
        raise RuntimeError("Could not find WORLD_COUNTRY_PATH in web/src/worldMapPaths.ts")
    return match.group(1)


def project_origin(origin: dict[str, Any], map_x: float, map_y: float, map_w: float, map_h: float) -> tuple[float, float]:
    x = ((origin["lon"] + 180) / 360) * 1000
    y = ((90 - origin["lat"]) / 180) * 500
    return map_x + (x / 1000) * map_w, map_y + (y / 500) * map_h


def generate_world_map(rows: list[PlotRow], generated_at: str) -> str:
    width, height = 1600, 940
    map_x, map_y, map_w, map_h = 72, 170, 1050, 560
    legend_x = 1160
    world_path = read_world_country_path()

    present = {row.provider_key for row in rows if row.provider_key != "openai"}
    origins = [origin for origin in ORIGINS if origin["id"] in present]

    parts = [
        laia_brand_mark(width - 186, 72, 104, 0.28),
        text(72, 82, "Where local models come from", "title"),
        text(
            72,
            116,
            f"Headquarters cities for labs represented in the local 4-bit comparison surface. Generated {generated_at}.",
            "subtitle",
        ),
        line(72, 138, width - 72, 138, LINE, 1.2),
        rect(map_x, map_y, map_w, map_h, "#ffffff", LINE, 18),
        f'<svg x="{map_x}" y="{map_y}" width="{map_w}" height="{map_h}" viewBox="0 0 1000 500">'
        f'<path d="{escape(world_path)}" fill="#fdfdfd" stroke="#111111" stroke-width="0.9"/>'
        "</svg>",
    ]

    for origin in origins:
        px, py = project_origin(origin, map_x, map_y, map_w, map_h)
        color = PROVIDERS[origin["id"]]["color"]
        parts.append(circle(px, py, 12, color, INK, 1.4, 0.35))
        parts.append(circle(px, py, 4.8, INK, "#ffffff", 1.4))

    parts.append(text(legend_x, 200, "Labs in current local rows", "label"))
    card_y = 238
    for index, origin in enumerate(origins):
        y = card_y + index * 58
        if y > height - 70:
            break
        color = PROVIDERS[origin["id"]]["color"]
        label = PROVIDERS[origin["id"]]["label"]
        parts.append(rect(legend_x, y - 26, 320, 42, "#ffffff", LINE, 12))
        parts.append(rect(legend_x, y - 26, 8, 42, color, None, 12))
        parts.append(text(legend_x + 20, y - 2, label, "small"))
        parts.append(text(legend_x + 20, y + 18, f"{origin['city']}, {origin['country']}", "small"))

    parts.append(text(72, 858, "OpenAI reference rows are excluded from the map. The current local field spans the US, Europe, the Gulf, and China.", "small"))
    return svg_root(width, height, "".join(parts))


def save_outputs(name: str, svg_content: str) -> None:
    svg_path = OUTPUT_DIR / f"{name}.svg"
    svg_path.write_text(svg_content)
    if cairosvg is not None:
        png_path = OUTPUT_DIR / f"{name}.png"
        cairosvg.svg2png(bytestring=svg_content.encode("utf-8"), write_to=str(png_path), background_color=BG)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(RESULTS_PATH.read_text())
    generated_raw = payload.get("generated_at")
    try:
        generated_at = datetime.fromisoformat(str(generated_raw).replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        generated_at = str(generated_raw)

    rows = payload.get("leaderboard", [])
    clean_rows = comparable_rows(
        [
            row
            for row in rows
            if not is_synthetic_row(row) and not is_smoke_row(row) and not is_smollm2_row(row)
        ]
    )
    public_ranked = public_rows(clean_rows)
    local_ranked = local_rows(clean_rows)
    frontier = efficiency_frontier(local_ranked)

    save_outputs("top-6-models", generate_bar_plot(public_ranked[:6], generated_at))
    save_outputs("pareto-frontier", generate_scatter_plot(local_ranked, frontier, generated_at))
    save_outputs("model-origins-map", generate_world_map(local_ranked, generated_at))

    print(f"Wrote assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
