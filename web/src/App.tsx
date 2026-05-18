import {
  BookOpen,
  CheckCircle2,
  Code2,
  Cpu,
  Eye,
  FileText,
  Lightbulb,
  LightbulbOff,
  ListFilter,
  RotateCcw,
  Search,
  ShieldCheck,
} from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";

type LeaderboardRow = {
  normalized_result_id?: string;
  variant_id: string;
  base_model_id: string;
  family: string;
  base_model_name: string;
  parameter_size_b: number;
  variant_name: string;
  quantization: string;
  precision?: string | null;
  model_repo?: string | null;
  file_name?: string | null;
  checksum_sha256?: string | null;
  file_size_bytes?: number | null;
  is_baseline?: boolean;
  backend_name?: string | null;
  backend_version?: string | null;
  backend_commit?: string | null;
  hardware_accelerator?: string | null;
  cpu_model?: string | null;
  gpu_name?: string | null;
  run_uuid?: string | null;
  global_mmlu_lite_pass_at_1?: number | null;
  global_mmlu_lite_micro_pass_at_1?: number | null;
  global_mmlu_lite_invalid_rate?: number | null;
  ifbench_prompt_level_loose?: number | null;
  ifbench_instruction_level_loose?: number | null;
  ifbench_prompt_level_strict?: number | null;
  ifbench_instruction_level_strict?: number | null;
  bfcl_v4_selected_accuracy?: number | null;
  bfcl_v4_invalid_rate?: number | null;
  bfcl_v4_non_live_accuracy?: number | null;
  bfcl_v4_live_accuracy?: number | null;
  bfcl_v4_multi_turn_accuracy?: number | null;
  bfcl_v4_agentic_accuracy?: number | null;
  ocrbench_v2_score?: number | null;
  ocrbench_v2_micro_score?: number | null;
  ocrbench_v2_en_score?: number | null;
  ocrbench_v2_cn_score?: number | null;
  mmmu_accuracy?: number | null;
  mmmu_invalid_rate?: number | null;
  mmmu_multiple_choice_accuracy?: number | null;
  mmmu_open_accuracy?: number | null;
  mbpp_pass_at_1?: number | null;
  mbpp_invalid_rate?: number | null;
  mbpp_compile_rate?: number | null;
  mbpp_runtime_error_rate?: number | null;
  rgb_all_rate?: number | null;
  rgb_rejection_rate?: number | null;
  rgb_fact_check_rate?: number | null;
  rgb_error_correction_rate?: number | null;
  simpleqa_f1?: number | null;
  simpleqa_correct_rate?: number | null;
  simpleqa_incorrect_rate?: number | null;
  simpleqa_hallucination_rate?: number | null;
  simpleqa_not_attempted_rate?: number | null;
  simpleqa_accuracy_given_attempted?: number | null;
  harmbench_attack_success_rate?: number | null;
  harmbench_refusal_rate?: number | null;
  model_intelligence_score?: number | null;
  model_intelligence_coverage?: number | null;
  model_intelligence_available_score?: number | null;
  benchmark_runtime_seconds?: number | null;
  benchmark_samples?: number | null;
  benchmark_correct_count?: number | null;
  benchmark_prompt_tokens?: number | null;
  benchmark_completion_tokens?: number | null;
  benchmark_total_tokens?: number | null;
  benchmark_reasoning_tokens?: number | null;
  benchmark_output_tokens_per_second?: number | null;
  benchmark_total_tokens_per_second?: number | null;
  benchmark_avg_latency_seconds?: number | null;
  benchmark_p50_latency_seconds?: number | null;
  benchmark_p95_latency_seconds?: number | null;
  benchmark_truncated_count?: number | null;
  benchmark_truncated_rate?: number | null;
  benchmark_tokens_per_correct_answer?: number | null;
  benchmark_seconds_per_correct_answer?: number | null;
  benchmark_time_to_first_token_seconds?: number | null;
  benchmark_inter_token_latency_seconds?: number | null;
  benchmark_end_to_end_latency_seconds?: number | null;
  benchmark_system_output_throughput_tokens_per_second?: number | null;
  benchmark_input_cost_usd?: number | null;
  benchmark_output_cost_usd?: number | null;
  benchmark_total_cost_usd?: number | null;
  benchmark_cost_per_correct_answer_usd?: number | null;
  metadata_json?: string | null;
};

type Payload = {
  generated_at: string;
  tagline: string;
  leaderboard: LeaderboardRow[];
  filters?: Record<string, unknown[]>;
};

type Filters = {
  query: string;
  family: string;
  parameterSize: string;
  maxModelSizeGb: string;
};

type QuantizationOption = {
  key: string;
  label: string;
  row: LeaderboardRow;
  score: number | null;
  rank: number;
};

type ComparableRow = LeaderboardRow;

type MetricColumn = {
  key: string;
  label: string;
  render: (row: ComparableRow) => ReactNode;
  primary?: boolean;
};

type SizeSort = "asc" | "desc" | null;

const MERGED_BENCHMARK_METRICS = [
  "global_mmlu_lite_pass_at_1",
  "global_mmlu_lite_micro_pass_at_1",
  "global_mmlu_lite_invalid_rate",
  "ifbench_prompt_level_loose",
  "ifbench_instruction_level_loose",
  "ifbench_prompt_level_strict",
  "ifbench_instruction_level_strict",
  "bfcl_v4_selected_accuracy",
  "bfcl_v4_invalid_rate",
  "bfcl_v4_non_live_accuracy",
  "bfcl_v4_live_accuracy",
  "bfcl_v4_multi_turn_accuracy",
  "bfcl_v4_agentic_accuracy",
  "ocrbench_v2_score",
  "ocrbench_v2_micro_score",
  "ocrbench_v2_en_score",
  "ocrbench_v2_cn_score",
  "mmmu_accuracy",
  "mmmu_invalid_rate",
  "mmmu_multiple_choice_accuracy",
  "mmmu_open_accuracy",
  "mbpp_pass_at_1",
  "mbpp_invalid_rate",
  "mbpp_compile_rate",
  "mbpp_runtime_error_rate",
  "rgb_all_rate",
  "rgb_rejection_rate",
  "rgb_fact_check_rate",
  "rgb_error_correction_rate",
  "simpleqa_f1",
  "simpleqa_correct_rate",
  "simpleqa_incorrect_rate",
  "simpleqa_hallucination_rate",
  "simpleqa_not_attempted_rate",
  "simpleqa_accuracy_given_attempted",
  "harmbench_attack_success_rate",
  "harmbench_refusal_rate",
] satisfies Array<keyof LeaderboardRow>;

const LAIA_INDEX_WEIGHTS = {
  global_mmlu_lite_pass_at_1: 0.25,
  ifbench_prompt_level_loose: 0.20,
  bfcl_v4_selected_accuracy: 0.20,
  mbpp_pass_at_1: 0.20,
  rgb_all_rate: 0.15,
} satisfies Partial<Record<keyof LeaderboardRow, number>>;

const emptyFilters: Filters = {
  query: "",
  family: "all",
  parameterSize: "all",
  maxModelSizeGb: "",
};

export function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(emptyFilters);
  const [showFilters, setShowFilters] = useState(false);
  const [showAllVersions, setShowAllVersions] = useState(false);
  const [page, setPage] = useState<"leaderboard" | "methodology">("leaderboard");
  const [sizeSort, setSizeSort] = useState<SizeSort>(null);

  useEffect(() => {
    loadPayload()
      .then(setPayload)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const rawRows = payload?.leaderboard ?? [];
  const realRows = useMemo(() => rawRows.filter((row) => !isSyntheticRow(row)), [rawRows]);
  const publishableRows = useMemo(() => realRows.filter((row) => !isSmokeRow(row)), [realRows]);
  const filteredRows = useMemo(
    () => applyFilters(publishableRows, filters),
    [publishableRows, filters],
  );
  const comparableRows = useMemo(() => buildComparableRows(filteredRows), [filteredRows]);
  const displayedComparableRows = useMemo(
    () => (
      showAllVersions
        ? comparableRows
        : comparableRows.filter((row) => isFourBitRow(row) || isHostedOpenAIRow(row))
    ),
    [comparableRows, showAllVersions],
  );
  const rankedRows = useMemo(() => rankRows(displayedComparableRows), [displayedComparableRows]);
  const visibleRows = useMemo(() => sortRowsForDisplay(rankedRows, sizeSort), [rankedRows, sizeSort]);
  const options = useMemo(() => optionSets(publishableRows), [publishableRows]);

  if (error) {
    return (
      <main className="state-shell">
        <p className="eyebrow">Local AI Analysis</p>
        <h1>Benchmark data failed to load.</h1>
        <p className="error">{error}</p>
      </main>
    );
  }

  if (!payload) {
    return (
      <main className="state-shell">
        <p className="eyebrow">Local AI Analysis</p>
        <h1>Loading benchmark results</h1>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <SiteHeader generatedAt={payload.generated_at} page={page} onNavigate={setPage} />

      {page === "leaderboard" && (
        <>
          <div className="page-subtitle">
            <p>
              Compare local models by LAIA Index across knowledge, instructions, tool use,
              coding, and RAG. Vision, factuality, and safety are shown when available.
              Updated {formatDate(payload.generated_at)}.
            </p>
          </div>

          <section className="workspace-grid" id="leaderboard">
            <section className="leaderboard-panel" aria-labelledby="leaderboard-title">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Local AI Leaderboard</p>
                  <h2 id="leaderboard-title">Capability rankings</h2>
                </div>
                <div className="panel-actions">
                  <button
                    className={`version-toggle-button ${showAllVersions ? "active" : ""}`}
                    type="button"
                    onClick={() => setShowAllVersions((current) => !current)}
                  >
                    {showAllVersions ? "Show 4 bit only" : "Show all versions"}
                  </button>
                  <button className="text-button" type="button" onClick={() => setShowFilters(!showFilters)}>
                    <ListFilter size={15} aria-hidden="true" />
                    {showFilters ? "Hide filters" : "Show filters"}
                  </button>
                </div>
              </div>

              {showFilters && (
                <FilterPanel
                  filters={filters}
                  options={options}
                  onChange={setFilters}
                  onReset={() => setFilters(emptyFilters)}
                />
              )}

              <LeaderboardTable
                rows={visibleRows}
                sizeSort={sizeSort}
                onToggleSizeSort={() => setSizeSort((current) => (
                  current === null ? "asc" : current === "asc" ? "desc" : null
                ))}
              />

              <IntelligenceSizeChart rows={visibleRows} />
            </section>
          </section>

          <footer className="footer">
            <span>Local AI Analysis</span>
            <span>{payload.tagline}</span>
          </footer>
        </>
      )}

      {page === "methodology" && <MethodologyPage />}
    </main>
  );
}

function SiteHeader({
  generatedAt,
  page,
  onNavigate,
}: {
  generatedAt: string;
  page: "leaderboard" | "methodology";
  onNavigate: (page: "leaderboard" | "methodology") => void;
}) {
  return (
    <header className="site-header">
      <button
        className="site-mark"
        style={{ all: "unset", cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "14px", fontWeight: 900 }}
        onClick={() => onNavigate("leaderboard")}
        aria-label="Local AI Analysis home"
      >
        <span style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 44, height: 44, border: "2px solid var(--ink)", background: "var(--ink)", color: "#fff", fontSize: "0.95rem", fontWeight: 900 }}>NC</span>
        <strong>Local AI Analysis</strong>
      </button>
      <nav aria-label="Primary navigation">
        <button
          className={page === "leaderboard" ? "nav-active" : ""}
          type="button"
          onClick={() => onNavigate("leaderboard")}
        >
          Leaderboard
        </button>
        <button
          className={page === "methodology" ? "nav-active" : ""}
          type="button"
          onClick={() => onNavigate("methodology")}
        >
          Methodology
        </button>
      </nav>
      <time dateTime={generatedAt}>Updated {formatDate(generatedAt)}</time>
    </header>
  );
}

function FilterPanel({
  filters,
  options,
  onChange,
  onReset,
}: {
  filters: Filters;
  options: ReturnType<typeof optionSets>;
  onChange: (filters: Filters) => void;
  onReset: () => void;
}) {
  return (
    <div className="filter-panel" aria-label="Leaderboard controls">
      <label className="search-field">
        <span>Text</span>
        <div>
          <Search size={16} aria-hidden="true" />
          <input
            type="search"
            value={filters.query}
            placeholder="Model, family, quantization"
            onChange={(event) => onChange({ ...filters, query: event.target.value })}
          />
        </div>
      </label>

      <div className="field-grid">
        <Select
          label="Family"
          value={filters.family}
          options={options.families}
          onChange={(family) => onChange({ ...filters, family })}
        />
        <Select
          label="Size"
          value={filters.parameterSize}
          options={options.parameterSizes}
          onChange={(parameterSize) => onChange({ ...filters, parameterSize })}
        />
        <NumberInput
          label="Max GB"
          value={filters.maxModelSizeGb}
          onChange={(maxModelSizeGb) => onChange({ ...filters, maxModelSizeGb })}
        />
      </div>

      <button className="reset-button" type="button" onClick={onReset}>
        <RotateCcw size={15} aria-hidden="true" />
        Reset filters
      </button>

    </div>
  );
}

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="all">All</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        min="0"
        step="0.1"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function LeaderboardTable({ rows, sizeSort, onToggleSizeSort }: {
  rows: ComparableRow[];
  sizeSort: SizeSort;
  onToggleSizeSort: () => void;
}) {
  const metricColumns = metricColumnsFor(rows);
  const columnCount = 2 + metricColumns.length;

  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            <th className="rank-heading">#</th>
            <th>Model</th>
            {metricColumns.map((column) => (
              <th className="numeric-heading" key={column.key}>
                {column.key === "model_size_gb" ? (
                  <button
                    className="sort-heading-button"
                    type="button"
                    onClick={onToggleSizeSort}
                    aria-label={`Sort by model size ${
                      sizeSort === "asc" ? "largest first" : sizeSort === "desc" ? "by LAIA Index" : "smallest first"
                    }`}
                    aria-sort={sizeSort === "asc" ? "ascending" : sizeSort === "desc" ? "descending" : "none"}
                  >
                    <span>{column.label}</span>
                    <span aria-hidden="true">{sizeSort === "asc" ? "↑" : sizeSort === "desc" ? "↓" : "↕"}</span>
                  </button>
                ) : (
                  column.label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {!rows.length && (
            <tr>
              <td className="empty-table" colSpan={columnCount}>
                No publishable benchmark rows match the current filters.
              </td>
            </tr>
          )}
          {rows.map((row, index) => (
            <tr
              className={leaderboardRowClass(row)}
              key={row.normalized_result_id ?? row.variant_id}
            >
              <td className="rank-cell">{String(index + 1).padStart(2, "0")}</td>
              <td className="model-cell">
                <div className="model-title-row">
                  <LabIcon row={row} />
                  <div className="model-title-stack">
                    <div className="model-title-line">
                      <strong title={row.variant_name}>{displayModelName(row)}</strong>
                      <ModelRunBadges row={row} />
                    </div>
                    <div className="model-subline">
                      <span>{providerLabel(row)} · {displayParameter(row)}</span>
                    </div>
                  </div>
                </div>
              </td>
              {metricColumns.map((column) => (
                <td
                  className={`numeric-cell ${column.primary ? "primary-score" : ""}`}
                  key={column.key}
                >
                  {column.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelRunBadges({ row }: { row: ComparableRow }) {
  const reasoningEnabled = reasoningKey(row) !== "off";
  const Icon = reasoningEnabled ? Lightbulb : LightbulbOff;
  const reasoningText = reasoningEnabled ? reasoningLabel(row) : "";
  return (
    <div className="model-badge-row" aria-label={`Run settings for ${displayModelName(row)}`}>
      <span className="model-badge">{quantizationLabel(row)}</span>
      <span className={`model-badge reasoning ${reasoningEnabled ? "on" : "off"}`}>
        <Icon size={13} aria-hidden="true" />
        {reasoningText ? <span>{reasoningText}</span> : null}
      </span>
    </div>
  );
}

function PointsWithDelta({ row }: { row: ComparableRow }) {
  const value = numeric(row.model_intelligence_score);
  return <span className="metric-stack">{formatPoints(value)}</span>;
}

function IntelligenceSizeChart({ rows }: { rows: ComparableRow[] }) {
  const points = rows
    .map((row) => ({
      row,
      sizeGb: modelSizeGb(row),
      score: numeric(row.model_intelligence_score),
      series: modelSeriesKey(row),
      quantization: quantizationLabel(row),
      quantizationTone: quantizationTone(row),
    }))
    .filter((point): point is {
      row: ComparableRow;
      sizeGb: number;
      score: number;
      series: string;
      quantization: string;
      quantizationTone: string;
    } => (
      point.sizeGb !== null && point.score !== null
    ));

  if (points.length < 2) {
    return null;
  }

  const width = 760;
  const height = 330;
  const padding = { top: 22, right: 34, bottom: 54, left: 62 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const maxSize = Math.max(...points.map((point) => point.sizeGb));
  const maxScore = Math.max(...points.map((point) => point.score), 0.01);
  const xMax = Math.max(1, Math.ceil(maxSize * 1.15));
  const yMax = Math.max(0.1, Math.ceil(maxScore * 120) / 100);
  const xTicks = [0, xMax / 2, xMax];
  const yTicks = [0, yMax / 2, yMax];
  const labelPoints = [...points]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  const labelIds = new Set(labelPoints.map((point) => point.row.variant_id));
  const seriesLines = Array.from(
    points.reduce((map, point) => {
      map.set(point.series, [...(map.get(point.series) ?? []), point]);
      return map;
    }, new Map<string, typeof points>()),
  )
    .map(([, seriesPoints]) => seriesPoints.sort((a, b) => a.sizeGb - b.sizeGb))
    .filter((seriesPoints) => seriesPoints.length > 1);
  const efficiencyRows = [...points]
    .map((point) => ({
      ...point,
      efficiency: (point.score * 100) / point.sizeGb,
    }))
    .sort((a, b) => b.efficiency - a.efficiency)
    .slice(0, 8);
  const maxEfficiency = Math.max(...efficiencyRows.map((point) => point.efficiency), 0.1);
  const legendItems = [
    { key: "bf16", label: "16 bit / BF16" },
    { key: "q8", label: "8 bit" },
    { key: "q4", label: "4 bit" },
    { key: "other", label: "Other" },
  ];

  const xFor = (value: number) => padding.left + (value / xMax) * plotWidth;
  const yFor = (value: number) => padding.top + plotHeight - (value / yMax) * plotHeight;

  return (
    <section className="chart-section" aria-labelledby="intelligence-size-title">
      <div className="chart-heading">
        <div>
          <p className="eyebrow">Efficiency Map</p>
          <h3 id="intelligence-size-title">Intelligence vs size</h3>
        </div>
        <span>{points.length} visible runs</span>
      </div>

      <div className="chart-legend" aria-label="Quantization legend">
        {legendItems.map((item) => (
          <span key={item.key}>
            <i className={`legend-dot quant-${item.key}`} aria-hidden="true" />
            {item.label}
          </span>
        ))}
      </div>

      <div className="chart-shell">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="intelligence-size-title">
          <rect
            x={padding.left}
            y={padding.top}
            width={plotWidth}
            height={plotHeight}
            className="chart-plot"
          />

          {xTicks.map((tick) => (
            <g key={`x-${tick}`}>
              <line
                className="chart-grid-line"
                x1={xFor(tick)}
                x2={xFor(tick)}
                y1={padding.top}
                y2={padding.top + plotHeight}
              />
              <text className="chart-axis-label" x={xFor(tick)} y={height - 22} textAnchor="middle">
                {tick === 0 ? "0" : `${tick.toFixed(tick < 10 ? 1 : 0)} GB`}
              </text>
            </g>
          ))}

          {yTicks.map((tick) => (
            <g key={`y-${tick}`}>
              <line
                className="chart-grid-line"
                x1={padding.left}
                x2={padding.left + plotWidth}
                y1={yFor(tick)}
                y2={yFor(tick)}
              />
              <text className="chart-axis-label" x={padding.left - 12} y={yFor(tick) + 4} textAnchor="end">
                {formatPoints(tick)}
              </text>
            </g>
          ))}

          <text className="chart-axis-title" x={padding.left + plotWidth / 2} y={height - 4} textAnchor="middle">
            Model size
          </text>
          <text
            className="chart-axis-title"
            transform={`translate(16 ${padding.top + plotHeight / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            LAIA Index
          </text>

          {seriesLines.map((seriesPoints) => (
            <polyline
              className="chart-series-line"
              key={seriesPoints.map((point) => point.row.variant_id).join("-")}
              points={seriesPoints.map((point) => `${xFor(point.sizeGb)},${yFor(point.score)}`).join(" ")}
            />
          ))}

          {points.map((point) => {
            const x = xFor(point.sizeGb);
            const y = yFor(point.score);
            const label = `${displayModelName(point.row)} ${point.quantization}`;
            const isLabeled = labelIds.has(point.row.variant_id);
            return (
              <g className="chart-point-group" key={point.row.variant_id}>
                <circle
                  className={`chart-point quant-${point.quantizationTone}`}
                  cx={x}
                  cy={y}
                  r={isLabeled ? 6 : 4.5}
                >
                  <title>
                    {`${label}: ${formatPoints(point.score)} at ${formatModelSize(point.row)}`}
                  </title>
                </circle>
                {isLabeled && (
                  <text className="chart-point-label" x={x + 9} y={y - 8}>
                    {displayModelName(point.row)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      <div className="efficiency-panel" aria-label="LAIA per GB ranking">
        <div className="efficiency-heading">
          <strong>LAIA per GB</strong>
          <span>Higher means more score per memory footprint.</span>
        </div>
        <div className="efficiency-list">
          {efficiencyRows.map((point) => (
            <div className="efficiency-row" key={`eff-${point.row.variant_id}`}>
              <div>
                <strong>{displayModelName(point.row)}</strong>
                <span>{point.quantization} · {formatModelSize(point.row)}</span>
              </div>
              <div className="efficiency-bar" aria-hidden="true">
                <span style={{ width: `${Math.max(6, (point.efficiency / maxEfficiency) * 100)}%` }} />
              </div>
              <b>{point.efficiency.toFixed(1)} pts/GB</b>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function LabIcon({ row }: { row: LeaderboardRow }) {
  const [candidateIndex, setCandidateIndex] = useState(0);
  const key = labKey(row);
  const candidates = [
    `/labs/${key}.svg`,
    `/labs/${key}.png`,
    `/labs/${key}.PNG`,
    `/labs/${key}.jpg`,
    `/labs/${key}.JPG`,
    `/labs/${key}.jpeg`,
    `/labs/${key}.JPEG`,
    `/labs/${key}.webp`,
    `/labs/${key}.WEBP`,
  ];
  const src = candidates[candidateIndex];
  const initials = labInitials(row);

  if (!key || candidateIndex >= candidates.length) {
    return <span className="lab-icon fallback" aria-hidden="true">{initials}</span>;
  }

  return (
    <span className="lab-icon" aria-hidden="true">
      <img
        src={src}
        alt=""
        onError={() => setCandidateIndex((index) => index + 1)}
      />
    </span>
  );
}

function MethodologyPage() {
  const benchmarks = [
    {
      name: "Global MMLU Lite",
      category: "Knowledge",
      metric: "Pass@1",
      icon: <BookOpen size={16} />,
      description:
        "A lightweight version of the Massive Multitask Language Understanding benchmark covering 57 academic subjects. Tests factual recall and reasoning across science, history, law, medicine, and more.",
      what: "Knowledge breadth",
    },
    {
      name: "IFBench",
      category: "Instructions",
      metric: "Prompt-level loose",
      icon: <FileText size={16} />,
      description:
        "Evaluates how accurately a model follows complex, multi-constraint instructions in a single prompt. Scored at both the prompt level (all constraints met) and instruction level (per constraint).",
      what: "Instruction following",
    },
    {
      name: "BFCL v4",
      category: "Tool calling",
      metric: "Selected accuracy",
      icon: <Code2 size={16} />,
      description:
        "Berkeley Function Calling Leaderboard v4. Tests the model's ability to invoke tools and APIs correctly across live, non-live, multi-turn, and agentic scenarios.",
      what: "Function / tool use",
    },
    {
      name: "MMMU",
      category: "Vision",
      metric: "Accuracy",
      icon: <Eye size={16} />,
      description:
        "Massive Multidiscipline Multimodal Understanding. Evaluates vision-language models across 30 university-level subjects using images combined with text questions.",
      what: "Visual reasoning",
    },
    {
      name: "OCRBench v2",
      category: "Vision",
      metric: "Score",
      icon: <FileText size={16} />,
      description:
        "Comprehensive OCR evaluation covering scene text, document understanding, handwriting, and formula recognition in both English and Chinese.",
      what: "Optical character recognition",
    },
    {
      name: "MBPP",
      category: "Coding",
      metric: "Pass@1",
      icon: <Code2 size={16} />,
      description:
        "Mostly Basic Programming Problems. Tests whether the model can write correct Python functions from natural-language specifications, evaluated by running the generated code against test suites.",
      what: "Code generation",
    },
    {
      name: "RGB",
      category: "RAG",
      metric: "All rate",
      icon: <BookOpen size={16} />,
      description:
        "Retrieval-augmented Generation Benchmark. Tests whether the model can correctly use retrieved context, reject irrelevant documents, verify facts against evidence, and correct prior wrong answers.",
      what: "Retrieval-augmented generation",
    },
    {
      name: "SimpleQA",
      category: "Factuality",
      metric: "F1",
      icon: <CheckCircle2 size={16} />,
      description:
        "Short-answer factual questions with a single, unambiguous correct answer. Measures hallucination rate alongside correct-answer rate. Scored by an LLM judge.",
      what: "Factuality & hallucination",
    },
    {
      name: "HarmBench",
      category: "Safety",
      metric: "Refusal rate",
      icon: <ShieldCheck size={16} />,
      description:
        "Standardized evaluation of model safety against behaviorally diverse harmful requests. Refusal rate measures how often the model correctly declines to comply with harmful prompts.",
      what: "Safety & refusals",
    },
  ];

  const principles = [
    {
      icon: <CheckCircle2 size={18} />,
      title: "Capability-first scoring",
      text: "The LAIA Index is a 100-point text-model score: Knowledge 25, Instructions 20, Tool calling 20, Coding 20, and RAG 15. Vision and judge-based results stay separate.",
    },
    {
      icon: <Cpu size={18} />,
      title: "Full traceability",
      text: "Every row links to the exact benchmark suite, model file, quantization level, backend version, hardware config, and run UUID. You can always reproduce what you see.",
    },
    {
      icon: <ShieldCheck size={18} />,
      title: "Separate judge-based metrics",
      text: "Factuality (SimpleQA) and Safety (HarmBench) require an LLM judge and are reported as distinct columns, not folded into the LAIA Index, to avoid bias.",
    },
    {
      icon: <Eye size={18} />,
      title: "Local hardware focus",
      text: "All benchmarks run on consumer hardware without cloud inference. Runtime is measured end-to-end, including model loading, so the number reflects real-world cost.",
    },
  ];

  return (
    <div className="methodology-page">
      <div className="method-hero">
        <p className="eyebrow">Methodology</p>
        <h1>Reproducibility first.</h1>
        <p className="method-lead">
          Every result in this leaderboard is produced by running open-source benchmarks on local
          hardware with published model files. The goal is a fair, transparent comparison of what
          models can actually do when running close to the user.
        </p>
      </div>

      <div className="method-section">
        <h2>LAIA Index</h2>
        <p>
          The LAIA Index is the primary ranking metric. It is a 100-point text-model score built
          from non-judge benchmarks: Knowledge, Instructions, Tool calling, Coding, and RAG. Vision
          benchmarks are kept as separate columns when available, but they do not affect LAIA. Missing
          text benchmarks do not receive guessed values, so partial runs show only the points earned
          from completed benchmark categories.
        </p>
        <dl className="method-detail-list">
          <div><dt>Scale</dt><dd>0 – 100 pts</dd></div>
          <div><dt>Weights</dt><dd>25 / 20 / 20 / 20 / 15</dd></div>
          <div><dt>Partial results</dt><dd>No imputation for missing benchmarks</dd></div>
          <div><dt>Exclusions</dt><dd>Vision, Factuality, Safety (separate columns)</dd></div>
        </dl>
      </div>

      <div className="method-section">
        <h2>Benchmark suite</h2>
        <p>
          Nine benchmarks cover the capability surface relevant to local model deployment.
          Each leaderboard row is a specific model quantization and reasoning mode.
        </p>
        <div className="benchmark-cards">
          {benchmarks.map((b) => (
            <div className="benchmark-card" key={b.name}>
              <div className="benchmark-card-header">
                <strong>{b.name}</strong>
                <span className="benchmark-badge">{b.category}</span>
              </div>
              <p>{b.description}</p>
              <dl className="benchmark-meta">
                <div><dt>Metric</dt><dd>{b.metric}</dd></div>
                <div><dt>Tests</dt><dd>{b.what}</dd></div>
              </dl>
            </div>
          ))}
        </div>
      </div>

      <div className="method-section">
        <h2>Principles</h2>
        <div className="method-principles">
          {principles.map((p) => (
            <div className="method-principle" key={p.title}>
              <span className="metric-icon">{p.icon}</span>
              <div>
                <strong>{p.title}</strong>
                <p>{p.text}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <footer className="footer">
        <span>Local AI Analysis</span>
        <span>Benchmarks run on local hardware · Results updated periodically</span>
      </footer>
    </div>
  );
}

async function loadPayload(): Promise<Payload> {
  const apiUrl = import.meta.env.VITE_API_URL;
  const urls = apiUrl ? [`${apiUrl.replace(/\/$/, "")}/api/leaderboard`, "/results.json"] : ["/results.json"];
  let lastError: unknown;
  for (const url of urls) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`${url} returned ${response.status}`);
      return response.json();
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError;
}

function applyFilters(rows: LeaderboardRow[], filters: Filters) {
  const query = filters.query.trim().toLowerCase();
  const maxModelSizeGb = parseNumber(filters.maxModelSizeGb);

  return rows.filter((row) => {
    const searchable = [
      row.variant_name,
      row.base_model_name,
      row.family,
      row.quantization,
      quantizationLabel(row),
      formatParameter(row.parameter_size_b),
      formatModelSize(row),
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    const sizeGb = modelSizeGb(row);

    return (
      (!query || searchable.includes(query)) &&
      (filters.family === "all" || row.family === filters.family) &&
      (filters.parameterSize === "all" || String(row.parameter_size_b) === filters.parameterSize) &&
      (maxModelSizeGb === null || meetsMaximum(sizeGb, maxModelSizeGb))
    );
  });
}

function optionSets(rows: LeaderboardRow[]) {
  return {
    families: unique(rows.map((row) => row.family)),
    parameterSizes: unique(rows.map((row) => String(row.parameter_size_b))),
  };
}

function buildComparableRows(rows: LeaderboardRow[]): ComparableRow[] {
  const groups = new Map<string, LeaderboardRow[]>();
  for (const row of rows) {
    const key = quantizationGroupKey(row);
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }

  return Array.from(groups.values()).flatMap((groupRows) => {
    const allQuantizationOptions = quantizationOptionsFor(groupRows);

    return allQuantizationOptions.map((selectedQuantization) => {
      const selectedRows = rowsForQuantization(groupRows, selectedQuantization.key);
      return mergeComparableRuns(selectedRows);
    });
  });
}

function mergeComparableRuns(rows: LeaderboardRow[]) {
  const base = bestComparableRun(rows);
  const merged: LeaderboardRow = { ...base };

  for (const key of MERGED_BENCHMARK_METRICS) {
    if (numeric(merged[key]) !== null) continue;
    const source = rows.find((row) => numeric(row[key]) !== null);
    if (source) {
      merged[key] = source[key] as never;
    }
  }
  const intelligence = laiaIndexValues(merged);
  merged.model_intelligence_score = intelligence.score;
  merged.model_intelligence_coverage = intelligence.coverage;
  merged.model_intelligence_available_score = intelligence.availableScore;

  return merged;
}

function laiaIndexValues(row: LeaderboardRow) {
  let weightedSum = 0;
  let coveredWeight = 0;

  for (const [key, weight] of Object.entries(LAIA_INDEX_WEIGHTS) as Array<
    [keyof typeof LAIA_INDEX_WEIGHTS, number]
  >) {
    const value = numeric(row[key]);
    if (value === null) continue;
    weightedSum += Math.max(0, Math.min(1, value)) * weight;
    coveredWeight += weight;
  }

  if (coveredWeight <= 0) {
    return { score: null, coverage: null, availableScore: null };
  }
  return {
    score: weightedSum,
    coverage: coveredWeight,
    availableScore: weightedSum / coveredWeight,
  };
}

function quantizationOptionsFor(rows: LeaderboardRow[], preferredReasoningKey?: string) {
  const byQuantization = new Map<string, QuantizationOption>();
  const grouped = new Map<string, LeaderboardRow[]>();
  for (const row of rows) {
    const key = quantizationKey(row);
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  }

  for (const [key, quantizationRows] of grouped.entries()) {
    const preferredRows = preferredReasoningKey
      ? quantizationRows.filter((row) => reasoningKey(row) === preferredReasoningKey)
      : [];
    const row = bestComparableRun(preferredRows.length ? preferredRows : quantizationRows);
    const option: QuantizationOption = {
      key,
      label: quantizationLabel(row),
      row,
      score: qualityValue(row),
      rank: quantizationRank(row),
    };
    byQuantization.set(key, option);
  }

  return Array.from(byQuantization.values()).sort((a, b) => {
    if (b.rank !== a.rank) return b.rank - a.rank;
    return (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY);
  });
}

function rowsForQuantization(rows: LeaderboardRow[], key: string) {
  return rows.filter((row) => quantizationKey(row) === key);
}

function bestComparableRun(rows: LeaderboardRow[]) {
  return [...rows].sort((a, b) => {
    const scoreDelta =
      (qualityValue(b) ?? Number.NEGATIVE_INFINITY) -
      (qualityValue(a) ?? Number.NEGATIVE_INFINITY);
    if (scoreDelta !== 0) return scoreDelta;
    const coverageDelta =
      (numeric(b.model_intelligence_coverage) ?? Number.NEGATIVE_INFINITY) -
      (numeric(a.model_intelligence_coverage) ?? Number.NEGATIVE_INFINITY);
    if (coverageDelta !== 0) return coverageDelta;
    return (
      (numeric(a.benchmark_runtime_seconds) ?? Number.POSITIVE_INFINITY) -
      (numeric(b.benchmark_runtime_seconds) ?? Number.POSITIVE_INFINITY)
    );
  })[0]!;
}

function metricColumnsFor(rows: LeaderboardRow[]): MetricColumn[] {
  const columns: MetricColumn[] = [];

  columns.push({
    key: "model_size_gb",
    label: "GB",
    render: (row) => formatModelSize(row),
  });

  if (hasMetric(rows, "model_intelligence_score")) {
    columns.push({
      key: "model_intelligence_score",
      label: "LAIA Index",
      render: (row) => <PointsWithDelta row={row} />,
      primary: true,
    });
  }
  if (hasMetric(rows, "global_mmlu_lite_pass_at_1")) {
    columns.push({
      key: "global_mmlu_lite_pass_at_1",
      label: "Knowledge",
      render: (row) => formatPercent(row.global_mmlu_lite_pass_at_1),
      primary: !hasMetric(rows, "model_intelligence_score"),
    });
  }
  if (hasMetric(rows, "ifbench_prompt_level_loose")) {
    columns.push({
      key: "ifbench_prompt_level_loose",
      label: "Instructions",
      render: (row) => formatPercent(row.ifbench_prompt_level_loose),
      primary: !hasMetric(rows, "global_mmlu_lite_pass_at_1"),
    });
  }
  if (hasMetric(rows, "bfcl_v4_selected_accuracy")) {
    columns.push({
      key: "bfcl_v4_selected_accuracy",
      label: "Tool calling",
      render: (row) => formatPercent(row.bfcl_v4_selected_accuracy),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose"),
    });
  }
  if (hasAnyMetric(rows, ["mmmu_accuracy", "ocrbench_v2_score"])) {
    columns.push({
      key: "vision",
      label: "Vision",
      render: (row) => formatPercent(averageMetric(row, ["mmmu_accuracy", "ocrbench_v2_score"])),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy"),
    });
  }
  if (hasMetric(rows, "mbpp_pass_at_1")) {
    columns.push({
      key: "mbpp_pass_at_1",
      label: "Coding",
      render: (row) => formatPercent(row.mbpp_pass_at_1),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasAnyMetric(rows, ["mmmu_accuracy", "ocrbench_v2_score"]),
    });
  }
  if (hasMetric(rows, "rgb_all_rate")) {
    columns.push({
      key: "rgb_all_rate",
      label: "RAG",
      render: (row) => formatPercent(row.rgb_all_rate),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasAnyMetric(rows, ["mmmu_accuracy", "ocrbench_v2_score"]) &&
        !hasMetric(rows, "mbpp_pass_at_1"),
    });
  }
  if (hasMetric(rows, "simpleqa_f1")) {
    columns.push({
      key: "simpleqa_f1",
      label: "Factuality",
      render: (row) => formatPercent(row.simpleqa_f1),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasAnyMetric(rows, ["mmmu_accuracy", "ocrbench_v2_score"]) &&
        !hasMetric(rows, "mbpp_pass_at_1") &&
        !hasMetric(rows, "rgb_all_rate"),
    });
  }
  if (hasMetric(rows, "harmbench_refusal_rate")) {
    columns.push({
      key: "harmbench_refusal_rate",
      label: "Safety",
      render: (row) => formatPercent(row.harmbench_refusal_rate),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasAnyMetric(rows, ["mmmu_accuracy", "ocrbench_v2_score"]) &&
        !hasMetric(rows, "mbpp_pass_at_1") &&
        !hasMetric(rows, "rgb_all_rate") &&
        !hasMetric(rows, "simpleqa_f1"),
    });
  }
  return columns;
}

function isSyntheticRow(row: LeaderboardRow) {
  if (!row.metadata_json) return false;
  try {
    const metadata = JSON.parse(row.metadata_json) as { synthetic?: unknown };
    return metadata.synthetic === true;
  } catch {
    return row.metadata_json.includes('"synthetic": true');
  }
}

function isSmokeRow(row: LeaderboardRow) {
  if (/\bsmoke\b/i.test(row.variant_name)) return true;
  return (
    qualityValue(row) !== null &&
    row.benchmark_runtime_seconds !== null &&
    row.benchmark_runtime_seconds !== undefined &&
    row.benchmark_runtime_seconds < 60
  );
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function rankRows<T extends LeaderboardRow>(rows: T[]) {
  return [...rows].sort((a, b) => scoreForRank(b) - scoreForRank(a));
}

function sortRowsForDisplay<T extends LeaderboardRow>(rows: T[], sizeSort: SizeSort) {
  if (!sizeSort) return rows;
  return [...rows].sort((a, b) => {
    const aSize = modelSizeGb(a);
    const bSize = modelSizeGb(b);
    if (aSize === null && bSize === null) return scoreForRank(b) - scoreForRank(a);
    if (aSize === null) return 1;
    if (bSize === null) return -1;
    const sizeDelta = sizeSort === "asc" ? aSize - bSize : bSize - aSize;
    if (sizeDelta !== 0) return sizeDelta;
    return scoreForRank(b) - scoreForRank(a);
  });
}

function scoreForRank(row: LeaderboardRow) {
  return qualityValue(row) ?? 0;
}



function numeric(value: unknown) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function hasMetric(rows: LeaderboardRow[], key: keyof LeaderboardRow) {
  return rows.some((row) => numeric(row[key]) !== null);
}

function hasAnyMetric(rows: LeaderboardRow[], keys: Array<keyof LeaderboardRow>) {
  return keys.some((key) => hasMetric(rows, key));
}

function averageMetric(row: LeaderboardRow, keys: Array<keyof LeaderboardRow>) {
  const values = keys
    .map((key) => numeric(row[key]))
    .filter((value): value is number => value !== null);
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}



function qualityValue(row: LeaderboardRow) {
  return (
    numeric(row.model_intelligence_score) ??
    numeric(row.global_mmlu_lite_pass_at_1) ??
    numeric(row.ifbench_prompt_level_loose) ??
    numeric(row.bfcl_v4_selected_accuracy) ??
    numeric(row.ocrbench_v2_score) ??
    numeric(row.mmmu_accuracy) ??
    numeric(row.mbpp_pass_at_1) ??
    numeric(row.rgb_all_rate) ??
    numeric(row.simpleqa_f1) ??
    numeric(row.harmbench_refusal_rate)
  );
}

function hasIntelligenceScore(row?: LeaderboardRow | null) {
  return numeric(row?.model_intelligence_score) !== null;
}

function formatQualityScore(row?: LeaderboardRow | null) {
  if (!row) return "n/a";
  return hasIntelligenceScore(row)
    ? formatPoints(row.model_intelligence_score)
    : formatPercent(qualityValue(row));
}

function secondaryQualityValue(row: LeaderboardRow) {
  return (
    numeric(row.model_intelligence_available_score) ??
    numeric(row.global_mmlu_lite_micro_pass_at_1) ??
    numeric(row.ifbench_instruction_level_loose) ??
    numeric(row.bfcl_v4_non_live_accuracy) ??
    numeric(row.ocrbench_v2_micro_score) ??
    numeric(row.mmmu_multiple_choice_accuracy) ??
    numeric(row.mbpp_compile_rate) ??
    numeric(row.rgb_rejection_rate) ??
    numeric(row.rgb_fact_check_rate) ??
    numeric(row.simpleqa_accuracy_given_attempted) ??
    numeric(row.simpleqa_not_attempted_rate) ??
    numeric(row.harmbench_attack_success_rate) ??
    numeric(row.global_mmlu_lite_invalid_rate)
  );
}

function formatSecondaryScore(row: LeaderboardRow) {
  return numeric(row.model_intelligence_available_score) !== null
    ? formatPoints(row.model_intelligence_available_score)
    : formatPercent(secondaryQualityValue(row));
}

function protocolLabel(row: LeaderboardRow) {
  if (
    row.model_intelligence_score !== null &&
    row.model_intelligence_score !== undefined
  ) {
    return "LAIA Index";
  }
  if (row.global_mmlu_lite_pass_at_1 !== null && row.global_mmlu_lite_pass_at_1 !== undefined) {
    return "Knowledge";
  }
  if (row.ifbench_prompt_level_loose !== null && row.ifbench_prompt_level_loose !== undefined) {
    return "Instructions";
  }
  if (row.bfcl_v4_selected_accuracy !== null && row.bfcl_v4_selected_accuracy !== undefined) {
    return "Tool calling";
  }
  if (row.ocrbench_v2_score !== null && row.ocrbench_v2_score !== undefined) {
    return "OCR";
  }
  if (row.mmmu_accuracy !== null && row.mmmu_accuracy !== undefined) {
    return "Vision reasoning";
  }
  if (row.mbpp_pass_at_1 !== null && row.mbpp_pass_at_1 !== undefined) {
    return "Coding";
  }
  if (row.rgb_all_rate !== null && row.rgb_all_rate !== undefined) {
    return "RAG";
  }
  if (row.simpleqa_f1 !== null && row.simpleqa_f1 !== undefined) {
    return "Factuality";
  }
  if (row.harmbench_refusal_rate !== null && row.harmbench_refusal_rate !== undefined) {
    return "Safety";
  }
  return "Unscored";
}

function safeMetadata(row: LeaderboardRow) {
  if (!row.metadata_json) return null;
  try {
    return JSON.parse(row.metadata_json) as unknown;
  } catch {
    return row.metadata_json;
  }
}

function apiModel(row: LeaderboardRow) {
  const apiModelValue = variantConfigValue(row, "api_model");
  return typeof apiModelValue === "string" ? apiModelValue : null;
}

function displayModelName(row: LeaderboardRow) {
  const apiName = apiModel(row);
  if (apiName) {
    return formatModelName(apiName);
  }
  return formatModelName(row.base_model_name || apiName || row.variant_name);
}

function labKey(row: LeaderboardRow) {
  const source = `${row.family} ${row.base_model_name} ${apiModel(row) ?? ""} ${row.variant_name}`.toLowerCase();
  if (source.includes("nemotron") || source.includes("nvidia")) return "nvidia";
  if (source.includes("qwen") || source.includes("alibaba")) return "qwen";
  if (source.includes("gemma") || source.includes("google")) return "google";
  if (source.includes("llama") || source.includes("meta")) return "meta";
  if (source.includes("mistral") || source.includes("ministral") || source.includes("mixtral")) return "mistral";
  if (source.includes("deepseek")) return "deepseek";
  if (source.includes("falcon") || source.includes("tii") || source.includes("technology innovation institute")) return "TechnologyInnovationINstitute";
  if (source.includes("phi") || source.includes("microsoft")) return "microsoft";
  if (source.includes("granite") || source.includes("ibm")) return "ibm";
  if (source.includes("olmo") || source.includes("ai2") || source.includes("allenai")) return "ai2";
  if (source.includes("openai") || source.includes("gpt-oss")) return "openai";
  if (source.includes("cohere") || source.includes("aya")) return "cohere";
  if (source.includes("liquid") || source.includes("lfm")) return "liquidAI";
  if (source.includes("smollm") || source.includes("hugging face") || source.includes("huggingface")) return "huggingface";
  return slugForAsset(row.family || row.base_model_name || apiModel(row) || row.variant_name);
}

function labInitials(row: LeaderboardRow) {
  const label = labKey(row) || row.family || row.base_model_name || "AI";
  return label.slice(0, 2).toUpperCase();
}

function slugForAsset(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

function quantizationGroupKey(row: LeaderboardRow) {
  return [
    displayModelName(row).toLowerCase(),
    providerLabel(row).toLowerCase(),
    displayParameter(row).toLowerCase(),
  ].join("|");
}

function quantizationKey(row: LeaderboardRow) {
  return slugForAsset(quantizationLabel(row) || row.quantization || "default");
}

function modelSeriesKey(row: LeaderboardRow) {
  return [
    providerLabel(row).toLowerCase(),
    displayModelName(row).toLowerCase(),
    reasoningKey(row),
  ].join("|");
}

function quantizationTone(row: LeaderboardRow) {
  const label = quantizationLabel(row).toLowerCase();
  if (label.includes("32") || label.includes("16")) return "bf16";
  if (label.includes("8")) return "q8";
  if (label.includes("4")) return "q4";
  return "other";
}

function quantizationLabel(row: LeaderboardRow) {
  const source = `${row.quantization} ${row.variant_name} ${apiModel(row) ?? ""} ${row.file_name ?? ""}`.toLowerCase();
  if (/\b(?:fp32|f32|32\s*bit|32b)\b/.test(source)) return "32 bit";
  if (/\b(?:bf16|fp16|f16|16\s*bit|16b)\b/.test(source)) return "16 bit";
  if (/\b(?:q8|q8_0|q8-k|int8|8\s*bit|8bit)\b/.test(source)) return "8 bit";
  if (/\b(?:q6|6\s*bit|6bit)\b/.test(source)) return "6 bit";
  if (/\b(?:q5|5\s*bit|5bit)\b/.test(source)) return "5 bit";
  if (/\b(?:q4|q4_k_m|q4-k-m|4\s*bit|4bit)\b/.test(source)) return "4 bit";
  if (/\bnemotron\b/.test(source) && /:4b\b/.test(source)) return "4 bit";
  if (row.quantization && row.quantization.toUpperCase() !== "SERVER") {
    return titleCaseModelName(row.quantization.replace(/_/g, " "));
  }
  return "Server";
}

function leaderboardRowClass(row: LeaderboardRow) {
  if (isHostedOpenAIRow(row)) return "hosted-openai-row";
  if (!isFourBitRow(row)) return "alternate-version-row";
  return undefined;
}

function isHostedOpenAIRow(row: LeaderboardRow) {
  return providerLabel(row) === "OpenAI";
}

function isFourBitRow(row: LeaderboardRow) {
  return quantizationLabel(row).toLowerCase() === "4 bit";
}

function quantizationRank(row: LeaderboardRow) {
  const label = quantizationLabel(row).toLowerCase();
  if (label.includes("32")) return 320;
  if (label.includes("16")) return 160;
  if (label.includes("8")) return 80;
  if (label.includes("6")) return 60;
  if (label.includes("5")) return 50;
  if (label.includes("4")) return 40;
  return 10;
}

function formatModelName(value: string) {
  const lastSegment = value.split("/").pop() ?? value;
  const withoutQuantSuffix = lastSegment.split("@")[0];
  const lfm = withoutQuantSuffix.match(/\blfm\s*(\d+(?:\.\d+)?)\s*[-_\s]*(\d+(?:\.\d+)?)([bm])\b/i);
  if (lfm) {
    const value = Number(lfm[2]);
    const billionSize = lfm[3].toLowerCase() === "m" ? value / 1000 : value;
    return `LFM ${lfm[1]} ${formatBillionSize(billionSize)}B`;
  }

  const withoutRunSuffix = lastSegment
    .replace(/\b(?:ollama|lm studio|omlx|all languages|smoke|mbpp|full|test)\b/gi, " ")
    .replace(/\breasoning\s+(?:none|off|on|low|medium|high|auto|unset)\b/gi, " ")
    .replace(/@(?:q\d+(?:[_-][a-z0-9]+)*|bf16|fp16|fp32|f16|f32|4bit|8bit|16bit)\b/gi, " ")
    .replace(/\b(?:mlx|bf16|fp16|fp32|q\d+(?:[_-][a-z0-9]+)*|gguf|4bit|8bit|16bit|it|instruct|chat)\b/gi, " ")
    .replace(/[:_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const qwen = withoutRunSuffix.match(/\bqwen\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (qwen) return `Qwen ${qwen[1]} ${qwen[2]}B`;

  const granite = withoutRunSuffix.match(/\bgranite\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (granite) return `Granite ${granite[1]} ${granite[2]}B`;

  const llama = withoutRunSuffix.match(/\bllama\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (llama) return `Llama ${llama[1]} ${llama[2]}B`;

  const olmo = withoutRunSuffix.match(/\bolmo\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (olmo) return `OLMo ${olmo[1]} ${olmo[2]}B`;

  const smollm = withoutRunSuffix.match(/\bsmollm\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)([bm])\b/i);
  if (smollm) {
    const value = Number(smollm[2]);
    const billionSize = smollm[3].toLowerCase() === "m" ? value / 1000 : value;
    return `SmolLM ${smollm[1]} ${formatBillionSize(billionSize)}B`;
  }

  const falcon = withoutRunSuffix.match(/\bfalcon\s*h\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (falcon) return `Falcon H${falcon[1]} ${falcon[2]}B`;

  const ministral = withoutRunSuffix.match(/\bministral\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (ministral) return `Ministral ${ministral[1]} ${ministral[2]}B`;

  const nemotron = withoutRunSuffix.match(/\b(?:nvidia\s*)?nemotron\s*(\d+)?\s*nano\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (nemotron) return `Nemotron ${nemotron[1] ? `${nemotron[1]} ` : ""}Nano ${nemotron[2]}B`;

  const phi = withoutRunSuffix.match(/\bphi\s*(\d+(?:\.\d+)?)\s*mini\b/i);
  if (phi) return `Phi ${phi[1]} Mini`;

  const gemmaEdge = withoutRunSuffix.match(/\bgemma\s*(\d+(?:\.\d+)?)\s*e\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (gemmaEdge) return `Gemma ${gemmaEdge[1]} E${gemmaEdge[2]}B`;

  const gemma = withoutRunSuffix.match(/\bgemma\s*(\d+(?:\.\d+)?)?.*?(\d+(?:\.\d+)?)\s*b\b/i);
  if (gemma) {
    const version = gemma[1] ? ` ${gemma[1]}` : "";
    return `Gemma${version} ${gemma[2]}B`;
  }

  return titleCaseModelName(withoutRunSuffix || value);
}

function formatBillionSize(value: number) {
  if (value < 1) return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return Number.isInteger(value) ? String(value) : value.toFixed(1).replace(/0$/, "");
}

function titleCaseModelName(value: string) {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => {
      if (/^\d+(?:\.\d+)?b$/i.test(part)) return part.toUpperCase();
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(" ");
}

function reasoningLabel(row: LeaderboardRow) {
  const value = reasoningValue(row);
  if (value === "none" || value === "off" || value === "false" || value === "0" || value === "unset") {
    return "Off";
  }
  if (value === "on" || value === "true" || value === "1") return "On";
  return titleCaseModelName(value);
}

function reasoningKey(row: LeaderboardRow) {
  const label = reasoningLabel(row).toLowerCase();
  return label === "off" ? "off" : slugForAsset(label);
}

function reasoningValue(row: LeaderboardRow) {
  const explicit = variantConfigValue(row, "reasoning_effort");
  if (typeof explicit === "string" && explicit.trim()) {
    return explicit.trim().toLowerCase();
  }
  const variantName = variantConfigValue(row, "name");
  const source = `${typeof variantName === "string" ? variantName : ""} ${row.variant_name}`;
  const match = source.match(/\breasoning\s+([a-z0-9_-]+)/i);
  return match?.[1]?.toLowerCase() ?? "none";
}

function modalityLabel(row: LeaderboardRow) {
  const explicit = variantConfigValue(row, "modality");
  if (typeof explicit === "string" && explicit.trim().toLowerCase() !== "auto") {
    return formatModality(explicit);
  }

  const inputModalities = variantConfigValue(row, "input_modalities");
  if (
    Array.isArray(inputModalities) &&
    inputModalities.some((value) => String(value).toLowerCase() === "image")
  ) {
    return "Multimodal";
  }

  if (hasVisionMetric(row)) return "Multimodal";
  return "Text";
}

function formatModality(value: string) {
  const normalized = value.trim().toLowerCase().replace("_", "-");
  if (
    normalized === "vision" ||
    normalized === "visual" ||
    normalized === "image" ||
    normalized === "vl" ||
    normalized === "mm" ||
    normalized === "multimodal" ||
    normalized === "multi-modal"
  ) {
    return "Multimodal";
  }
  return "Text";
}

function hasVisionMetric(row: LeaderboardRow) {
  return (
    row.ocrbench_v2_score !== null &&
    row.ocrbench_v2_score !== undefined
  ) || (
    row.mmmu_accuracy !== null &&
    row.mmmu_accuracy !== undefined
  );
}

function variantConfigValue(row: LeaderboardRow, key: string) {
  const metadata = safeMetadata(row);
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) return null;
  const variantConfig = (metadata as { variant_config?: unknown }).variant_config;
  if (!variantConfig || typeof variantConfig !== "object" || Array.isArray(variantConfig)) {
    return null;
  }
  return (variantConfig as Record<string, unknown>)[key] ?? null;
}

function meetsMaximum(value: number | null | undefined, maximum: number) {
  return value !== null && value !== undefined && Number.isFinite(value) && value <= maximum;
}

function parseNumber(value: string) {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function formatParameter(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${formatBillionSize(value)}B`;
}

function displayParameter(row: LeaderboardRow) {
  if (row.parameter_size_b && row.parameter_size_b > 0) return formatParameter(row.parameter_size_b);
  const modelName = displayModelName(row);
  const match = modelName.match(/(\d+(?:\.\d+)?)B$/i);
  return match ? `${match[1]}B` : "n/a";
}

function providerLabel(row: LeaderboardRow) {
  const source = `${row.family} ${row.base_model_name} ${apiModel(row) ?? ""} ${row.model_repo ?? ""} ${row.variant_name}`.toLowerCase();
  if (source.includes("openai") || /\bgpt-[\w.-]+/.test(source)) return "OpenAI";
  if (source.includes("nemotron") || source.includes("nvidia")) return "NVIDIA";
  if (source.includes("liquid") || source.includes("lfm")) return "Liquid AI";
  if (source.includes("qwen") || source.includes("alibaba")) return "Alibaba";
  if (source.includes("gemma") || source.includes("google")) return "Google";
  if (source.includes("llama") || source.includes("meta")) return "Meta";
  if (source.includes("mistral") || source.includes("ministral") || source.includes("mixtral")) return "Mistral AI";
  if (source.includes("granite") || source.includes("ibm")) return "IBM";
  if (source.includes("olmo")) return "AI2";
  if (source.includes("falcon")) return "TII";
  if (source.includes("smollm")) return "Hugging Face";
  if (source.includes("phi") || source.includes("microsoft")) return "Microsoft";
  if (source.includes("deepseek")) return "DeepSeek";
  if (source.includes("cohere") || source.includes("aya")) return "Cohere";
  return row.family || "Local";
}

function formatModelSize(row: LeaderboardRow) {
  const sizeGb = modelSizeGb(row);
  if (sizeGb === null) return "n/a";
  const prefix = fileSizeBytes(row) === null ? "~" : "";
  const decimals = sizeGb < 10 ? 1 : 0;
  return `${prefix}${sizeGb.toFixed(decimals)} GB`;
}

function modelSizeGb(row: LeaderboardRow) {
  const bytes = fileSizeBytes(row);
  if (bytes !== null) return bytes / 1024 ** 3;
  return estimatedModelSizeGb(row);
}

function fileSizeBytes(row: LeaderboardRow) {
  const topLevelSize = numeric(row.file_size_bytes);
  if (topLevelSize !== null) return topLevelSize;

  const metadata = safeMetadata(row);
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) return null;
  const modelFile = (metadata as { model_file?: unknown }).model_file;
  if (modelFile && typeof modelFile === "object" && !Array.isArray(modelFile)) {
    const modelFileSize = numeric((modelFile as { size_bytes?: unknown }).size_bytes);
    if (modelFileSize !== null) return modelFileSize;
  }

  const variantConfig = (metadata as { variant_config?: unknown }).variant_config;
  if (variantConfig && typeof variantConfig === "object" && !Array.isArray(variantConfig)) {
    const variantSize = numeric((variantConfig as { file_size_bytes?: unknown }).file_size_bytes);
    if (variantSize !== null) return variantSize;
  }

  const apiModelValue = apiModel(row);
  if (apiModelValue && apiModelValue in KNOWN_LM_STUDIO_MODEL_SIZES) {
    return KNOWN_LM_STUDIO_MODEL_SIZES[apiModelValue];
  }

  return null;
}

const KNOWN_LM_STUDIO_MODEL_SIZES: Record<string, number> = {
  "lfm2.5-350m@bf16": 711_500_000,
  "lfm2.5-350m@q8_0": 379_200_000,
  "lfm2.5-350m@q4_k_m": 229_300_000,
  "qwen3.5-0.8b@bf16": 1_700_000_000,
  "qwen3.5-0.8b@q8_0": 1_200_000_000,
  "qwen3.5-0.8b@q4_k_m": 934_900_000,
};

function estimatedModelSizeGb(row: LeaderboardRow) {
  const parameterSize = numeric(row.parameter_size_b);
  const bytesPerParameter = estimatedBytesPerParameter(row);
  if (parameterSize === null || bytesPerParameter === null) return null;
  return (parameterSize * 1_000_000_000 * bytesPerParameter) / 1024 ** 3;
}

function estimatedBytesPerParameter(row: LeaderboardRow) {
  const source = [
    row.quantization,
    row.precision,
    row.variant_name,
    row.base_model_name,
    apiModel(row),
    row.file_name,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  if (/\b(?:fp32|f32|32\s*bit|32b)\b/.test(source)) return 4;
  if (/\b(?:bf16|fp16|f16|16\s*bit|16b)\b/.test(source)) return 2;
  if (/\b(?:q8|int8|8\s*bit|8bit)\b/.test(source)) return 1;
  if (/\b(?:q6|6\s*bit|6bit)\b/.test(source)) return 0.75;
  if (/\b(?:q5|5\s*bit|5bit)\b/.test(source)) return 0.625;
  if (/\b(?:q4|4\s*bit|4bit)\b/.test(source)) return 0.5;
  if (/\b(?:q3|3\s*bit|3bit)\b/.test(source)) return 0.375;
  if (/\b(?:q2|2\s*bit|2bit)\b/.test(source)) return 0.25;
  return null;
}

function formatPercent(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "n/a"
    : `${(value * 100).toFixed(1)}%`;
}

function formatPoints(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "n/a"
    : `${(value * 100).toFixed(1)} pts`;
}

function formatDuration(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

function shortId(value?: string | null) {
  if (!value) return "n/a";
  return value.length > 12 ? `${value.slice(0, 8)}...${value.slice(-4)}` : value;
}
