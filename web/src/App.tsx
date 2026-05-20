import {
  BarChart3,
  BookOpen,
  Braces,
  Brain,
  Code2,
  Database,
  Eye,
  FileText,
  Gauge,
  Info,
  ExternalLink,
  Lightbulb,
  LightbulbOff,
  Search,
  ShieldCheck,
  Table2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";

type LeaderboardRow = {
  [key: string]: unknown;
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
  file_size_bytes?: number | null;
  backend_name?: string | null;
  run_uuid?: string | null;
  metadata_json?: string | null;
  global_mmlu_lite_pass_at_1?: number | null;
  global_mmlu_lite_micro_pass_at_1?: number | null;
  global_mmlu_lite_invalid_rate?: number | null;
  ifbench_prompt_level_loose?: number | null;
  ifbench_instruction_level_loose?: number | null;
  bfcl_v4_selected_accuracy?: number | null;
  bfcl_v4_invalid_rate?: number | null;
  ocrbench_v2_micro_score?: number | null;
  ocrbench_v2_score?: number | null;
  mmmu_accuracy?: number | null;
  mbpp_pass_at_1?: number | null;
  mbpp_invalid_rate?: number | null;
  rgb_all_rate?: number | null;
  rgb_rejection_rate?: number | null;
  rgb_fact_check_rate?: number | null;
  rgb_error_correction_rate?: number | null;
  simpleqa_f1?: number | null;
  harmbench_refusal_rate?: number | null;
  model_intelligence_score?: number | null;
  model_intelligence_coverage?: number | null;
  model_intelligence_available_score?: number | null;
  benchmark_runtime_seconds?: number | null;
  benchmark_samples?: number | null;
  benchmark_total_tokens?: number | null;
  benchmark_prompt_tokens?: number | null;
  benchmark_completion_tokens?: number | null;
  benchmark_reasoning_tokens?: number | null;
  benchmark_total_cost_usd?: number | null;
  benchmark_p95_latency_seconds?: number | null;
  benchmark_output_tokens_per_second?: number | null;
  benchmark_truncated_rate?: number | null;
  benchmark_truncated_count?: number | null;
};

type Payload = {
  generated_at: string;
  tagline: string;
  leaderboard: LeaderboardRow[];
};

type Page = "leaderboard" | "benchmarks" | "efficiency" | "models" | "methodology";

type Filters = {
  query: string;
  family: string;
  parameterSize: string;
};

type Capability = {
  id: string;
  label: string;
  benchmark: string;
  metricLabel: string;
  weight?: number;
  icon: ReactNode;
  value: (row: LeaderboardRow) => number | null;
  description: string;
  includedInLaia: boolean;
};

const PAGE_LABELS: Record<Page, string> = {
  leaderboard: "Leaderboard",
  benchmarks: "Benchmarks",
  efficiency: "Efficiency",
  models: "Models",
  methodology: "Methodology",
};

const emptyFilters: Filters = {
  query: "",
  family: "all",
  parameterSize: "all",
};

const INDEX_PARAMETER_LIMITS = [
  { label: "Any parameters", value: "all" },
  { label: "Up to 1B", value: "1" },
  { label: "Up to 2B", value: "2" },
  { label: "Up to 4B", value: "4" },
  { label: "Up to 8B", value: "8" },
  { label: "Up to 16B", value: "16" },
];

const INDEX_GB_LIMITS = [
  { label: "Any GB", value: "all" },
  { label: "Up to 1 GB", value: "1" },
  { label: "Up to 2 GB", value: "2" },
  { label: "Up to 4 GB", value: "4" },
  { label: "Up to 8 GB", value: "8" },
  { label: "Up to 16 GB", value: "16" },
];

const TEXT_CAPABILITIES: Capability[] = [
  {
    id: "knowledge",
    label: "Knowledge",
    benchmark: "Global MMLU Lite",
    metricLabel: "Pass@1",
    weight: 20,
    icon: <BookOpen size={16} />,
    value: (row) => numeric(row.global_mmlu_lite_pass_at_1),
    description: "Academic and factual breadth across multilingual subjects.",
    includedInLaia: true,
  },
  {
    id: "instructions",
    label: "Instructions",
    benchmark: "IFBench",
    metricLabel: "Prompt-level loose",
    weight: 20,
    icon: <FileText size={16} />,
    value: (row) => numeric(row.ifbench_prompt_level_loose),
    description: "Multi-constraint instruction following.",
    includedInLaia: true,
  },
  {
    id: "tool-calling",
    label: "Tools",
    benchmark: "BFCL v4",
    metricLabel: "Selected accuracy",
    weight: 20,
    icon: <Braces size={16} />,
    value: (row) => numeric(row.bfcl_v4_selected_accuracy),
    description: "Function selection and argument generation.",
    includedInLaia: true,
  },
  {
    id: "coding",
    label: "Coding",
    benchmark: "MBPP",
    metricLabel: "Pass@1",
    weight: 20,
    icon: <Code2 size={16} />,
    value: (row) => numeric(row.mbpp_pass_at_1),
    description: "Python code generation validated by tests.",
    includedInLaia: true,
  },
  {
    id: "rag",
    label: "Grounding",
    benchmark: "RGB",
    metricLabel: "All rate",
    weight: 20,
    icon: <Database size={16} />,
    value: (row) => numeric(row.rgb_all_rate),
    description: "Use retrieved evidence, reject noise, and detect factual errors.",
    includedInLaia: true,
  },
];

const EXTRA_CAPABILITIES: Capability[] = [
  {
    id: "vision",
    label: "Vision",
    benchmark: "MMMU + OCRBench v2",
    metricLabel: "Average score",
    icon: <Eye size={16} />,
    value: (row) => averageMetric(row, ["ocrbench_v2_micro_score", "mmmu_accuracy"]),
    description: "Multimodal visual reasoning and OCR understanding.",
    includedInLaia: false,
  },
  {
    id: "factuality",
    label: "Factuality",
    benchmark: "SimpleQA",
    metricLabel: "F1",
    icon: <Brain size={16} />,
    value: (row) => numeric(row.simpleqa_f1),
    description: "Short-answer factuality scored with a judge.",
    includedInLaia: false,
  },
  {
    id: "safety",
    label: "Safety",
    benchmark: "HarmBench",
    metricLabel: "Refusal rate",
    icon: <ShieldCheck size={16} />,
    value: (row) => numeric(row.harmbench_refusal_rate),
    description: "Refusal behavior on harmful requests.",
    includedInLaia: false,
  },
];

const CAPABILITIES = [...TEXT_CAPABILITIES, ...EXTRA_CAPABILITIES];

const LAIA_INDEX_WEIGHTS = {
  global_mmlu_lite_pass_at_1: 0.2,
  ifbench_prompt_level_loose: 0.2,
  bfcl_v4_selected_accuracy: 0.2,
  mbpp_pass_at_1: 0.2,
  rgb_all_rate: 0.2,
} satisfies Partial<Record<keyof LeaderboardRow, number>>;

const MERGED_BENCHMARK_METRICS = [
  "global_mmlu_lite_pass_at_1",
  "global_mmlu_lite_micro_pass_at_1",
  "global_mmlu_lite_invalid_rate",
  "ifbench_prompt_level_loose",
  "ifbench_instruction_level_loose",
  "bfcl_v4_selected_accuracy",
  "bfcl_v4_invalid_rate",
  "ocrbench_v2_score",
  "ocrbench_v2_micro_score",
  "mmmu_accuracy",
  "mbpp_pass_at_1",
  "mbpp_invalid_rate",
  "rgb_all_rate",
  "rgb_rejection_rate",
  "rgb_fact_check_rate",
  "rgb_error_correction_rate",
  "simpleqa_f1",
  "harmbench_refusal_rate",
] satisfies Array<keyof LeaderboardRow>;

const RAW_TABLE_COLUMNS = [
  "variant_name",
  "family",
  "parameter_size_b",
  "quantization",
  "model_intelligence_score",
  "global_mmlu_lite_pass_at_1",
  "ifbench_prompt_level_loose",
  "bfcl_v4_selected_accuracy",
  "mbpp_pass_at_1",
  "rgb_all_rate",
  "ocrbench_v2_micro_score",
  "mmmu_accuracy",
  "simpleqa_f1",
  "harmbench_refusal_rate",
  "benchmark_samples",
  "benchmark_runtime_seconds",
  "benchmark_total_tokens",
  "benchmark_prompt_tokens",
  "benchmark_completion_tokens",
  "benchmark_output_tokens_per_second",
  "benchmark_truncated_rate",
  "benchmark_total_cost_usd",
  "run_uuid",
];

export function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<Page>("leaderboard");
  const [filters, setFilters] = useState<Filters>(emptyFilters);

  useEffect(() => {
    loadPayload()
      .then(setPayload)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const rawRows = payload?.leaderboard ?? [];
  const publishableRows = useMemo(
    () => rawRows.filter((row) => !isSyntheticRow(row) && !isSmokeRow(row)),
    [rawRows],
  );
  const comparableRows = useMemo(() => buildComparableRows(publishableRows), [publishableRows]);
  const filteredRows = useMemo(
    () => applyFilters(comparableRows, filters),
    [comparableRows, filters],
  );
  const leaderboardRows = useMemo(
    () => filteredRows.filter((row) => isFourBitRow(row) || isHostedOpenAIRow(row)).sort((a, b) => scoreForRank(b) - scoreForRank(a)),
    [filteredRows],
  );
  const options = useMemo(() => optionSets(comparableRows), [comparableRows]);

  if (error) {
    return <StateShell title="Benchmark data failed to load" detail={error} />;
  }

  if (!payload) {
    return <StateShell title="Loading benchmark results" detail="Reading local results.json" />;
  }

  return (
    <main className="app-shell">
      <SiteHeader generatedAt={payload.generated_at} page={page} onNavigate={setPage} />

      <section className="hero-band">
        <div>
          <p className="eyebrow">Local AI Analysis</p>
          <h1>{PAGE_LABELS[page]}</h1>
          <p>
            Compare local and hosted models by LAIA Index, benchmark coverage, quantization, and
            reproducible run metadata. Updated {formatDate(payload.generated_at)}.
          </p>
        </div>
      </section>

      {page === "models" && (
        <FilterPanel filters={filters} options={options} onChange={setFilters} />
      )}

      {page === "leaderboard" && <LeaderboardPage rows={leaderboardRows} />}
      {page === "benchmarks" && <BenchmarksPage rows={leaderboardRows} />}
      {page === "efficiency" && <EfficiencyPage rows={leaderboardRows} />}
      {page === "models" && <ModelsPage rows={leaderboardRows} allRows={publishableRows} />}
      {page === "methodology" && <MethodologyPage />}

      <footer className="footer">
        <span>Local AI Analysis</span>
        <span>{payload.tagline}</span>
      </footer>
    </main>
  );
}

function StateShell({ title, detail }: { title: string; detail: string }) {
  return (
    <main className="state-shell">
      <p className="eyebrow">Local AI Analysis</p>
      <h1>{title}</h1>
      <p>{detail}</p>
    </main>
  );
}

function SiteHeader({
  generatedAt,
  page,
  onNavigate,
}: {
  generatedAt: string;
  page: Page;
  onNavigate: (page: Page) => void;
}) {
  const pages: Page[] = ["leaderboard", "benchmarks", "efficiency", "models", "methodology"];
  return (
    <header className="site-header">
      <button className="site-mark" type="button" onClick={() => onNavigate("leaderboard")}>
        <span>NC</span>
        <strong>Local AI Analysis</strong>
      </button>
      <nav aria-label="Primary navigation">
        {pages.map((item) => (
          <button
            className={page === item ? "nav-active" : ""}
            key={item}
            type="button"
            onClick={() => onNavigate(item)}
          >
            {PAGE_LABELS[item]}
          </button>
        ))}
      </nav>
      <time dateTime={generatedAt}>Updated {formatDate(generatedAt)}</time>
    </header>
  );
}

function FilterPanel({
  filters,
  options,
  onChange,
}: {
  filters: Filters;
  options: ReturnType<typeof optionSets>;
  onChange: (filters: Filters) => void;
}) {
  const quickFamilies = ["OpenAI", "Alibaba", "Google", "Mistral AI", "Meta", "NVIDIA"].filter((option) =>
    options.families.includes(option),
  );
  return (
    <section className="filter-panel" aria-label="Filters">
      <label className="search-field">
        <span>Search</span>
        <div>
          <Search size={16} aria-hidden="true" />
          <input
            type="search"
            value={filters.query}
            placeholder="Model, lab, quantization"
            onChange={(event) => onChange({ ...filters, query: event.target.value })}
          />
        </div>
      </label>
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
      <div className="quick-filters" aria-label="Quick provider filters">
        <button
          className={filters.family === "all" ? "active" : ""}
          type="button"
          onClick={() => onChange({ ...filters, family: "all" })}
        >
          All labs
        </button>
        {quickFamilies.map((family) => (
          <button
            className={filters.family === family ? "active" : ""}
            key={family}
            type="button"
            onClick={() => onChange({ ...filters, family })}
          >
            {family}
          </button>
        ))}
      </div>
    </section>
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
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function LeaderboardPage({ rows }: { rows: LeaderboardRow[] }) {
  const [parameterLimit, setParameterLimit] = useState("all");
  const [gbLimit, setGbLimit] = useState("all");
  const chartRows = useMemo(
    () => topIndexRows(rows, parameterLimit, gbLimit),
    [rows, parameterLimit, gbLimit],
  );
  const maxModelSize = useMemo(
    () => Math.max(...rows.map(modelSizeGb).filter((size): size is number => size !== null), 0.01),
    [rows],
  );

  return (
    <section className="page-grid leaderboard-view">
      <div className="highlights-heading">
        <h2>Highlights</h2>
      </div>
      <div className="highlight-grid single-highlight">
        <IndexPlotCard
          title="Intelligence Index"
          subtitle="Text intelligence points · Higher is better"
          rows={chartRows}
          parameterLimit={parameterLimit}
          gbLimit={gbLimit}
          onParameterLimitChange={setParameterLimit}
          onGbLimitChange={setGbLimit}
        />
      </div>
      <ProviderLegend rows={rows} />
      <LaiaFormulaNote />
      <LandscapeSection rows={rows} />

      <div className="section-heading">
        <div>
          <p className="eyebrow">Capability Ranking</p>
          <h2>Models ranked by LAIA Index</h2>
        </div>
        <p>Default view shows 4-bit local rows and hosted OpenAI references.</p>
      </div>
      <div className="ranking-list">
        {rows.map((row, index) => (
          <LeaderboardRowCard
            row={row}
            rank={index + 1}
            maxModelSizeGb={maxModelSize}
            key={row.normalized_result_id ?? row.variant_id}
          />
        ))}
      </div>
    </section>
  );
}

function LaiaFormulaNote() {
  return (
    <section className="laia-note" aria-label="LAIA Index formula">
      <strong>LAIA Index</strong>
      {TEXT_CAPABILITIES.map((capability) => (
        <span key={capability.id}>{capability.label} {capability.weight} pts</span>
      ))}
      <em>Judge and vision benchmarks are reported separately.</em>
    </section>
  );
}

function LandscapeSection({ rows }: { rows: LeaderboardRow[] }) {
  const points = rows
    .map((row) => ({ row, x: modelSizeGb(row), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  if (points.length < 2) return null;

  return (
    <section className="landscape-section">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Model Landscape</p>
          <h2>Score versus footprint</h2>
        </div>
        <p>Higher and further left is better: stronger LAIA Index with less local memory.</p>
      </div>
      <div className="landscape-grid">
        <SizeIntelligencePlot rows={rows} />
        <CompactEfficiencyRanking rows={rows} />
      </div>
    </section>
  );
}

function SizeIntelligencePlot({ rows }: { rows: LeaderboardRow[] }) {
  const points = rows
    .map((row) => ({ row, x: modelSizeGb(row), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  const width = 620;
  const height = 330;
  const pad = { top: 22, right: 24, bottom: 44, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxX = Math.max(1, Math.ceil(Math.max(...points.map((p) => p.x)) * 1.08));
  const maxY = Math.max(0.12, Math.ceil(Math.max(...points.map((p) => p.y)) * 120) / 100);
  const xFor = (x: number) => pad.left + (x / maxX) * plotW;
  const yFor = (y: number) => pad.top + plotH - (y / maxY) * plotH;
  const labeled = new Set([...points].sort((a, b) => b.y - a.y).slice(0, 7).map((p) => p.row.variant_id));

  return (
    <article className="landscape-panel">
      <div className="panel-heading">
        <span className="metric-icon"><Gauge size={16} /></span>
        <div>
          <h3>Intelligence vs GB</h3>
          <p>Model footprint from exported file size when available.</p>
        </div>
      </div>
      <div className="landscape-scatter">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="LAIA Index versus model size in GB">
          {[0, maxX / 2, maxX].map((tick) => (
            <g key={`landscape-x-${tick}`}>
              <line className="grid-line" x1={xFor(tick)} x2={xFor(tick)} y1={pad.top} y2={pad.top + plotH} />
              <text className="axis-label" x={xFor(tick)} y={height - 16} textAnchor="middle">
                {tick === 0 ? "0" : `${tick.toFixed(tick < 10 ? 1 : 0)} GB`}
              </text>
            </g>
          ))}
          {[0, maxY / 2, maxY].map((tick) => (
            <g key={`landscape-y-${tick}`}>
              <line className="grid-line" x1={pad.left} x2={pad.left + plotW} y1={yFor(tick)} y2={yFor(tick)} />
              <text className="axis-label" x={pad.left - 9} y={yFor(tick) + 4} textAnchor="end">
                {formatPoints(tick)}
              </text>
            </g>
          ))}
          <text className="axis-title" x={pad.left + plotW / 2} y={height - 2} textAnchor="middle">Model size in GB</text>
          <text className="axis-title" x={12} y={pad.top + plotH / 2} textAnchor="middle" transform={`rotate(-90 12 ${pad.top + plotH / 2})`}>
            LAIA Index
          </text>
          {points.map((point) => {
            const isLabeled = labeled.has(point.row.variant_id);
            return (
              <g key={`landscape-${point.row.variant_id}`}>
                <circle
                  className={`scatter-point quant-${quantizationTone(point.row)}`}
                  cx={xFor(point.x)}
                  cy={yFor(point.y)}
                  r={isLabeled ? 6 : 4.5}
                  style={{ fill: providerColor(point.row) }}
                >
                  <title>{displayModelName(point.row)} · {formatPoints(point.y)} · {formatModelSize(point.row)}</title>
                </circle>
                {isLabeled && (
                  <text className="point-label" x={xFor(point.x) + 9} y={yFor(point.y) - 8}>
                    {shortModelLabel(point.row)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </article>
  );
}

function CompactEfficiencyRanking({ rows }: { rows: LeaderboardRow[] }) {
  const items = rows
    .map((row) => {
      const size = modelSizeGb(row);
      const score = numeric(row.model_intelligence_score);
      return size && score ? { row, value: (score * 100) / size } : null;
    })
    .filter((item): item is { row: LeaderboardRow; value: number } => item !== null)
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);
  const max = Math.max(...items.map((item) => item.value), 0.01);

  return (
    <article className="landscape-panel">
      <div className="panel-heading">
        <span className="metric-icon"><BarChart3 size={16} /></span>
        <div>
          <h3>Best points per GB</h3>
          <p>Compact models that preserve benchmark value.</p>
        </div>
      </div>
      <div className="compact-efficiency-list">
        {items.map(({ row, value }, index) => (
          <div className="compact-efficiency-row" key={`compact-eff-${row.variant_id}`}>
            <span className="bar-rank">{String(index + 1).padStart(2, "0")}</span>
            <LabIcon row={row} />
            <div>
              <strong>{displayModelName(row)}</strong>
              <small>{quantizationLabel(row)} · {formatModelSize(row)}</small>
            </div>
            <div className="efficiency-bar"><span style={{ width: `${(value / max) * 100}%` }} /></div>
            <b>{value.toFixed(1)}</b>
          </div>
        ))}
      </div>
    </article>
  );
}

function ProviderLegend({ rows }: { rows: LeaderboardRow[] }) {
  const providers = unique(rows.map(providerLabel)).slice(0, 12);
  return (
    <div className="provider-legend" aria-label="Provider colors">
      {providers.map((provider) => (
        <span key={provider}>
          <i style={{ background: providerColorName(provider) }} aria-hidden="true" />
          {provider}
          {provider === "OpenAI" && <b>closed</b>}
        </span>
      ))}
    </div>
  );
}

function IndexPlotCard({
  title,
  subtitle,
  rows,
  parameterLimit,
  gbLimit,
  onParameterLimitChange,
  onGbLimitChange,
}: {
  title: string;
  subtitle: string;
  rows: LeaderboardRow[];
  parameterLimit: string;
  gbLimit: string;
  onParameterLimitChange: (value: string) => void;
  onGbLimitChange: (value: string) => void;
}) {
  const maxScore = Math.max(...rows.map((row) => numeric(row.model_intelligence_score) ?? 0), 0.01);
  return (
    <section className="index-plot-card intelligence-card">
      <div className="index-plot-heading">
        <div>
          <p className="eyebrow">Top 10</p>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <div className="index-plot-controls" aria-label="Index plot filters">
          <label>
            <span>Parameters</span>
            <select value={parameterLimit} onChange={(event) => onParameterLimitChange(event.target.value)}>
              {INDEX_PARAMETER_LIMITS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Footprint</span>
            <select value={gbLimit} onChange={(event) => onGbLimitChange(event.target.value)}>
              {INDEX_GB_LIMITS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="index-plot-list" aria-label={title}>
        {rows.length === 0 && <p className="empty-note">No rows match the selected size limits.</p>}
        {rows.map((row, index) => {
          const score = numeric(row.model_intelligence_score);
          const height = score === null ? 0 : Math.max(4, (score / maxScore) * 100);
          return (
            <article
              className={`index-plot-column ${rowToneClass(row)}`}
              key={`${title}-${row.variant_id}`}
              style={{ "--provider-color": providerColor(row) } as CSSProperties}
            >
              <div className="index-column-track">
                <span className="index-column-bar">
                  <i
                    style={{
                      height: `${height}%`,
                      background: isHostedOpenAIRow(row) ? "var(--ink)" : providerColor(row),
                    }}
                  />
                </span>
                <strong>{formatIndexNumber(score ?? 0)}</strong>
              </div>
              <div className="index-column-label">
                <LabIcon row={row} />
                <b>{shortModelLabel(row)}</b>
                <span>{quantizationLabel(row)} · {formatModelSize(row)}</span>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function LeaderboardRowCard({
  row,
  rank,
  maxModelSizeGb,
}: {
  row: LeaderboardRow;
  rank: number;
  maxModelSizeGb: number;
}) {
  return (
    <article
      className={`leaderboard-row ${rowToneClass(row)}`}
      style={{ "--provider-color": providerColor(row) } as CSSProperties}
    >
      <div className="rank-number">{String(rank).padStart(2, "0")}</div>
      <ModelIdentity row={row} />
      <div className="model-barplot">
        <CapabilityStrip row={row} compact maxModelSizeGb={maxModelSizeGb} />
      </div>
    </article>
  );
}

function ModelIdentity({ row }: { row: LeaderboardRow }) {
  const reasoningEnabled = reasoningKey(row) !== "off";
  const ReasoningIcon = reasoningEnabled ? Lightbulb : LightbulbOff;
  const openai = isHostedOpenAIRow(row);
  return (
    <div className="model-identity">
      <LabIcon row={row} />
      <div>
        <div className="model-title-line">
          <strong>{displayModelName(row)}</strong>
          <span className={`model-badge quant-${quantizationTone(row)}`}>{quantizationLabel(row)}</span>
          <span
            className={`model-badge icon-only reasoning-badge ${reasoningEnabled ? "on" : "off"}`}
            title={reasoningEnabled ? `Reasoning ${reasoningValue(row)}` : "Reasoning disabled"}
          >
            <ReasoningIcon size={13} aria-hidden="true" />
          </span>
          {openai && <span className="model-badge closed-source">Closed source</span>}
        </div>
        <span>{providerLabel(row)} · {displayParameter(row)}</span>
      </div>
    </div>
  );
}

function CapabilityStrip({
  row,
  compact = false,
  maxModelSizeGb,
}: {
  row: LeaderboardRow;
  compact?: boolean;
  maxModelSizeGb?: number;
}) {
  const size = modelSizeGb(row);
  const items = [
    {
      id: "laia",
      label: "LAIA Index",
      normalized: numeric(row.model_intelligence_score),
      display: (value: number | null) => formatPoints(value),
      tone: "laia",
      aria: `LAIA Index: ${formatPoints(numeric(row.model_intelligence_score))}`,
    },
    ...TEXT_CAPABILITIES.map((capability) => {
      const value = numeric(capability.value(row));
      return {
        id: capability.id,
        label: capability.label,
        normalized: value,
        display: (itemValue: number | null) => formatPercent(itemValue),
        tone: "default",
        aria: `${capability.label}: ${formatPercent(value)}`,
      };
    }),
    {
      id: "size",
      label: "GB",
      normalized: size === null || !maxModelSizeGb ? null : size / maxModelSizeGb,
      display: () => (size === null ? "n/a" : formatModelSize(row)),
      tone: "size",
      aria: `Model footprint: ${formatModelSize(row)}`,
    },
  ];

  return (
    <div className={`capability-strip ${compact ? "compact" : ""}`}>
      {items.map((item) => {
        const normalized = item.normalized;
        const width = normalized === null ? 0 : Math.max(2, Math.min(100, normalized * 100));
        return (
          <div className={`mini-capability mini-${item.tone}`} key={item.id}>
            <span>{item.label}</span>
            <div className="mini-bar-track" aria-label={item.aria}>
              <div
                className={`mini-bar-fill ${normalized === null ? "missing" : ""}`}
                style={{ width: `${width}%` }}
              >
              </div>
              <b>{item.display(normalized)}</b>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function BenchmarksPage({ rows }: { rows: LeaderboardRow[] }) {
  return (
    <section className="page-grid">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Benchmark Barplots</p>
          <h2>Capability by capability</h2>
        </div>
        <p>Human labels are primary; benchmark names stay visible for reproducibility.</p>
      </div>
      {CAPABILITIES.map((capability) => (
        <BenchmarkBarChart capability={capability} rows={rows} key={capability.id} />
      ))}
    </section>
  );
}

function BenchmarkBarChart({ capability, rows }: { capability: Capability; rows: LeaderboardRow[] }) {
  const chartRows = rows
    .map((row) => ({ row, value: capability.value(row) }))
    .filter((item): item is { row: LeaderboardRow; value: number } => item.value !== null)
    .sort((a, b) => b.value - a.value)
    .slice(0, 18);

  return (
    <section className="chart-card">
      <div className="chart-card-heading">
        <span className="metric-icon">{capability.icon}</span>
        <div>
          <h3>{capability.label}</h3>
          <p>{capability.benchmark} · {capability.metricLabel}</p>
        </div>
        {capability.weight ? <b>{capability.weight} LAIA pts</b> : <b>Separate</b>}
      </div>
      <p className="chart-description">{capability.description}</p>
      <div className="bar-list">
        {chartRows.length ? chartRows.map(({ row, value }, index) => (
          <BenchmarkBar row={row} value={value} index={index} key={`${capability.id}-${row.variant_id}`} />
        )) : <p className="empty-note">No completed results yet.</p>}
      </div>
    </section>
  );
}

function BenchmarkBar({ row, value, index }: { row: LeaderboardRow; value: number; index: number }) {
  return (
    <div className={`benchmark-bar-row ${rowToneClass(row)}`}>
      <span className="bar-rank">{String(index + 1).padStart(2, "0")}</span>
      <ModelIdentity row={row} />
      <ScoreBar value={value} tone="benchmark" />
      <strong>{formatPercent(value)}</strong>
      <span>{formatSamples(row)}</span>
      <span>{formatTruncation(row)}</span>
    </div>
  );
}

function EfficiencyPage({ rows }: { rows: LeaderboardRow[] }) {
  return (
    <section className="page-grid">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Efficiency</p>
          <h2>Score, size, and quantization</h2>
        </div>
        <p>Use this page to see whether precision or memory footprint is winning.</p>
      </div>
      <EfficiencyScatter rows={rows} />
      <QuantizationLadder rows={rows} />
      <EfficiencyBars rows={rows} />
    </section>
  );
}

function EfficiencyScatter({ rows }: { rows: LeaderboardRow[] }) {
  const points = rows
    .map((row) => ({ row, x: modelSizeGb(row), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  if (points.length < 2) return null;

  const width = 920;
  const height = 390;
  const pad = { top: 26, right: 34, bottom: 48, left: 56 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxX = Math.max(1, Math.ceil(Math.max(...points.map((p) => p.x)) * 1.1));
  const maxY = Math.max(0.12, Math.ceil(Math.max(...points.map((p) => p.y)) * 120) / 100);
  const xFor = (x: number) => pad.left + (x / maxX) * plotW;
  const yFor = (y: number) => pad.top + plotH - (y / maxY) * plotH;
  const labeled = new Set([...points].sort((a, b) => b.y - a.y).slice(0, 6).map((p) => p.row.variant_id));

  return (
    <section className="chart-card">
      <div className="chart-card-heading">
        <span className="metric-icon"><Gauge size={16} /></span>
        <div>
          <h3>LAIA Index vs model size</h3>
          <p>Memory footprint on x-axis, capability points on y-axis.</p>
        </div>
      </div>
      <div className="scatter-shell">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="LAIA Index versus model size">
          {[0, maxX / 2, maxX].map((tick) => (
            <g key={`x-${tick}`}>
              <line className="grid-line" x1={xFor(tick)} x2={xFor(tick)} y1={pad.top} y2={pad.top + plotH} />
              <text className="axis-label" x={xFor(tick)} y={height - 18} textAnchor="middle">
                {tick === 0 ? "0" : `${tick.toFixed(tick < 10 ? 1 : 0)} GB`}
              </text>
            </g>
          ))}
          {[0, maxY / 2, maxY].map((tick) => (
            <g key={`y-${tick}`}>
              <line className="grid-line" x1={pad.left} x2={pad.left + plotW} y1={yFor(tick)} y2={yFor(tick)} />
              <text className="axis-label" x={pad.left - 10} y={yFor(tick) + 4} textAnchor="end">
                {formatPoints(tick)}
              </text>
            </g>
          ))}
          {points.map((point) => {
            const isLabeled = labeled.has(point.row.variant_id);
            return (
              <g key={point.row.variant_id}>
                <circle className={`scatter-point quant-${quantizationTone(point.row)}`} cx={xFor(point.x)} cy={yFor(point.y)} r={isLabeled ? 6 : 4.5}>
                  <title>{displayModelName(point.row)} · {formatPoints(point.y)} · {formatModelSize(point.row)}</title>
                </circle>
                {isLabeled && (
                  <text className="point-label" x={xFor(point.x) + 9} y={yFor(point.y) - 8}>
                    {displayModelName(point.row)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}

function QuantizationLadder({ rows }: { rows: LeaderboardRow[] }) {
  const groups = Array.from(groupBy(rows, quantizationGroupKey).values())
    .map((groupRows) => groupRows.sort((a, b) => quantizationRank(a) - quantizationRank(b)))
    .filter((groupRows) => groupRows.length > 1)
    .slice(0, 14);

  return (
    <section className="chart-card">
      <div className="chart-card-heading">
        <span className="metric-icon"><BarChart3 size={16} /></span>
        <div>
          <h3>Quantization ladder</h3>
          <p>Same model family across available precision levels.</p>
        </div>
      </div>
      <div className="ladder-list">
        {groups.length ? groups.map((groupRows) => (
          <div className="ladder-row" key={quantizationGroupKey(groupRows[0])}>
            <ModelIdentity row={groupRows[0]} />
            <div className="ladder-track">
              {groupRows.map((row) => (
                <div className={`ladder-chip quant-${quantizationTone(row)}`} key={row.variant_id}>
                  <span>{quantizationLabel(row)}</span>
                  <b>{formatPoints(row.model_intelligence_score)}</b>
                  <small>{formatModelSize(row)}</small>
                </div>
              ))}
            </div>
          </div>
        )) : <p className="empty-note">No multi-quantization groups visible.</p>}
      </div>
    </section>
  );
}

function EfficiencyBars({ rows }: { rows: LeaderboardRow[] }) {
  const items = rows
    .map((row) => {
      const size = modelSizeGb(row);
      const score = numeric(row.model_intelligence_score);
      return size && score ? { row, value: (score * 100) / size } : null;
    })
    .filter((item): item is { row: LeaderboardRow; value: number } => item !== null)
    .sort((a, b) => b.value - a.value)
    .slice(0, 14);
  const max = Math.max(...items.map((item) => item.value), 0.01);

  return (
    <section className="chart-card">
      <div className="chart-card-heading">
        <span className="metric-icon"><Gauge size={16} /></span>
        <div>
          <h3>Best score per GB</h3>
          <p>Higher means more benchmark value per local footprint.</p>
        </div>
      </div>
      <div className="efficiency-bars">
        {items.map(({ row, value }) => (
          <div className="efficiency-row" key={`eff-${row.variant_id}`}>
            <ModelIdentity row={row} />
            <div className="efficiency-bar"><span style={{ width: `${(value / max) * 100}%` }} /></div>
            <strong>{value.toFixed(1)} pts/GB</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function ModelsPage({ rows, allRows }: { rows: LeaderboardRow[]; allRows: LeaderboardRow[] }) {
  const [selectedRow, setSelectedRow] = useState<LeaderboardRow | null>(null);
  const tableRows = rows.length ? rows : allRows;
  return (
    <section className="page-grid models-page">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Model Registry</p>
          <h2>Models, quantization, and run coverage</h2>
        </div>
        <p>Every visible model version with source links, benchmark coverage, and raw metrics for verification.</p>
      </div>

      <div className="model-registry">
        {tableRows.map((row) => (
          <article
            className={`model-registry-row ${rowToneClass(row)}`}
            key={`registry-${row.normalized_result_id ?? row.variant_id}`}
            style={{ "--provider-color": providerColor(row) } as CSSProperties}
          >
            <ModelIdentity row={row} />
            <div className="registry-metrics">
              <MetricPill label="Size" value={formatModelSize(row)} />
              <MetricPill label="LAIA" value={formatPoints(numeric(row.model_intelligence_score))} />
              <MetricPill label="Coverage" value={coverageLabel(row)} />
              <MetricPill label="Truncation" value={formatTruncation(row).replace("trunc ", "")} />
            </div>
            <BenchmarkCoverage row={row} />
            <div className="registry-actions">
              <ModelSourceLink row={row} />
              <button className="details-button" type="button" onClick={() => setSelectedRow(row)}>
                <Info size={14} aria-hidden="true" />
                Details
              </button>
            </div>
          </article>
        ))}
      </div>

      {selectedRow && <ModelDetailsDrawer row={selectedRow} onClose={() => setSelectedRow(null)} />}

      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Full Data Table</p>
          <h2>Raw exported metrics</h2>
        </div>
        <p>Column-level view for debugging benchmark output, tokens, cost, and truncation.</p>
      </div>
      <div className="full-table-shell">
        <table>
          <thead>
            <tr>
              {RAW_TABLE_COLUMNS.map((column) => <th key={column}>{humanizeColumn(column)}</th>)}
            </tr>
          </thead>
          <tbody>
            {tableRows.map((row) => (
              <tr className={rowToneClass(row)} key={row.normalized_result_id ?? row.variant_id}>
                {RAW_TABLE_COLUMNS.map((column) => (
                  <td key={column}>{formatCell(row[column])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MetricPill({ label, value }: { label: string; value: string }) {
  return (
    <span className="metric-pill">
      <small>{label}</small>
      <b>{value}</b>
    </span>
  );
}

function BenchmarkCoverage({ row }: { row: LeaderboardRow }) {
  return (
    <div className="benchmark-coverage" aria-label="Benchmark coverage">
      {CAPABILITIES.map((capability) => {
        const complete = capability.value(row) !== null;
        return (
          <span className={complete ? "complete" : "missing"} key={`coverage-${row.variant_id}-${capability.id}`}>
            {capability.label}
          </span>
        );
      })}
    </div>
  );
}

function ModelSourceLink({ row }: { row: LeaderboardRow }) {
  const link = modelSourceLink(row);
  if (!link.href) {
    return <span className="source-link disabled">{link.label}</span>;
  }
  return (
    <a className={`source-link ${link.kind}`} href={link.href} target="_blank" rel="noreferrer">
      {link.label}
      <ExternalLink size={13} aria-hidden="true" />
    </a>
  );
}

function ModelDetailsDrawer({ row, onClose }: { row: LeaderboardRow; onClose: () => void }) {
  const link = modelSourceLink(row);
  return (
    <div className="drawer-backdrop" role="presentation">
      <aside className="model-drawer" aria-label={`${displayModelName(row)} details`}>
        <div className="drawer-heading">
          <ModelIdentity row={row} />
          <button className="drawer-close" type="button" onClick={onClose} aria-label="Close model details">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="drawer-section">
          <h3>Model</h3>
          <dl>
            <DetailItem label="Provider" value={providerLabel(row)} />
            <DetailItem label="Parameters" value={displayParameter(row)} />
            <DetailItem label="Quantization" value={quantizationLabel(row)} />
            <DetailItem label="File size" value={formatModelSize(row)} />
            <DetailItem label="Backend" value={String(row.backend_name ?? "n/a")} />
            <DetailItem label="Reasoning" value={reasoningKey(row) === "off" ? "Disabled" : reasoningValue(row)} />
            <DetailItem label="API model" value={apiModel(row) ?? "n/a"} />
            <DetailItem label="Source" value={link.label} />
          </dl>
          {link.href && (
            <a className={`source-link ${link.kind}`} href={link.href} target="_blank" rel="noreferrer">
              Open source link
              <ExternalLink size={13} aria-hidden="true" />
            </a>
          )}
        </div>
        <div className="drawer-section">
          <h3>Benchmark Coverage</h3>
          <BenchmarkCoverage row={row} />
        </div>
        <div className="drawer-section">
          <h3>Run Signals</h3>
          <dl>
            <DetailItem label="Samples" value={formatSamples(row).replace(" samples", "")} />
            <DetailItem label="Runtime" value={formatSeconds(numeric(row.benchmark_runtime_seconds))} />
            <DetailItem label="Total tokens" value={formatCount(numeric(row.benchmark_total_tokens))} />
            <DetailItem label="Prompt tokens" value={formatCount(numeric(row.benchmark_prompt_tokens))} />
            <DetailItem label="Completion tokens" value={formatCount(numeric(row.benchmark_completion_tokens))} />
            <DetailItem label="Reasoning tokens" value={formatCount(numeric(row.benchmark_reasoning_tokens))} />
            <DetailItem label="Output tok/s" value={formatNumber(numeric(row.benchmark_output_tokens_per_second))} />
            <DetailItem label="P95 latency" value={formatSeconds(numeric(row.benchmark_p95_latency_seconds))} />
            <DetailItem label="Truncated" value={formatTruncation(row).replace("trunc ", "")} />
            <DetailItem label="Total cost" value={formatUsd(numeric(row.benchmark_total_cost_usd))} />
          </dl>
        </div>
      </aside>
    </div>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}

function MethodologyPage() {
  return (
    <section className="methodology-page">
      <div className="method-hero">
        <p className="eyebrow">Methodology</p>
        <h2>Text capability first. Judge and vision results stay separate.</h2>
        <p>
          LAIA Index is a 100-point text-model score built only from non-judge benchmarks.
          Vision, factuality, and safety are reported as companion capabilities to avoid mixing
          different evaluation assumptions.
        </p>
      </div>

      <div className="method-grid">
        {CAPABILITIES.map((capability) => (
          <article className="method-card" key={capability.id}>
            <span className="metric-icon">{capability.icon}</span>
            <div>
              <h3>{capability.label}</h3>
              <p>{capability.benchmark} · {capability.metricLabel}</p>
              <small>{capability.description}</small>
            </div>
            <b>{capability.includedInLaia ? `${capability.weight} pts` : "Separate"}</b>
          </article>
        ))}
      </div>

      <section className="method-section">
        <div>
          <p className="eyebrow">LAIA Formula</p>
          <h3>100 points from non-judge text benchmarks</h3>
        </div>
        <div className="formula-grid">
          {TEXT_CAPABILITIES.map((capability) => (
            <article key={`formula-${capability.id}`}>
              <span>{capability.weight} pts</span>
              <strong>{capability.label}</strong>
              <small>{capability.benchmark} · {capability.metricLabel}</small>
            </article>
          ))}
        </div>
      </section>

      <section className="method-section">
        <div>
          <p className="eyebrow">Interpretation</p>
          <h3>What is included and what is not</h3>
        </div>
        <div className="method-copy-grid">
          <article>
            <h4>Included in LAIA</h4>
            <p>Global MMLU Lite, IFBench, BFCL v4, MBPP, and RGB are treated as the core text suite. They do not require an external LLM judge for the final score.</p>
          </article>
          <article>
            <h4>Reported separately</h4>
            <p>Vision, factuality, and safety have different evaluation assumptions. They remain visible in benchmark pages without changing the headline LAIA Index.</p>
          </article>
          <article>
            <h4>Quantization</h4>
            <p>4-bit, 8-bit, and 16-bit/BF16 rows are separate model versions. The public default shows 4-bit local models plus hosted references; all versions can be revealed.</p>
          </article>
          <article>
            <h4>Reproducibility</h4>
            <p>The site surfaces sample counts, truncation, token usage, runtime, and source metadata so unusual runs can be inspected instead of hidden.</p>
          </article>
        </div>
      </section>
    </section>
  );
}

function ScoreBar({ value, tone = "default" }: { value?: number | null; tone?: string }) {
  const normalized = numeric(value);
  const width = normalized === null ? 0 : Math.max(2, Math.min(100, normalized * 100));
  return (
    <span className={`score-bar tone-${tone}`} aria-label={formatPercent(normalized)}>
      <i style={{ width: `${width}%` }} />
    </span>
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
    `/labs/${key}.jpeg`,
    `/labs/${key}.webp`,
  ];
  const src = candidates[candidateIndex];
  if (!key || candidateIndex >= candidates.length) {
    return <span className="lab-icon fallback">{labInitials(row)}</span>;
  }
  return (
    <span className={`lab-icon lab-${key}`}>
      <img src={src} alt="" onError={() => setCandidateIndex((index) => index + 1)} />
    </span>
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

function buildComparableRows(rows: LeaderboardRow[]) {
  return Array.from(groupBy(rows, quantizationGroupKey).values()).flatMap((groupRows) => {
    const byQuant = groupBy(groupRows, quantizationKey);
    return Array.from(byQuant.values()).map(mergeComparableRuns);
  });
}

function mergeComparableRuns(rows: LeaderboardRow[]) {
  const base = bestComparableRun(rows);
  const merged: LeaderboardRow = { ...base };
  for (const key of MERGED_BENCHMARK_METRICS) {
    if (numeric(merged[key]) !== null) continue;
    const source = rows.find((row) => numeric(row[key]) !== null);
    if (source) merged[key] = source[key];
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
  for (const [key, weight] of Object.entries(LAIA_INDEX_WEIGHTS) as Array<[keyof typeof LAIA_INDEX_WEIGHTS, number]>) {
    const value = numeric(row[key]);
    if (value === null) continue;
    weightedSum += Math.max(0, Math.min(1, value)) * weight;
    coveredWeight += weight;
  }
  if (coveredWeight <= 0) return { score: null, coverage: null, availableScore: null };
  return { score: weightedSum, coverage: coveredWeight, availableScore: weightedSum / coveredWeight };
}

function bestComparableRun(rows: LeaderboardRow[]) {
  return [...rows].sort((a, b) => {
    const scoreDelta = scoreForRank(b) - scoreForRank(a);
    if (scoreDelta !== 0) return scoreDelta;
    return (numeric(b.model_intelligence_coverage) ?? -1) - (numeric(a.model_intelligence_coverage) ?? -1);
  })[0]!;
}

function applyFilters(rows: LeaderboardRow[], filters: Filters) {
  const query = filters.query.trim().toLowerCase();
  return rows.filter((row) => {
    const searchable = [
      row.variant_name,
      row.base_model_name,
      row.family,
      providerLabel(row),
      displayModelName(row),
      quantizationLabel(row),
      formatModelSize(row),
    ].join(" ").toLowerCase();
    return (
      (!query || searchable.includes(query)) &&
      (filters.family === "all" || row.family === filters.family || providerLabel(row) === filters.family) &&
      (filters.parameterSize === "all" || String(row.parameter_size_b) === filters.parameterSize)
    );
  });
}

function topIndexRows(rows: LeaderboardRow[], parameterLimit: string, gbLimit: string) {
  const maxParameters = parameterLimit === "all" ? null : Number(parameterLimit);
  const maxGb = gbLimit === "all" ? null : Number(gbLimit);
  return rows
    .filter((row) => {
      const score = numeric(row.model_intelligence_score);
      const parameters = numeric(row.parameter_size_b);
      const size = modelSizeGb(row);
      return (
        score !== null &&
        score > 0 &&
        (maxParameters === null || (parameters !== null && parameters <= maxParameters)) &&
        (maxGb === null || (size !== null && size <= maxGb))
      );
    })
    .sort((a, b) => scoreForRank(b) - scoreForRank(a))
    .slice(0, 10);
}

function optionSets(rows: LeaderboardRow[]) {
  return {
    families: unique(rows.flatMap((row) => [row.family, providerLabel(row)])),
    parameterSizes: unique(rows.map((row) => String(row.parameter_size_b))),
  };
}

function groupBy<T>(items: T[], keyFor: (item: T) => string) {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const key = keyFor(item);
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  return groups;
}

function scoreForRank(row: LeaderboardRow) {
  return numeric(row.model_intelligence_score) ?? numeric(row.global_mmlu_lite_pass_at_1) ?? numeric(row.rgb_all_rate) ?? 0;
}

function rowToneClass(row: LeaderboardRow) {
  if (isHostedOpenAIRow(row)) return "hosted-openai-row";
  if (!isFourBitRow(row)) return "alternate-version-row";
  return "";
}

function isHostedOpenAIRow(row: LeaderboardRow) {
  return providerLabel(row) === "OpenAI";
}

function isFourBitRow(row: LeaderboardRow) {
  return quantizationLabel(row).toLowerCase() === "4 bit";
}

function isSyntheticRow(row: LeaderboardRow) {
  if (!row.metadata_json) return false;
  return row.metadata_json.includes('"synthetic": true');
}

function isSmokeRow(row: LeaderboardRow) {
  if (/\bsmoke\b/i.test(row.variant_name)) return true;
  const samples = numeric(row.benchmark_samples);
  return samples !== null && samples <= 5;
}

function numeric(value: unknown) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function averageMetric(row: LeaderboardRow, keys: string[]) {
  const values = keys.map((key) => numeric(row[key])).filter((value): value is number => value !== null);
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function safeMetadata(row: LeaderboardRow) {
  if (!row.metadata_json) return null;
  try {
    return JSON.parse(row.metadata_json) as unknown;
  } catch {
    return null;
  }
}

function variantConfigValue(row: LeaderboardRow, key: string) {
  const metadata = safeMetadata(row);
  if (!metadata || typeof metadata !== "object" || Array.isArray(metadata)) return null;
  const variantConfig = (metadata as { variant_config?: unknown }).variant_config;
  if (!variantConfig || typeof variantConfig !== "object" || Array.isArray(variantConfig)) return null;
  return (variantConfig as Record<string, unknown>)[key] ?? null;
}

function apiModel(row: LeaderboardRow) {
  const value = variantConfigValue(row, "api_model");
  return typeof value === "string" ? value : null;
}

function displayModelName(row: LeaderboardRow) {
  const apiName = apiModel(row);
  return formatModelName(apiName || row.base_model_name || row.variant_name);
}

function shortModelLabel(row: LeaderboardRow) {
  return displayModelName(row)
    .replace(/\s+Instruct\b/i, "")
    .replace(/\s+Reasoning\b/i, "");
}

function formatModelName(value: string) {
  const lastSegment = value.split("/").pop() ?? value;
  const withoutQuantSuffix = lastSegment.split("@")[0];
  if (/nemotron[-_\s]*3[-_\s]*nano/i.test(lastSegment)) return "NVIDIA Nemotron 3 Nano 4B";
  const lfm = withoutQuantSuffix.match(/\blfm\s*(\d+(?:\.\d+)?)\s*[-_\s]*(\d+(?:\.\d+)?)([bm])\b/i);
  if (lfm) return `LFM ${lfm[1]} ${formatBillionSize(Number(lfm[2]) / (lfm[3].toLowerCase() === "m" ? 1000 : 1))}B`;
  const clean = lastSegment
    .replace(/@(?:q\d+(?:[_-][a-z0-9]+)*|bf16|fp16|fp32|f16|f32|4bit|8bit|16bit)\b/gi, " ")
    .replace(/\b(?:ollama|lm studio|omlx|all languages|smoke|reasoning|none|server|mlx|gguf|bf16|fp16|q\d+(?:[_-][a-z0-9]+)*|it|instruct|chat)\b/gi, " ")
    .replace(/[:_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const qwen = clean.match(/\bqwen\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (qwen) return `Qwen ${qwen[1]} ${qwen[2]}B`;
  const gemmaEdge = clean.match(/\bgemma\s*(\d+(?:\.\d+)?)\s*e\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (gemmaEdge) return `Gemma ${gemmaEdge[1]} E${gemmaEdge[2]}B`;
  const generic = clean.match(/\b([a-z]+(?:\s+h)?)(?:\s+)?(\d+(?:\.\d+)?)?(?:\s+)?(\d+(?:\.\d+)?)([bm])\b/i);
  if (generic) {
    const family = titleCaseModelName(generic[1].trim());
    const version = generic[2] ? ` ${generic[2]}` : "";
    const size = generic[4].toLowerCase() === "m" ? Number(generic[3]) / 1000 : Number(generic[3]);
    return `${family}${version} ${formatBillionSize(size)}B`;
  }
  return titleCaseModelName(clean || value);
}

function titleCaseModelName(value: string) {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => (/^\d+(?:\.\d+)?b$/i.test(part) ? part.toUpperCase() : part.charAt(0).toUpperCase() + part.slice(1)))
    .join(" ");
}

function displayParameter(row: LeaderboardRow) {
  if (row.parameter_size_b && row.parameter_size_b > 0) return `${formatBillionSize(row.parameter_size_b)}B`;
  const match = displayModelName(row).match(/(\d+(?:\.\d+)?)B$/i);
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
  if (source.includes("mistral") || source.includes("ministral")) return "Mistral AI";
  if (source.includes("granite") || source.includes("ibm")) return "IBM";
  if (source.includes("olmo") || source.includes("allenai")) return "AI2";
  if (source.includes("falcon") || source.includes("tii")) return "TII";
  if (source.includes("smollm")) return "Hugging Face";
  if (source.includes("phi") || source.includes("microsoft")) return "Microsoft";
  return row.family || "Local";
}

function providerColor(row: LeaderboardRow) {
  return providerColorName(providerLabel(row));
}

function providerColorName(provider: string) {
  const colors: Record<string, string> = {
    OpenAI: "#111111",
    Alibaba: "#6d4cff",
    Google: "#63b35d",
    "Mistral AI": "#ff8a2a",
    Meta: "#4f8dff",
    NVIDIA: "#76b900",
    "Liquid AI": "#f2c94c",
    IBM: "#6f84a3",
    AI2: "#b06df5",
    TII: "#e56f52",
    Microsoft: "#2f80ed",
    "Hugging Face": "#b88700",
  };
  return colors[provider] ?? "#7a7a74";
}

function labKey(row: LeaderboardRow) {
  const source = `${providerLabel(row)} ${row.family} ${row.base_model_name} ${apiModel(row) ?? ""} ${row.variant_name}`.toLowerCase();
  if (source.includes("nvidia") || source.includes("nemotron")) return "nvidia";
  if (source.includes("qwen") || source.includes("alibaba")) return "qwen";
  if (source.includes("google") || source.includes("gemma")) return "google";
  if (source.includes("meta") || source.includes("llama")) return "meta";
  if (source.includes("mistral") || source.includes("ministral")) return "mistral";
  if (source.includes("tii") || source.includes("falcon")) return "TechnologyInnovationINstitute";
  if (source.includes("microsoft") || source.includes("phi")) return "microsoft";
  if (source.includes("ibm") || source.includes("granite")) return "ibm";
  if (source.includes("ai2") || source.includes("olmo")) return "ai2";
  if (source.includes("openai")) return "openai";
  if (source.includes("liquid") || source.includes("lfm")) return "liquidAI";
  if (source.includes("hugging") || source.includes("smollm")) return "huggingface";
  return slugForAsset(row.family || row.base_model_name || "ai");
}

function labInitials(row: LeaderboardRow) {
  return (providerLabel(row) || row.family || "AI").slice(0, 2).toUpperCase();
}

function quantizationLabel(row: LeaderboardRow) {
  const source = `${row.quantization} ${row.variant_name} ${apiModel(row) ?? ""} ${row.file_name ?? ""}`.toLowerCase();
  if (/\b(?:fp32|f32|32\s*bit|32b)\b/.test(source)) return "32 bit";
  if (/\b(?:bf16|fp16|f16|16\s*bit|16b)\b/.test(source)) return "16 bit";
  if (/\b(?:q8|q8_0|int8|8\s*bit|8bit)\b/.test(source)) return "8 bit";
  if (/\b(?:q6|6\s*bit|6bit)\b/.test(source)) return "6 bit";
  if (/\b(?:q5|5\s*bit|5bit)\b/.test(source)) return "5 bit";
  if (/\b(?:q4|q4_k_m|4\s*bit|4bit)\b/.test(source)) return "4 bit";
  if (row.quantization && row.quantization.toUpperCase() !== "SERVER") return titleCaseModelName(row.quantization.replace(/_/g, " "));
  return "Server";
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

function quantizationTone(row: LeaderboardRow) {
  const label = quantizationLabel(row).toLowerCase();
  if (label.includes("16") || label.includes("32")) return "bf16";
  if (label.includes("8")) return "q8";
  if (label.includes("4")) return "q4";
  return "other";
}

function quantizationKey(row: LeaderboardRow) {
  return slugForAsset(quantizationLabel(row) || row.quantization || "default");
}

function quantizationGroupKey(row: LeaderboardRow) {
  return [displayModelName(row).toLowerCase(), providerLabel(row).toLowerCase(), displayParameter(row).toLowerCase()].join("|");
}

function reasoningValue(row: LeaderboardRow) {
  const explicit = variantConfigValue(row, "reasoning_effort");
  if (typeof explicit === "string" && explicit.trim()) return explicit.trim().toLowerCase();
  const match = row.variant_name.match(/\breasoning\s+([a-z0-9_-]+)/i);
  return match?.[1]?.toLowerCase() ?? "none";
}

function reasoningKey(row: LeaderboardRow) {
  const value = reasoningValue(row);
  return ["none", "off", "false", "0", "unset"].includes(value) ? "off" : slugForAsset(value);
}

function fileSizeBytes(row: LeaderboardRow) {
  const topLevelSize = numeric(row.file_size_bytes);
  if (topLevelSize !== null) return topLevelSize;
  const metadata = safeMetadata(row);
  if (metadata && typeof metadata === "object" && !Array.isArray(metadata)) {
    const modelFile = (metadata as { model_file?: { size_bytes?: unknown } }).model_file;
    const modelFileSize = numeric(modelFile?.size_bytes);
    if (modelFileSize !== null) return modelFileSize;
  }
  const apiModelValue = apiModel(row);
  if (apiModelValue && apiModelValue in KNOWN_LM_STUDIO_MODEL_SIZES) return KNOWN_LM_STUDIO_MODEL_SIZES[apiModelValue];
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

function modelSizeGb(row: LeaderboardRow) {
  const bytes = fileSizeBytes(row);
  if (bytes !== null) return bytes / 1024 ** 3;
  const parameterSize = numeric(row.parameter_size_b);
  const bytesPerParameter = estimatedBytesPerParameter(row);
  if (parameterSize === null || bytesPerParameter === null) return null;
  return (parameterSize * 1_000_000_000 * bytesPerParameter) / 1024 ** 3;
}

function estimatedBytesPerParameter(row: LeaderboardRow) {
  const source = `${row.quantization} ${row.precision ?? ""} ${row.variant_name} ${apiModel(row) ?? ""}`.toLowerCase();
  if (/\b(?:bf16|fp16|f16|16\s*bit|16b)\b/.test(source)) return 2;
  if (/\b(?:q8|int8|8\s*bit|8bit)\b/.test(source)) return 1;
  if (/\b(?:q6|6\s*bit|6bit)\b/.test(source)) return 0.75;
  if (/\b(?:q5|5\s*bit|5bit)\b/.test(source)) return 0.625;
  if (/\b(?:q4|4\s*bit|4bit)\b/.test(source)) return 0.5;
  return null;
}

function formatModelSize(row: LeaderboardRow) {
  const size = modelSizeGb(row);
  if (size === null) return "n/a";
  const prefix = fileSizeBytes(row) === null ? "~" : "";
  return `${prefix}${size.toFixed(size < 10 ? 1 : 0)} GB`;
}

function formatSamples(row: LeaderboardRow) {
  const value = numeric(row.benchmark_samples);
  if (value === null) return "samples n/a";
  return `${value.toLocaleString()} samples`;
}

function formatTruncation(row: LeaderboardRow) {
  const value = numeric(row.benchmark_truncated_rate);
  if (value === null) return "trunc n/a";
  const count = numeric(row.benchmark_truncated_count);
  return count === null ? `${(value * 100).toFixed(1)}% trunc` : `${(value * 100).toFixed(1)}% trunc · ${count}`;
}

function coverageLabel(row: LeaderboardRow) {
  const complete = CAPABILITIES.filter((capability) => capability.value(row) !== null).length;
  return `${complete}/${CAPABILITIES.length}`;
}

function modelSourceLink(row: LeaderboardRow) {
  if (isHostedOpenAIRow(row)) {
    return { label: "Hosted API", href: "", kind: "hosted" };
  }
  const candidates = [
    row.model_repo,
    variantConfigValue(row, "model_repo"),
    apiModel(row),
  ].filter((value): value is string => typeof value === "string" && value.trim().length > 0);
  const exactRepo = candidates.find((value) => /^[\w.-]+\/[\w.-]+$/.test(value) && !value.includes("@"));
  if (exactRepo) {
    return { label: "Hugging Face", href: `https://huggingface.co/${exactRepo}`, kind: "exact" };
  }
  const search = [apiModel(row), displayModelName(row), providerLabel(row)].filter(Boolean).join(" ");
  if (search.trim()) {
    return {
      label: "HF search",
      href: `https://huggingface.co/models?search=${encodeURIComponent(search)}`,
      kind: "search",
    };
  }
  return { label: "Source n/a", href: "", kind: "missing" };
}

function formatSeconds(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = Math.round(value % 60);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatCount(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : Math.round(value).toLocaleString();
}

function formatNumber(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : value.toFixed(value < 10 ? 2 : 1);
}

function formatUsd(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (value === 0) return "$0";
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}

function formatCell(value: unknown) {
  if (value === null || value === undefined || value === "") return "n/a";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "n/a";
    if (Math.abs(value) <= 1 && value !== 0) return value.toFixed(4);
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  return String(value);
}

function humanizeColumn(value: string) {
  return value.replace(/_/g, " ");
}

function formatPercent(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function formatPoints(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(1)} pts`;
}

function formatIndexNumber(value: number) {
  return String(Math.round(value * 100));
}

function formatBillionSize(value: number) {
  if (value < 1) return value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  return Number.isInteger(value) ? String(value) : value.toFixed(1).replace(/0$/, "");
}

function formatDate(value: string) {
  return new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function slugForAsset(value: string) {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
