import {
  ArrowUpRight,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  FileJson,
  Info,
  Lightbulb,
  LightbulbOff,
  ListFilter,
  RotateCcw,
  Scale,
  Search,
  Server,
  ShieldCheck,
  SlidersHorizontal,
  X,
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
  backend: string;
  hardware: string;
  minQuality: string;
  maxRuntimeMinutes: string;
};

type QuantizationOption = {
  key: string;
  label: string;
  row: LeaderboardRow;
  score: number | null;
  rank: number;
};

type QuantizationComparison = {
  groupKey: string;
  baseline: QuantizationOption;
  selected: QuantizationOption;
  options: QuantizationOption[];
};

type ReasoningOption = {
  key: string;
  label: string;
  row: LeaderboardRow;
  score: number | null;
  rank: number;
};

type ReasoningComparison = {
  groupKey: string;
  baseline: ReasoningOption;
  selected: ReasoningOption;
  options: ReasoningOption[];
};

type ScoreBaseline = {
  label: string;
  row: LeaderboardRow;
};

type ComparableRow = LeaderboardRow & {
  quantizationComparison?: QuantizationComparison;
  reasoningComparison?: ReasoningComparison;
  scoreBaseline?: ScoreBaseline;
};

type MetricColumn = {
  key: string;
  label: string;
  render: (row: ComparableRow) => ReactNode;
  primary?: boolean;
};

const emptyFilters: Filters = {
  query: "",
  family: "all",
  parameterSize: "all",
  backend: "all",
  hardware: "all",
  minQuality: "",
  maxRuntimeMinutes: "",
};

export function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(emptyFilters);
  const [selectedQuantizations, setSelectedQuantizations] = useState<Record<string, string>>({});
  const [selectedReasoning, setSelectedReasoning] = useState<Record<string, string>>({});
  const [activeVariantId, setActiveVariantId] = useState<string | null>(null);
  const [drawerRow, setDrawerRow] = useState<LeaderboardRow | null>(null);

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
  const comparableRows = useMemo(
    () => buildComparableRows(filteredRows, selectedQuantizations, selectedReasoning),
    [filteredRows, selectedQuantizations, selectedReasoning],
  );
  const rankedRows = useMemo(() => rankRows(comparableRows), [comparableRows]);
  const activeRow =
    rankedRows.find((row) => row.variant_id === activeVariantId) ?? rankedRows[0] ?? null;
  const options = useMemo(() => optionSets(publishableRows), [publishableRows]);
  const hiddenSyntheticCount = rawRows.length - realRows.length;
  const hiddenSmokeCount = realRows.length - publishableRows.length;
  const bestQuality = bestRow(publishableRows, qualityValue);
  const fastestRun = minRow(publishableRows, (row) => row.benchmark_runtime_seconds);
  const topBackend = mostCommon(publishableRows.map((row) => row.backend_name ?? "unknown"));
  const handleSelectQuantization = (
    groupKey: string,
    quantizationKey: string,
    variantId: string,
  ) => {
    setSelectedQuantizations((current) => ({ ...current, [groupKey]: quantizationKey }));
    setActiveVariantId(variantId);
  };
  const handleSelectReasoning = (
    groupKey: string,
    reasoningKey: string,
    variantId: string,
  ) => {
    setSelectedReasoning((current) => ({ ...current, [groupKey]: reasoningKey }));
    setActiveVariantId(variantId);
  };

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
      <SiteHeader generatedAt={payload.generated_at} />

      <section className="console-hero" id="overview">
        <div className="title-stack">
          <p className="eyebrow">Local benchmark console</p>
          <h1>Local AI Analysis</h1>
          <p>
            Reproducible benchmark runs for local models served through Ollama, LM Studio,
            and oMLX. Smoke checks stay out of the public ranking.
          </p>
        </div>

        <div className="run-summary" aria-label="Current best result">
          <div className="summary-topline">
            <span className="summary-bars">|| | || | |||</span>
            <span className="summary-label">Best row</span>
          </div>
          <strong>{bestQuality ? displayModelName(bestQuality) : "No publishable result"}</strong>
          <dl>
            <div>
              <dt>Score</dt>
              <dd>{formatQualityScore(bestQuality)}</dd>
            </div>
            <div>
              <dt>Runtime</dt>
              <dd>{formatDuration(bestQuality?.benchmark_runtime_seconds)}</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="metric-strip" aria-label="Benchmark status">
        <MetricCell
          label="Publishable Rows"
          value={String(publishableRows.length)}
          detail={`${hiddenSyntheticCount + hiddenSmokeCount} hidden`}
          icon={<Database size={18} />}
        />
        <MetricCell
          label="Top Score"
          value={formatQualityScore(bestQuality)}
          detail={bestQuality ? protocolLabel(bestQuality) : "No score"}
          icon={<Scale size={18} />}
        />
        <MetricCell
          label="Fastest Full Run"
          value={formatDuration(fastestRun?.benchmark_runtime_seconds)}
          detail={fastestRun ? displayModelName(fastestRun) : "n/a"}
          icon={<Clock3 size={18} />}
        />
        <MetricCell
          label="Primary Backend"
          value={topBackend ?? "n/a"}
          detail={`${countUnique(publishableRows, "backend_name")} backend(s)`}
          icon={<Server size={18} />}
        />
      </section>

      <section className="workspace-grid" id="leaderboard">
        <section className="leaderboard-panel" aria-labelledby="leaderboard-title">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Leaderboard</p>
              <h2 id="leaderboard-title">Real benchmark results</h2>
            </div>
            <span className="row-count">
              {rankedRows.length} of {publishableRows.length}
            </span>
          </div>

          <FilterPanel
            filters={filters}
            options={options}
            hiddenSyntheticCount={hiddenSyntheticCount}
            hiddenSmokeCount={hiddenSmokeCount}
            onChange={setFilters}
            onReset={() => setFilters(emptyFilters)}
          />

          <LeaderboardTable
            rows={rankedRows}
            activeVariantId={activeRow?.variant_id}
            onActivate={setActiveVariantId}
            onOpenRaw={setDrawerRow}
            onSelectQuantization={handleSelectQuantization}
            onSelectReasoning={handleSelectReasoning}
          />
        </section>
      </section>

      <section className="analysis-grid" id="analysis">
        <SelectedRunPanel row={activeRow} onOpenRaw={setDrawerRow} />
        <MetricCoveragePanel rows={publishableRows} />
        <DataStatusPanel
          generatedAt={payload.generated_at}
          hiddenSyntheticCount={hiddenSyntheticCount}
          hiddenSmokeCount={hiddenSmokeCount}
          totalRows={rawRows.length}
        />
      </section>

      <section className="methodology-band" id="methodology">
        <div>
          <p className="eyebrow">Methodology</p>
          <h2>Reproducibility first.</h2>
        </div>
        <div className="method-rows">
          <MethodRow
            icon={<CheckCircle2 size={18} />}
            title="Quality"
            text="Global MMLU Lite, IFBench, BFCL v4, OCRBench v2, MMMU, MBPP, and RGB are the publishable quality protocols for the current workflow."
          />
          <MethodRow
            icon={<Cpu size={18} />}
            title="Runtime"
            text="Runtime, backend, hardware, model IDs, and raw run metadata stay attached to each row."
          />
          <MethodRow
            icon={<ShieldCheck size={18} />}
            title="Publishing"
            text="The website only renders full, publishable rows in the public leaderboard."
          />
        </div>
      </section>

      <footer className="footer">
        <span>Local AI Analysis</span>
        <span>{payload.tagline}</span>
      </footer>

      {drawerRow && <MetadataDrawer row={drawerRow} onClose={() => setDrawerRow(null)} />}
    </main>
  );
}

function SiteHeader({ generatedAt }: { generatedAt: string }) {
  return (
    <header className="site-header">
      <a href="#overview" className="site-mark" aria-label="Local AI Analysis home">
        <span>NC</span>
        <strong>Local AI Analysis</strong>
      </a>
      <nav aria-label="Primary navigation">
        <a href="#leaderboard">Leaderboard</a>
        <a href="#analysis">Analysis</a>
        <a href="#methodology">Methodology</a>
      </nav>
      <time dateTime={generatedAt}>Updated {formatDate(generatedAt)}</time>
    </header>
  );
}

function MetricCell({
  label,
  value,
  detail,
  icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: ReactNode;
}) {
  return (
    <div className="metric-cell">
      <span className="metric-icon">{icon}</span>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

function FilterPanel({
  filters,
  options,
  hiddenSyntheticCount,
  hiddenSmokeCount,
  onChange,
  onReset,
}: {
  filters: Filters;
  options: ReturnType<typeof optionSets>;
  hiddenSyntheticCount: number;
  hiddenSmokeCount: number;
  onChange: (filters: Filters) => void;
  onReset: () => void;
}) {
  return (
    <div className="filter-panel" aria-label="Leaderboard controls">
      <div className="filter-heading">
        <div>
          <p className="eyebrow">Controls</p>
          <h2>Filter runs</h2>
        </div>
        <ListFilter size={18} aria-hidden="true" />
      </div>

      <label className="search-field">
        <span>Search</span>
        <div>
          <Search size={16} aria-hidden="true" />
          <input
            type="search"
            value={filters.query}
            placeholder="Model, family, backend"
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
          label="Parameters"
          value={filters.parameterSize}
          options={options.parameterSizes}
          onChange={(parameterSize) => onChange({ ...filters, parameterSize })}
        />
        <Select
          label="Backend"
          value={filters.backend}
          options={options.backends}
          onChange={(backend) => onChange({ ...filters, backend })}
        />
        <Select
          label="Hardware"
          value={filters.hardware}
          options={options.hardware}
          onChange={(hardware) => onChange({ ...filters, hardware })}
        />
        <NumberInput
          label="Min quality %"
          value={filters.minQuality}
          onChange={(minQuality) => onChange({ ...filters, minQuality })}
        />
        <NumberInput
          label="Max runtime min"
          value={filters.maxRuntimeMinutes}
          onChange={(maxRuntimeMinutes) => onChange({ ...filters, maxRuntimeMinutes })}
        />
      </div>

      <button className="reset-button" type="button" onClick={onReset}>
        <RotateCcw size={15} aria-hidden="true" />
        Reset filters
      </button>

      <div className="filter-note">
        <SlidersHorizontal size={16} aria-hidden="true" />
        <span>
          Hidden from public view: {hiddenSyntheticCount} non-publishable row(s),{" "}
          {hiddenSmokeCount} smoke run(s).
        </span>
      </div>
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

function LeaderboardTable({
  rows,
  activeVariantId,
  onActivate,
  onOpenRaw,
  onSelectQuantization,
  onSelectReasoning,
}: {
  rows: ComparableRow[];
  activeVariantId?: string;
  onActivate: (variantId: string) => void;
  onOpenRaw: (row: LeaderboardRow) => void;
  onSelectQuantization: (groupKey: string, quantizationKey: string, variantId: string) => void;
  onSelectReasoning: (groupKey: string, reasoningKey: string, variantId: string) => void;
}) {
  const metricColumns = metricColumnsFor(rows);
  const columnCount = 3 + metricColumns.length;

  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            <th className="rank-heading">#</th>
            <th>Model</th>
            {metricColumns.map((column) => (
              <th className="numeric-heading" key={column.key}>
                {column.label}
              </th>
            ))}
            <th className="action-heading">Raw</th>
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
              className={row.variant_id === activeVariantId ? "active-row" : undefined}
              key={row.variant_id}
              onClick={() => onActivate(row.variant_id)}
            >
              <td className="rank-cell">{String(index + 1).padStart(2, "0")}</td>
              <td className="model-cell">
                <div className="model-title-row">
                  <LabIcon row={row} />
                  <div className="model-title-stack">
                    <div className="model-title-line">
                      <strong title={row.variant_name}>{displayModelName(row)}</strong>
                    </div>
                    <div className="model-subline">
                      <span>
                        {row.family} · {formatParameter(row.parameter_size_b)}
                      </span>
                    </div>
                    <div className="model-option-stack">
                      <QuantizationSwitch row={row} onSelect={onSelectQuantization} />
                      <ReasoningSwitch row={row} onSelect={onSelectReasoning} />
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
              <td className="action-cell">
                <button
                  className="icon-button"
                  type="button"
                  aria-label={`Open raw metadata for ${row.variant_name}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onOpenRaw(row);
                  }}
                >
                  <FileJson size={16} aria-hidden="true" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QuantizationSwitch({
  row,
  onSelect,
}: {
  row: ComparableRow;
  onSelect: (groupKey: string, quantizationKey: string, variantId: string) => void;
}) {
  const comparison = row.quantizationComparison;
  const options =
    comparison?.options ?? [
      {
        key: quantizationKey(row),
        label: quantizationLabel(row),
        row,
        score: qualityValue(row),
        rank: quantizationRank(row),
      },
    ];
  const activeKey = comparison?.selected.key ?? quantizationKey(row);
  const baselineKey = comparison?.baseline.key ?? activeKey;

  return (
    <div className="quantization-switch" aria-label={`Quantizations for ${displayModelName(row)}`}>
      {options.map((option) => (
        <button
          className={[
            "quantization-chip",
            option.key === activeKey ? "active" : "",
            option.key === baselineKey ? "baseline" : "",
          ]
            .filter(Boolean)
            .join(" ")}
          key={option.key}
          type="button"
          title={`${option.label}: ${formatQualityScore(option.row)}`}
          onClick={(event) => {
            event.stopPropagation();
            if (comparison) {
              onSelect(comparison.groupKey, option.key, option.row.variant_id);
            }
          }}
        >
          <span>{option.label}</span>
          <small>{formatQualityScore(option.row)}</small>
        </button>
      ))}
    </div>
  );
}

function ReasoningSwitch({
  row,
  onSelect,
}: {
  row: ComparableRow;
  onSelect: (groupKey: string, reasoningKey: string, variantId: string) => void;
}) {
  const comparison = row.reasoningComparison;
  const options =
    comparison?.options ?? [
      {
        key: reasoningKey(row),
        label: reasoningLabel(row),
        row,
        score: qualityValue(row),
        rank: reasoningRank(row),
      },
    ];
  const activeKey = comparison?.selected.key ?? reasoningKey(row);
  const baselineKey = comparison?.baseline.key ?? activeKey;

  return (
    <div className="reasoning-switch" aria-label={`Reasoning modes for ${displayModelName(row)}`}>
      {options.map((option) => {
        const enabled = reasoningKey(option.row) !== "off";
        const Icon = enabled ? Lightbulb : LightbulbOff;
        return (
          <button
            className={[
              "reasoning-chip",
              option.key === activeKey ? "active" : "",
              option.key === baselineKey ? "baseline" : "",
              enabled ? "on" : "off",
            ]
              .filter(Boolean)
              .join(" ")}
            key={option.key}
            type="button"
            title={`Reasoning ${option.label}: ${formatQualityScore(option.row)}`}
            onClick={(event) => {
              event.stopPropagation();
              if (comparison) {
                onSelect(comparison.groupKey, option.key, option.row.variant_id);
              }
            }}
          >
            <Icon size={13} aria-hidden="true" />
            <span>{option.label}</span>
            <small>{formatQualityScore(option.row)}</small>
          </button>
        );
      })}
    </div>
  );
}

function PointsWithDelta({ row }: { row: ComparableRow }) {
  const value = numeric(row.model_intelligence_score);
  const baseline = row.scoreBaseline;
  const baselineValue = numeric(baseline?.row.model_intelligence_score);
  const showDelta = baseline && !sameRun(baseline.row, row) && value !== null && baselineValue !== null;

  if (!showDelta) {
    return <span className="metric-stack">{formatPoints(value)}</span>;
  }

  const delta = value - baselineValue;
  return (
    <span className="metric-stack">
      <span>{formatPoints(value)}</span>
      <small>
        <span>{baseline.label} {formatPoints(baselineValue)}</span>
        <strong className={deltaClass(delta)}>{formatSignedPoints(delta)}</strong>
      </small>
    </span>
  );
}

function LabIcon({ row }: { row: LeaderboardRow }) {
  const [candidateIndex, setCandidateIndex] = useState(0);
  const key = labKey(row);
  const candidates = [`/labs/${key}.svg`, `/labs/${key}.png`, `/labs/${key}.jpg`, `/labs/${key}.webp`];
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

function SelectedRunPanel({
  row,
  onOpenRaw,
}: {
  row: ComparableRow | null;
  onOpenRaw: (row: LeaderboardRow) => void;
}) {
  if (!row) {
    return (
      <section className="analysis-panel selected-run" aria-labelledby="selected-run-title">
        <div className="analysis-heading">
          <p className="eyebrow">Selected Run</p>
          <h2 id="selected-run-title">No result selected</h2>
        </div>
        <p className="muted">Run a full benchmark to populate the analysis panel.</p>
      </section>
    );
  }

  return (
    <section className="analysis-panel selected-run" aria-labelledby="selected-run-title">
      <div className="analysis-heading">
        <p className="eyebrow">Selected Run</p>
        <h2 id="selected-run-title">{displayModelName(row)}</h2>
      </div>

      <div className="score-display">
        <span>
          {protocolLabel(row)} · {quantizationLabel(row)} · Reasoning {reasoningLabel(row)}
        </span>
        <strong>{formatQualityScore(row)}</strong>
        <ScoreComparison row={row} />
      </div>

      <dl className="detail-list">
        <DetailItem label="Runtime" value={formatDuration(row.benchmark_runtime_seconds)} />
        <DetailItem label="Quantization" value={quantizationLabel(row)} />
        <DetailItem label="Reasoning" value={reasoningLabel(row)} />
        <DetailItem label="Modality" value={modalityLabel(row)} />
        <DetailItem label="Suite coverage" value={formatPercent(row.model_intelligence_coverage)} />
        <DetailItem label="Secondary score" value={formatSecondaryScore(row)} />
        <DetailItem label="Backend" value={row.backend_name ?? "unknown"} />
        <DetailItem label="API model" value={apiModel(row) ?? row.base_model_name} />
        <DetailItem label="Run label" value={row.variant_name} />
        <DetailItem label="Run UUID" value={shortId(row.run_uuid)} />
      </dl>

      <button className="text-button" type="button" onClick={() => onOpenRaw(row)}>
        Open metadata
        <ArrowUpRight size={15} aria-hidden="true" />
      </button>
    </section>
  );
}

function ScoreComparison({ row }: { row: ComparableRow }) {
  const baseline = row.scoreBaseline;
  if (!baseline || sameRun(baseline.row, row)) return null;
  const selectedScore = numeric(row.model_intelligence_score);
  const baselineScore = numeric(baseline.row.model_intelligence_score);
  if (selectedScore === null || baselineScore === null) return null;
  const delta = selectedScore - baselineScore;

  return (
    <small className="score-compare">
      <span>
        {baseline.label} baseline {formatPoints(baselineScore)}
      </span>
      <strong className={deltaClass(delta)}>{formatSignedPoints(delta)}</strong>
    </small>
  );
}

function MetricCoveragePanel({ rows }: { rows: LeaderboardRow[] }) {
  const metrics = [
    ["GMMLU Lite", hasMetric(rows, "global_mmlu_lite_pass_at_1")],
    ["IFBench", hasMetric(rows, "ifbench_prompt_level_loose")],
    ["BFCL v4", hasMetric(rows, "bfcl_v4_selected_accuracy")],
    ["OCRBench v2", hasMetric(rows, "ocrbench_v2_score")],
    ["MMMU", hasMetric(rows, "mmmu_accuracy")],
    ["MBPP", hasMetric(rows, "mbpp_pass_at_1")],
    ["RGB", hasMetric(rows, "rgb_all_rate")],
    ["SimpleQA", hasMetric(rows, "simpleqa_f1")],
    ["HarmBench", hasMetric(rows, "harmbench_refusal_rate")],
    ["Intelligence score", hasMetric(rows, "model_intelligence_score")],
    ["Micro average", hasMetric(rows, "global_mmlu_lite_micro_pass_at_1")],
    ["Invalid rate", hasMetric(rows, "global_mmlu_lite_invalid_rate")],
    ["Runtime", hasMetric(rows, "benchmark_runtime_seconds")],
  ] as const;

  return (
    <section className="analysis-panel" aria-labelledby="coverage-title">
      <div className="analysis-heading">
        <p className="eyebrow">Metrics</p>
        <h2 id="coverage-title">Available fields</h2>
      </div>
      <div className="coverage-grid">
        {metrics.map(([label, available]) => (
          <span className={available ? "coverage-pill available" : "coverage-pill"} key={label}>
            {available ? <CheckCircle2 size={14} /> : <Info size={14} />}
            {label}
          </span>
        ))}
      </div>
      <p className="muted">
        The website only surfaces metrics produced by the current API benchmark path.
      </p>
    </section>
  );
}

function DataStatusPanel({
  generatedAt,
  hiddenSyntheticCount,
  hiddenSmokeCount,
  totalRows,
}: {
  generatedAt: string;
  hiddenSyntheticCount: number;
  hiddenSmokeCount: number;
  totalRows: number;
}) {
  return (
    <section className="analysis-panel" aria-labelledby="data-status-title">
      <div className="analysis-heading">
        <p className="eyebrow">Data Status</p>
        <h2 id="data-status-title">Website export</h2>
      </div>
      <dl className="detail-list">
        <DetailItem label="Exported" value={formatDateTime(generatedAt)} />
        <DetailItem label="Total rows" value={String(totalRows)} />
        <DetailItem label="Non-publishable rows hidden" value={String(hiddenSyntheticCount)} />
        <DetailItem label="Smoke runs hidden" value={String(hiddenSmokeCount)} />
      </dl>
    </section>
  );
}

function DetailItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function MethodRow({
  icon,
  title,
  text,
}: {
  icon: ReactNode;
  title: string;
  text: string;
}) {
  return (
    <div className="method-row">
      <span className="metric-icon">{icon}</span>
      <strong>{title}</strong>
      <p>{text}</p>
    </div>
  );
}

function MetadataDrawer({ row, onClose }: { row: LeaderboardRow; onClose: () => void }) {
  const [rawPayload, setRawPayload] = useState<unknown | null>(null);
  const [rawError, setRawError] = useState<string | null>(null);

  useEffect(() => {
    const apiUrl = import.meta.env.VITE_API_URL;
    if (!apiUrl) return;
    fetch(`${apiUrl.replace(/\/$/, "")}/api/variants/${row.variant_id}/raw`)
      .then((response) => {
        if (!response.ok) throw new Error(`Raw endpoint returned ${response.status}`);
        return response.json();
      })
      .then(setRawPayload)
      .catch((err) => setRawError(err instanceof Error ? err.message : String(err)));
  }, [row.variant_id]);

  const fallbackPayload = {
    variant: row.variant_name,
    base_model: row.base_model_name,
    model_repo: row.model_repo,
    file_name: row.file_name,
    checksum_sha256: row.checksum_sha256,
    quantization: row.quantization,
    backend: {
      name: row.backend_name,
      version: row.backend_version,
      commit: row.backend_commit,
    },
    hardware: {
      accelerator: row.hardware_accelerator,
      cpu_model: row.cpu_model,
      gpu_name: row.gpu_name,
    },
    run_uuid: row.run_uuid,
    metrics: {
      global_mmlu_lite_pass_at_1: row.global_mmlu_lite_pass_at_1,
      global_mmlu_lite_micro_pass_at_1: row.global_mmlu_lite_micro_pass_at_1,
      global_mmlu_lite_invalid_rate: row.global_mmlu_lite_invalid_rate,
      ifbench_prompt_level_loose: row.ifbench_prompt_level_loose,
      ifbench_instruction_level_loose: row.ifbench_instruction_level_loose,
      ifbench_prompt_level_strict: row.ifbench_prompt_level_strict,
      ifbench_instruction_level_strict: row.ifbench_instruction_level_strict,
      bfcl_v4_selected_accuracy: row.bfcl_v4_selected_accuracy,
      bfcl_v4_invalid_rate: row.bfcl_v4_invalid_rate,
      bfcl_v4_non_live_accuracy: row.bfcl_v4_non_live_accuracy,
      bfcl_v4_live_accuracy: row.bfcl_v4_live_accuracy,
      bfcl_v4_multi_turn_accuracy: row.bfcl_v4_multi_turn_accuracy,
      bfcl_v4_agentic_accuracy: row.bfcl_v4_agentic_accuracy,
      ocrbench_v2_score: row.ocrbench_v2_score,
      ocrbench_v2_micro_score: row.ocrbench_v2_micro_score,
      ocrbench_v2_en_score: row.ocrbench_v2_en_score,
      ocrbench_v2_cn_score: row.ocrbench_v2_cn_score,
      mmmu_accuracy: row.mmmu_accuracy,
      mmmu_invalid_rate: row.mmmu_invalid_rate,
      mmmu_multiple_choice_accuracy: row.mmmu_multiple_choice_accuracy,
      mmmu_open_accuracy: row.mmmu_open_accuracy,
      mbpp_pass_at_1: row.mbpp_pass_at_1,
      mbpp_invalid_rate: row.mbpp_invalid_rate,
      mbpp_compile_rate: row.mbpp_compile_rate,
      mbpp_runtime_error_rate: row.mbpp_runtime_error_rate,
      rgb_all_rate: row.rgb_all_rate,
      rgb_rejection_rate: row.rgb_rejection_rate,
      rgb_fact_check_rate: row.rgb_fact_check_rate,
      rgb_error_correction_rate: row.rgb_error_correction_rate,
      simpleqa_f1: row.simpleqa_f1,
      simpleqa_correct_rate: row.simpleqa_correct_rate,
      simpleqa_incorrect_rate: row.simpleqa_incorrect_rate,
      simpleqa_hallucination_rate: row.simpleqa_hallucination_rate,
      simpleqa_not_attempted_rate: row.simpleqa_not_attempted_rate,
      simpleqa_accuracy_given_attempted: row.simpleqa_accuracy_given_attempted,
      harmbench_attack_success_rate: row.harmbench_attack_success_rate,
      harmbench_refusal_rate: row.harmbench_refusal_rate,
      model_intelligence_score: row.model_intelligence_score,
      model_intelligence_coverage: row.model_intelligence_coverage,
      model_intelligence_available_score: row.model_intelligence_available_score,
      benchmark_runtime_seconds: row.benchmark_runtime_seconds,
    },
    metadata: safeMetadata(row),
  };

  return (
    <aside className="drawer" aria-label="Raw metadata">
      <div className="drawer-header">
        <div>
          <p className="eyebrow">Raw Metadata</p>
          <h2>{row.variant_name}</h2>
        </div>
        <button className="icon-button close-button" type="button" onClick={onClose}>
          <X size={17} aria-hidden="true" />
        </button>
      </div>
      {rawError && <p className="error">Raw API unavailable: {rawError}</p>}
      <pre>{JSON.stringify(rawPayload ?? fallbackPayload, null, 2)}</pre>
    </aside>
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
  const minQuality = parseNumber(filters.minQuality);
  const maxRuntimeSeconds = parseNumber(filters.maxRuntimeMinutes);

  return rows.filter((row) => {
    const searchable = [
      row.variant_name,
      row.base_model_name,
      row.family,
      row.quantization,
      quantizationLabel(row),
      row.backend_name,
      row.hardware_accelerator,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

    return (
      (!query || searchable.includes(query)) &&
      (filters.family === "all" || row.family === filters.family) &&
      (filters.parameterSize === "all" || String(row.parameter_size_b) === filters.parameterSize) &&
      (filters.backend === "all" || (row.backend_name ?? "unknown") === filters.backend) &&
      (filters.hardware === "all" ||
        (row.hardware_accelerator ?? "unknown") === filters.hardware) &&
      (minQuality === null || meetsMinimum(qualityValue(row), minQuality / 100)) &&
      (maxRuntimeSeconds === null ||
        meetsMaximum(row.benchmark_runtime_seconds, maxRuntimeSeconds * 60))
    );
  });
}

function optionSets(rows: LeaderboardRow[]) {
  return {
    families: unique(rows.map((row) => row.family)),
    parameterSizes: unique(rows.map((row) => String(row.parameter_size_b))),
    backends: unique(rows.map((row) => row.backend_name ?? "unknown")),
    hardware: unique(rows.map((row) => row.hardware_accelerator ?? "unknown")),
  };
}

function buildComparableRows(
  rows: LeaderboardRow[],
  selectedQuantizations: Record<string, string>,
  selectedReasoning: Record<string, string>,
): ComparableRow[] {
  const groups = new Map<string, LeaderboardRow[]>();
  for (const row of rows) {
    const key = quantizationGroupKey(row);
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }

  return Array.from(groups.entries()).map(([groupKey, groupRows]) => {
    const allQuantizationOptions = quantizationOptionsFor(groupRows);
    const baselineQuantization = baselineQuantizationOption(allQuantizationOptions);
    const selectedQuantizationKey = selectedQuantizations[groupKey] ?? baselineQuantization.key;
    const currentQuantizationRows = rowsForQuantization(groupRows, selectedQuantizationKey);
    const fallbackQuantizationRows = rowsForQuantization(groupRows, baselineQuantization.key);
    const candidateRows = currentQuantizationRows.length
      ? currentQuantizationRows
      : fallbackQuantizationRows;
    const reasoningOptions = reasoningOptionsFor(candidateRows.length ? candidateRows : groupRows);
    const baselineReasoning = baselineReasoningOption(reasoningOptions);
    const selectedReasoningOption =
      reasoningOptions.find((option) => option.key === selectedReasoning[groupKey]) ??
      baselineReasoning;
    const quantizationOptions = quantizationOptionsFor(groupRows, selectedReasoningOption.key);
    const selectedQuantization =
      quantizationOptions.find((option) => option.key === selectedQuantizationKey) ??
      baselineQuantization;
    const selectedRows = rowsForQuantization(groupRows, selectedQuantization.key);
    const selectedReasoningRows = selectedRows.filter(
      (row) => reasoningKey(row) === selectedReasoningOption.key,
    );
    const selectedRow = bestComparableRun(
      selectedReasoningRows.length ? selectedReasoningRows : selectedRows,
    );
    const scoreBaseline = scoreBaselineFor(groupRows, baselineQuantization.key);

    return {
      ...selectedRow,
      quantizationComparison: {
        groupKey,
        baseline:
          quantizationOptions.find((option) => option.key === baselineQuantization.key) ??
          baselineQuantization,
        selected: selectedQuantization,
        options: quantizationOptions,
      },
      reasoningComparison: {
        groupKey,
        baseline: baselineReasoning,
        selected:
          reasoningOptions.find((option) => option.key === reasoningKey(selectedRow)) ??
          selectedReasoningOption,
        options: reasoningOptions,
      },
      scoreBaseline,
    };
  });
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

function compareQuantizationOptions(a: QuantizationOption, b: QuantizationOption) {
  const scoreDelta = (a.score ?? Number.NEGATIVE_INFINITY) - (b.score ?? Number.NEGATIVE_INFINITY);
  if (scoreDelta !== 0) return scoreDelta;
  const coverageDelta =
    (numeric(a.row.model_intelligence_coverage) ?? Number.NEGATIVE_INFINITY) -
    (numeric(b.row.model_intelligence_coverage) ?? Number.NEGATIVE_INFINITY);
  if (coverageDelta !== 0) return coverageDelta;
  return (
    (numeric(b.row.benchmark_runtime_seconds) ?? Number.POSITIVE_INFINITY) -
    (numeric(a.row.benchmark_runtime_seconds) ?? Number.POSITIVE_INFINITY)
  );
}

function baselineQuantizationOption(options: QuantizationOption[]) {
  const preferred = options.find((option) => option.rank === 160);
  return preferred ?? options[0]!;
}

function reasoningOptionsFor(rows: LeaderboardRow[]) {
  const byReasoning = new Map<string, ReasoningOption>();
  for (const row of rows) {
    const key = reasoningKey(row);
    const option: ReasoningOption = {
      key,
      label: reasoningLabel(row),
      row,
      score: qualityValue(row),
      rank: reasoningRank(row),
    };
    const current = byReasoning.get(key);
    if (!current || compareReasoningOptions(option, current) > 0) {
      byReasoning.set(key, option);
    }
  }

  return Array.from(byReasoning.values()).sort((a, b) => {
    if (a.rank !== b.rank) return a.rank - b.rank;
    return (b.score ?? Number.NEGATIVE_INFINITY) - (a.score ?? Number.NEGATIVE_INFINITY);
  });
}

function compareReasoningOptions(a: ReasoningOption, b: ReasoningOption) {
  const scoreDelta = (a.score ?? Number.NEGATIVE_INFINITY) - (b.score ?? Number.NEGATIVE_INFINITY);
  if (scoreDelta !== 0) return scoreDelta;
  return (
    (numeric(b.row.benchmark_runtime_seconds) ?? Number.POSITIVE_INFINITY) -
    (numeric(a.row.benchmark_runtime_seconds) ?? Number.POSITIVE_INFINITY)
  );
}

function baselineReasoningOption(options: ReasoningOption[]) {
  const preferred = options.find((option) => option.key === "off");
  return preferred ?? options[0]!;
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

function scoreBaselineFor(rows: LeaderboardRow[], baselineQuantizationKey: string): ScoreBaseline {
  const baselineRows = rowsForQuantization(rows, baselineQuantizationKey);
  const reasoningOptions = reasoningOptionsFor(baselineRows.length ? baselineRows : rows);
  const baselineReasoning = baselineReasoningOption(reasoningOptions);
  return {
    label: `${quantizationLabel(baselineReasoning.row)} ${baselineReasoning.label}`,
    row: baselineReasoning.row,
  };
}

function metricColumnsFor(rows: LeaderboardRow[]): MetricColumn[] {
  const columns: MetricColumn[] = [];

  if (hasMetric(rows, "model_intelligence_score")) {
    columns.push({
      key: "model_intelligence_score",
      label: "Intel. pts",
      render: (row) => <PointsWithDelta row={row} />,
      primary: true,
    });
  }
  if (hasMetric(rows, "model_intelligence_coverage")) {
    columns.push({
      key: "model_intelligence_coverage",
      label: "Coverage",
      render: (row) => formatPercent(row.model_intelligence_coverage),
    });
  }
  if (hasMetric(rows, "global_mmlu_lite_pass_at_1")) {
    columns.push({
      key: "global_mmlu_lite_pass_at_1",
      label: "GMMLU Lite",
      render: (row) => formatPercent(row.global_mmlu_lite_pass_at_1),
      primary: !hasMetric(rows, "model_intelligence_score"),
    });
  }
  if (hasMetric(rows, "ifbench_prompt_level_loose")) {
    columns.push({
      key: "ifbench_prompt_level_loose",
      label: "IFBench",
      render: (row) => formatPercent(row.ifbench_prompt_level_loose),
      primary: !hasMetric(rows, "global_mmlu_lite_pass_at_1"),
    });
  }
  if (hasMetric(rows, "ifbench_instruction_level_loose")) {
    columns.push({
      key: "ifbench_instruction_level_loose",
      label: "IF Inst.",
      render: (row) => formatPercent(row.ifbench_instruction_level_loose),
    });
  }
  if (hasMetric(rows, "bfcl_v4_selected_accuracy")) {
    columns.push({
      key: "bfcl_v4_selected_accuracy",
      label: "BFCL v4",
      render: (row) => formatPercent(row.bfcl_v4_selected_accuracy),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose"),
    });
  }
  if (hasMetric(rows, "bfcl_v4_invalid_rate")) {
    columns.push({
      key: "bfcl_v4_invalid_rate",
      label: "BFCL Invalid",
      render: (row) => formatPercent(row.bfcl_v4_invalid_rate),
    });
  }
  if (hasMetric(rows, "ocrbench_v2_score")) {
    columns.push({
      key: "ocrbench_v2_score",
      label: "OCRBench v2",
      render: (row) => formatPercent(row.ocrbench_v2_score),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy"),
    });
  }
  if (hasMetric(rows, "ocrbench_v2_micro_score")) {
    columns.push({
      key: "ocrbench_v2_micro_score",
      label: "OCR Micro",
      render: (row) => formatPercent(row.ocrbench_v2_micro_score),
    });
  }
  if (hasMetric(rows, "mmmu_accuracy")) {
    columns.push({
      key: "mmmu_accuracy",
      label: "MMMU",
      render: (row) => formatPercent(row.mmmu_accuracy),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasMetric(rows, "ocrbench_v2_score"),
    });
  }
  if (hasMetric(rows, "mmmu_invalid_rate")) {
    columns.push({
      key: "mmmu_invalid_rate",
      label: "MMMU Invalid",
      render: (row) => formatPercent(row.mmmu_invalid_rate),
    });
  }
  if (hasMetric(rows, "mbpp_pass_at_1")) {
    columns.push({
      key: "mbpp_pass_at_1",
      label: "MBPP",
      render: (row) => formatPercent(row.mbpp_pass_at_1),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasMetric(rows, "ocrbench_v2_score") &&
        !hasMetric(rows, "mmmu_accuracy"),
    });
  }
  if (hasMetric(rows, "mbpp_invalid_rate")) {
    columns.push({
      key: "mbpp_invalid_rate",
      label: "MBPP Invalid",
      render: (row) => formatPercent(row.mbpp_invalid_rate),
    });
  }
  if (hasMetric(rows, "rgb_all_rate")) {
    columns.push({
      key: "rgb_all_rate",
      label: "RGB",
      render: (row) => formatPercent(row.rgb_all_rate),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasMetric(rows, "ocrbench_v2_score") &&
        !hasMetric(rows, "mmmu_accuracy") &&
        !hasMetric(rows, "mbpp_pass_at_1"),
    });
  }
  if (hasMetric(rows, "rgb_rejection_rate")) {
    columns.push({
      key: "rgb_rejection_rate",
      label: "RGB Rej.",
      render: (row) => formatPercent(row.rgb_rejection_rate),
    });
  }
  if (hasMetric(rows, "simpleqa_f1")) {
    columns.push({
      key: "simpleqa_f1",
      label: "SimpleQA",
      render: (row) => formatPercent(row.simpleqa_f1),
      primary:
        !hasMetric(rows, "global_mmlu_lite_pass_at_1") &&
        !hasMetric(rows, "ifbench_prompt_level_loose") &&
        !hasMetric(rows, "bfcl_v4_selected_accuracy") &&
        !hasMetric(rows, "ocrbench_v2_score") &&
        !hasMetric(rows, "mmmu_accuracy") &&
        !hasMetric(rows, "mbpp_pass_at_1") &&
        !hasMetric(rows, "rgb_all_rate"),
    });
  }
  if (hasMetric(rows, "simpleqa_hallucination_rate")) {
    columns.push({
      key: "simpleqa_hallucination_rate",
      label: "SQA Wrong",
      render: (row) => formatPercent(row.simpleqa_hallucination_rate),
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
        !hasMetric(rows, "ocrbench_v2_score") &&
        !hasMetric(rows, "mmmu_accuracy") &&
        !hasMetric(rows, "mbpp_pass_at_1") &&
        !hasMetric(rows, "rgb_all_rate") &&
        !hasMetric(rows, "simpleqa_f1"),
    });
  }
  if (hasMetric(rows, "harmbench_attack_success_rate")) {
    columns.push({
      key: "harmbench_attack_success_rate",
      label: "Harm ASR",
      render: (row) => formatPercent(row.harmbench_attack_success_rate),
    });
  }
  if (hasMetric(rows, "global_mmlu_lite_micro_pass_at_1")) {
    columns.push({
      key: "global_mmlu_lite_micro_pass_at_1",
      label: "Micro",
      render: (row) => formatPercent(row.global_mmlu_lite_micro_pass_at_1),
    });
  }
  if (hasMetric(rows, "global_mmlu_lite_invalid_rate")) {
    columns.push({
      key: "global_mmlu_lite_invalid_rate",
      label: "Invalid",
      render: (row) => formatPercent(row.global_mmlu_lite_invalid_rate),
    });
  }
  if (hasMetric(rows, "benchmark_runtime_seconds")) {
    columns.push({
      key: "benchmark_runtime_seconds",
      label: "Runtime",
      render: (row) => formatDuration(row.benchmark_runtime_seconds),
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

function scoreForRank(row: LeaderboardRow) {
  return qualityValue(row) ?? 0;
}

function bestRow(rows: LeaderboardRow[], selector: (row: LeaderboardRow) => number | null) {
  return rows.reduce<LeaderboardRow | null>((best, row) => {
    const value = selector(row);
    if (value === null) return best;
    if (!best || value > (selector(best) ?? Number.NEGATIVE_INFINITY)) return row;
    return best;
  }, null);
}

function minRow(rows: LeaderboardRow[], selector: (row: LeaderboardRow) => number | null | undefined) {
  return rows.reduce<LeaderboardRow | null>((best, row) => {
    const value = numeric(selector(row));
    if (value === null) return best;
    if (!best || value < (numeric(selector(best)) ?? Number.POSITIVE_INFINITY)) return row;
    return best;
  }, null);
}

function numeric(value: unknown) {
  if (value === null || value === undefined || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function sameRun(a: LeaderboardRow, b: LeaderboardRow) {
  return rowIdentity(a) === rowIdentity(b);
}

function rowIdentity(row: LeaderboardRow) {
  return (
    row.normalized_result_id ??
    [
      row.variant_id,
      row.run_uuid ?? "",
      row.variant_name,
      quantizationKey(row),
      reasoningKey(row),
    ].join("|")
  );
}

function hasMetric(rows: LeaderboardRow[], key: keyof LeaderboardRow) {
  return rows.some((row) => numeric(row[key]) !== null);
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
    return "Model Intelligence";
  }
  if (row.global_mmlu_lite_pass_at_1 !== null && row.global_mmlu_lite_pass_at_1 !== undefined) {
    return "Global MMLU Lite";
  }
  if (row.ifbench_prompt_level_loose !== null && row.ifbench_prompt_level_loose !== undefined) {
    return "IFBench";
  }
  if (row.bfcl_v4_selected_accuracy !== null && row.bfcl_v4_selected_accuracy !== undefined) {
    return "BFCL v4";
  }
  if (row.ocrbench_v2_score !== null && row.ocrbench_v2_score !== undefined) {
    return "OCRBench v2";
  }
  if (row.mmmu_accuracy !== null && row.mmmu_accuracy !== undefined) {
    return "MMMU";
  }
  if (row.mbpp_pass_at_1 !== null && row.mbpp_pass_at_1 !== undefined) {
    return "MBPP";
  }
  if (row.rgb_all_rate !== null && row.rgb_all_rate !== undefined) {
    return "RGB";
  }
  if (row.simpleqa_f1 !== null && row.simpleqa_f1 !== undefined) {
    return "SimpleQA";
  }
  if (row.harmbench_refusal_rate !== null && row.harmbench_refusal_rate !== undefined) {
    return "HarmBench Safety";
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
  if (apiName && shouldPreferApiModelName(row.base_model_name, apiName)) {
    return formatModelName(apiName);
  }
  return formatModelName(row.base_model_name || apiName || row.variant_name);
}

function shouldPreferApiModelName(baseName: string, apiName: string) {
  if (!baseName) return true;
  const normalizedBase = baseName.toLowerCase();
  const normalizedApi = apiName.toLowerCase();
  if (normalizedApi.includes("gemma-4-e2b") && !normalizedBase.includes("gemma-4")) {
    return true;
  }
  return false;
}

function labKey(row: LeaderboardRow) {
  const source = `${row.family} ${row.base_model_name} ${apiModel(row) ?? ""} ${row.variant_name}`.toLowerCase();
  if (source.includes("qwen") || source.includes("alibaba")) return "qwen";
  if (source.includes("gemma") || source.includes("google")) return "google";
  if (source.includes("llama") || source.includes("meta")) return "meta";
  if (source.includes("mistral") || source.includes("mixtral")) return "mistral";
  if (source.includes("deepseek")) return "deepseek";
  if (source.includes("phi") || source.includes("microsoft")) return "microsoft";
  if (source.includes("granite") || source.includes("ibm")) return "ibm";
  if (source.includes("openai") || source.includes("gpt-oss")) return "openai";
  if (source.includes("cohere") || source.includes("aya")) return "cohere";
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
    formatModelName(row.base_model_name || apiModel(row) || row.variant_name).toLowerCase(),
    row.family.toLowerCase(),
    String(row.parameter_size_b),
  ].join("|");
}

function quantizationKey(row: LeaderboardRow) {
  return slugForAsset(quantizationLabel(row) || row.quantization || "default");
}

function quantizationLabel(row: LeaderboardRow) {
  const source = `${row.quantization} ${row.variant_name} ${apiModel(row) ?? ""} ${row.file_name ?? ""}`.toLowerCase();
  if (/\b(?:fp32|f32|32\s*bit|32b)\b/.test(source)) return "32 bit";
  if (/\b(?:bf16|fp16|f16|16\s*bit|16b)\b/.test(source)) return "16 bit";
  if (/\b(?:q8|8\s*bit|8b|8bit)\b/.test(source)) return "8 bit";
  if (/\b(?:q6|6\s*bit|6b|6bit)\b/.test(source)) return "6 bit";
  if (/\b(?:q5|5\s*bit|5b|5bit)\b/.test(source)) return "5 bit";
  if (/\b(?:q4|4\s*bit|4b|4bit)\b/.test(source)) return "4 bit";
  if (row.quantization && row.quantization.toUpperCase() !== "SERVER") {
    return row.quantization.toUpperCase();
  }
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

function formatModelName(value: string) {
  const lastSegment = value.split("/").pop() ?? value;
  const withoutRunSuffix = lastSegment
    .replace(/\b(?:ollama|lm studio|omlx|all languages|smoke|mbpp|full|test)\b/gi, " ")
    .replace(/\breasoning\s+(?:none|off|on|low|medium|high|auto|unset)\b/gi, " ")
    .replace(/\b(?:mlx|bf16|fp16|fp32|q\d+|gguf|4bit|8bit|it|instruct|chat)\b/gi, " ")
    .replace(/[:_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const qwen = withoutRunSuffix.match(/\bqwen\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)\s*b\b/i);
  if (qwen) return `Qwen ${qwen[1]} ${qwen[2]}B`;

  const gemmaE2b = withoutRunSuffix.match(/\bgemma\s*(\d+(?:\.\d+)?)\s*e\s*2\s*b\b/i);
  if (gemmaE2b) return `Gemma ${gemmaE2b[1]} E2B`;

  const gemma = withoutRunSuffix.match(/\bgemma\s*(\d+(?:\.\d+)?)?.*?(\d+(?:\.\d+)?)\s*b\b/i);
  if (gemma) {
    const version = gemma[1] ? ` ${gemma[1]}` : "";
    return `Gemma${version} ${gemma[2]}B`;
  }

  return titleCaseModelName(withoutRunSuffix || value);
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

function reasoningRank(row: LeaderboardRow) {
  const key = reasoningKey(row);
  if (key === "off") return 0;
  if (key === "low") return 1;
  if (key === "medium") return 2;
  if (key === "high") return 3;
  if (key === "on" || key === "auto") return 4;
  return 5;
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

function countUnique(rows: LeaderboardRow[], key: keyof LeaderboardRow) {
  return new Set(rows.map((row) => row[key]).filter(Boolean)).size;
}

function mostCommon(values: string[]) {
  const counts = new Map<string, number>();
  for (const value of values.filter(Boolean)) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
}

function meetsMinimum(value: number | null | undefined, minimum: number) {
  return value !== null && value !== undefined && Number.isFinite(value) && value >= minimum;
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

function formatDateTime(value: string) {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatParameter(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1)}B`;
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

function formatSignedPoints(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(1)} pts`;
}

function deltaClass(value: number) {
  if (value > 0.00001) return "positive";
  if (value < -0.00001) return "negative";
  return "neutral";
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
