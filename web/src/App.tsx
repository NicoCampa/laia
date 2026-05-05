import {
  ArrowUpRight,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  FileJson,
  Info,
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
  quantization: string;
  backend: string;
  hardware: string;
  minQuality: string;
  maxRuntimeMinutes: string;
};

type MetricColumn = {
  key: string;
  label: string;
  render: (row: LeaderboardRow) => string;
  primary?: boolean;
};

const emptyFilters: Filters = {
  query: "",
  family: "all",
  parameterSize: "all",
  quantization: "all",
  backend: "all",
  hardware: "all",
  minQuality: "",
  maxRuntimeMinutes: "",
};

export function App() {
  const [payload, setPayload] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(emptyFilters);
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
  const rankedRows = useMemo(() => rankRows(filteredRows), [filteredRows]);
  const activeRow =
    rankedRows.find((row) => row.variant_id === activeVariantId) ?? rankedRows[0] ?? null;
  const options = useMemo(() => optionSets(publishableRows), [publishableRows]);
  const hiddenSyntheticCount = rawRows.length - realRows.length;
  const hiddenSmokeCount = realRows.length - publishableRows.length;
  const bestQuality = bestRow(publishableRows, qualityValue);
  const fastestRun = minRow(publishableRows, (row) => row.benchmark_runtime_seconds);
  const topBackend = mostCommon(publishableRows.map((row) => row.backend_name ?? "unknown"));

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
          <p className="eyebrow">Benchmark Console</p>
          <h1>Local model leaderboard</h1>
          <p>
            Publishable runs from your local benchmark pipeline. Synthetic seed data and
            short smoke checks are hidden from the leaderboard.
          </p>
        </div>

        <div className="run-summary" aria-label="Current best result">
          <span className="summary-label">Best current row</span>
          <strong>{bestQuality?.variant_name ?? "No publishable result"}</strong>
          <dl>
            <div>
              <dt>Quality</dt>
              <dd>{formatPercent(bestQuality ? qualityValue(bestQuality) : null)}</dd>
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
          label="Top Quality"
          value={formatPercent(bestQuality ? qualityValue(bestQuality) : null)}
          detail={bestQuality ? protocolLabel(bestQuality) : "No score"}
          icon={<Scale size={18} />}
        />
        <MetricCell
          label="Fastest Full Run"
          value={formatDuration(fastestRun?.benchmark_runtime_seconds)}
          detail={fastestRun?.variant_name ?? "n/a"}
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
        <FilterPanel
          filters={filters}
          options={options}
          hiddenSyntheticCount={hiddenSyntheticCount}
          hiddenSmokeCount={hiddenSmokeCount}
          onChange={setFilters}
          onReset={() => setFilters(emptyFilters)}
        />

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

          <LeaderboardTable
            rows={rankedRows}
            activeVariantId={activeRow?.variant_id}
            onActivate={setActiveVariantId}
            onOpenRaw={setDrawerRow}
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
            text="Global MMLU Lite, IFBench, and BFCL v4 are the publishable quality protocols for the current workflow."
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
        <span>LA</span>
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
    <aside className="filter-panel" aria-label="Leaderboard controls">
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
          label="Quantization"
          value={filters.quantization}
          options={options.quantizations}
          onChange={(quantization) => onChange({ ...filters, quantization })}
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
    </aside>
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
}: {
  rows: LeaderboardRow[];
  activeVariantId?: string;
  onActivate: (variantId: string) => void;
  onOpenRaw: (row: LeaderboardRow) => void;
}) {
  const metricColumns = metricColumnsFor(rows);
  const columnCount = 5 + metricColumns.length;

  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            <th className="rank-heading">#</th>
            <th>Model</th>
            <th>Backend</th>
            <th>Protocol</th>
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
                <strong>{row.variant_name}</strong>
                <span>
                  {row.family} · {formatParameter(row.parameter_size_b)} · {row.quantization}
                </span>
              </td>
              <td>
                <span className="table-pill">{row.backend_name ?? "unknown"}</span>
              </td>
              <td>{protocolLabel(row)}</td>
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

function SelectedRunPanel({
  row,
  onOpenRaw,
}: {
  row: LeaderboardRow | null;
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
        <h2 id="selected-run-title">{row.variant_name}</h2>
      </div>

      <div className="score-display">
        <span>{protocolLabel(row)}</span>
        <strong>{formatPercent(qualityValue(row))}</strong>
      </div>

      <dl className="detail-list">
        <DetailItem label="Runtime" value={formatDuration(row.benchmark_runtime_seconds)} />
        <DetailItem label="Secondary score" value={formatPercent(secondaryQualityValue(row))} />
        <DetailItem label="Backend" value={row.backend_name ?? "unknown"} />
        <DetailItem label="API model" value={apiModel(row) ?? row.base_model_name} />
        <DetailItem label="Run UUID" value={shortId(row.run_uuid)} />
      </dl>

      <button className="text-button" type="button" onClick={() => onOpenRaw(row)}>
        Open metadata
        <ArrowUpRight size={15} aria-hidden="true" />
      </button>
    </section>
  );
}

function MetricCoveragePanel({ rows }: { rows: LeaderboardRow[] }) {
  const metrics = [
    ["GMMLU Lite", hasMetric(rows, "global_mmlu_lite_pass_at_1")],
    ["IFBench", hasMetric(rows, "ifbench_prompt_level_loose")],
    ["BFCL v4", hasMetric(rows, "bfcl_v4_selected_accuracy")],
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
      (filters.quantization === "all" || row.quantization === filters.quantization) &&
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
    quantizations: unique(rows.map((row) => row.quantization)),
    backends: unique(rows.map((row) => row.backend_name ?? "unknown")),
    hardware: unique(rows.map((row) => row.hardware_accelerator ?? "unknown")),
  };
}

function metricColumnsFor(rows: LeaderboardRow[]): MetricColumn[] {
  const columns: MetricColumn[] = [];

  if (hasMetric(rows, "global_mmlu_lite_pass_at_1")) {
    columns.push({
      key: "global_mmlu_lite_pass_at_1",
      label: "GMMLU Lite",
      render: (row) => formatPercent(row.global_mmlu_lite_pass_at_1),
      primary: true,
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

function rankRows(rows: LeaderboardRow[]) {
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

function hasMetric(rows: LeaderboardRow[], key: keyof LeaderboardRow) {
  return rows.some((row) => numeric(row[key]) !== null);
}

function qualityValue(row: LeaderboardRow) {
  return (
    numeric(row.global_mmlu_lite_pass_at_1) ??
    numeric(row.ifbench_prompt_level_loose) ??
    numeric(row.bfcl_v4_selected_accuracy)
  );
}

function secondaryQualityValue(row: LeaderboardRow) {
  return (
    numeric(row.global_mmlu_lite_micro_pass_at_1) ??
    numeric(row.ifbench_instruction_level_loose) ??
    numeric(row.bfcl_v4_non_live_accuracy) ??
    numeric(row.global_mmlu_lite_invalid_rate)
  );
}

function protocolLabel(row: LeaderboardRow) {
  if (row.global_mmlu_lite_pass_at_1 !== null && row.global_mmlu_lite_pass_at_1 !== undefined) {
    return "Global MMLU Lite";
  }
  if (row.ifbench_prompt_level_loose !== null && row.ifbench_prompt_level_loose !== undefined) {
    return "IFBench";
  }
  if (row.bfcl_v4_selected_accuracy !== null && row.bfcl_v4_selected_accuracy !== undefined) {
    return "BFCL v4";
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
  const metadata = safeMetadata(row);
  if (!metadata || typeof metadata !== "object") return null;
  const variantConfig = (metadata as { variant_config?: unknown }).variant_config;
  if (!variantConfig || typeof variantConfig !== "object") return null;
  const apiModelValue = (variantConfig as { api_model?: unknown }).api_model;
  return typeof apiModelValue === "string" ? apiModelValue : null;
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
