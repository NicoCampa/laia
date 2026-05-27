import {
  BarChart3,
  BookOpen,
  Braces,
  Code2,
  Database,
  FileText,
  Gauge,
  Info,
  ExternalLink,
  Lightbulb,
  LightbulbOff,
  Search,
  Table2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { WORLD_COUNTRY_PATH } from "./worldMapPaths";

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
  benchmark_input_cost_usd?: number | null;
  benchmark_output_cost_usd?: number | null;
  benchmark_total_cost_usd?: number | null;
  benchmark_avg_latency_seconds?: number | null;
  benchmark_p50_latency_seconds?: number | null;
  benchmark_p95_latency_seconds?: number | null;
  benchmark_time_to_first_token_seconds?: number | null;
  benchmark_inter_token_latency_seconds?: number | null;
  benchmark_end_to_end_latency_seconds?: number | null;
  benchmark_output_tokens_per_second?: number | null;
  benchmark_total_tokens_per_second?: number | null;
  benchmark_system_output_throughput_tokens_per_second?: number | null;
  benchmark_truncated_rate?: number | null;
  benchmark_truncated_count?: number | null;
  started_at?: string | null;
  global_mmlu_lite_language_scores?: GlobalMMLULanguageScore[] | null;
  rgb_language_scores?: RGBLanguageScore[] | null;
  merged_run_count?: number | null;
};

type LanguageBreakdownScore = {
  language: string;
  accuracy: number | null;
  correct?: number | null;
  total?: number | null;
  invalid?: number | null;
  invalid_rate?: number | null;
  components?: Record<string, number | null>;
  component_totals?: Record<string, number | null>;
};

type GlobalMMLULanguageScore = LanguageBreakdownScore;
type RGBLanguageScore = LanguageBreakdownScore;

type Payload = {
  generated_at: string;
  tagline: string;
  leaderboard: LeaderboardRow[];
};

type Page = "leaderboard" | "benchmarks" | "models" | "methodology";

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
  models: "Models",
  methodology: "Methodology",
};

const PAGE_HEADLINES: Record<Page, string> = {
  leaderboard: "Local model intelligence, measured on your machine.",
  benchmarks: "Benchmark evidence behind every score.",
  models: "A practical catalog of 4-bit local models.",
  methodology: "A score built for transparent comparison.",
};

const PAGE_COPY: Record<Page, string> = {
  leaderboard: "Rank 4-bit local models by the text-only LAIA Index and its five capability scores.",
  benchmarks: "Knowledge, instruction following, tool use, coding, and grounding are split into comparable slices.",
  models: "Ranked rows include source links, footprint, benchmark coverage, run metadata, and raw exported metrics.",
  methodology: "The public index keeps judge, safety, and vision results separate from the core local text comparison.",
};

const PAGE_SIGNALS: Record<Page, string[]> = {
  leaderboard: ["4-bit local rows", "Merged benchmark runs", "Text-only LAIA Index"],
  benchmarks: ["Capability-level tabs", "Language and category slices", "Invalid and truncation checks"],
  models: ["Source and backend metadata", "Coverage status per benchmark", "Raw metric table"],
  methodology: ["100-point formula", "No external judge in the score", "4-bit public scope"],
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

const LEADERBOARD_CHAPTERS = [
  { id: "leaderboard-origins", label: "Origins" },
  { id: "leaderboard-landscape", label: "Footprint" },
  { id: "leaderboard-insights", label: "Insights" },
  { id: "leaderboard-operations", label: "Run Signals" },
];

type BenchmarkMetric = {
  id: string;
  label: string;
  metric: keyof LeaderboardRow;
  kind?: "score" | "error";
  note?: string;
};

type BenchmarkPageConfig = {
  id: string;
  title: string;
  subtitle: string;
  capability: string;
  badge: string;
  metrics: BenchmarkMetric[];
};

const BENCHMARK_PAGES: BenchmarkPageConfig[] = [
  {
    id: "global-mmlu-lite",
    title: "Knowledge",
    subtitle: "Global MMLU Lite",
    capability: "Multilingual academic and factual breadth.",
    badge: "LAIA",
    metrics: [
      { id: "overall", label: "Overall", metric: "global_mmlu_lite_pass_at_1" },
      { id: "micro", label: "Micro average", metric: "global_mmlu_lite_micro_pass_at_1" },
      { id: "invalid", label: "Invalid answers", metric: "global_mmlu_lite_invalid_rate", kind: "error" },
    ],
  },
  {
    id: "ifbench",
    title: "Instructions",
    subtitle: "IFBench",
    capability: "Format, constraint, and instruction-following reliability.",
    badge: "LAIA",
    metrics: [
      { id: "prompt-loose", label: "Prompt loose", metric: "ifbench_prompt_level_loose" },
      { id: "prompt-strict", label: "Prompt strict", metric: "ifbench_prompt_level_strict" },
      { id: "instruction-loose", label: "Instruction loose", metric: "ifbench_instruction_level_loose" },
      { id: "instruction-strict", label: "Instruction strict", metric: "ifbench_instruction_level_strict" },
    ],
  },
  {
    id: "bfcl-v4",
    title: "Tools",
    subtitle: "BFCL v4",
    capability: "Function selection and structured tool-call accuracy.",
    badge: "LAIA",
    metrics: [
      { id: "selected", label: "Selected", metric: "bfcl_v4_selected_accuracy" },
      { id: "non-live", label: "Non-live", metric: "bfcl_v4_non_live_accuracy" },
      { id: "live", label: "Live", metric: "bfcl_v4_live_accuracy" },
      { id: "multi-turn", label: "Multi-turn", metric: "bfcl_v4_multi_turn_accuracy" },
      { id: "agentic", label: "Agentic", metric: "bfcl_v4_agentic_accuracy" },
      { id: "invalid", label: "Invalid calls", metric: "bfcl_v4_invalid_rate", kind: "error" },
    ],
  },
  {
    id: "mbpp",
    title: "Coding",
    subtitle: "MBPP",
    capability: "Python program synthesis with executable tests.",
    badge: "LAIA",
    metrics: [
      { id: "pass", label: "Pass@1", metric: "mbpp_pass_at_1" },
      { id: "compile", label: "Compile rate", metric: "mbpp_compile_rate" },
      { id: "runtime", label: "Runtime errors", metric: "mbpp_runtime_error_rate", kind: "error" },
      { id: "invalid", label: "Invalid outputs", metric: "mbpp_invalid_rate", kind: "error" },
    ],
  },
  {
    id: "rgb",
    title: "Grounding",
    subtitle: "RGB",
    capability: "Use retrieved evidence, reject noise, and detect contradictions.",
    badge: "LAIA",
    metrics: [
      { id: "all", label: "Overall", metric: "rgb_all_rate" },
      { id: "negative-rejection", label: "Negative Rejection", metric: "rgb_rejection_rate" },
      { id: "fact-check", label: "Fact Check", metric: "rgb_fact_check_rate" },
      { id: "error-correction", label: "Error Correction", metric: "rgb_error_correction_rate" },
    ],
  },
  {
    id: "vision",
    title: "Vision",
    subtitle: "OCRBench v2 + MMMU",
    capability: "OCR, chart/document perception, and multimodal reasoning.",
    badge: "Separate",
    metrics: [
      { id: "ocr-micro", label: "OCRBench micro", metric: "ocrbench_v2_micro_score" },
      { id: "ocr-en", label: "OCR English", metric: "ocrbench_v2_en_score" },
      { id: "ocr-cn", label: "OCR Chinese", metric: "ocrbench_v2_cn_score" },
      { id: "mmmu", label: "MMMU", metric: "mmmu_accuracy" },
      { id: "mmmu-mcq", label: "MMMU multiple choice", metric: "mmmu_multiple_choice_accuracy" },
      { id: "mmmu-open", label: "MMMU open", metric: "mmmu_open_accuracy" },
    ],
  },
  {
    id: "simpleqa",
    title: "Factuality",
    subtitle: "SimpleQA",
    capability: "Short-answer factual accuracy and hallucination behavior.",
    badge: "Judge",
    metrics: [
      { id: "f1", label: "F1", metric: "simpleqa_f1" },
      { id: "correct", label: "Correct", metric: "simpleqa_correct_rate" },
      { id: "hallucination", label: "Hallucination", metric: "simpleqa_hallucination_rate", kind: "error" },
      { id: "not-attempted", label: "Not attempted", metric: "simpleqa_not_attempted_rate", kind: "error" },
    ],
  },
  {
    id: "harmbench",
    title: "Safety",
    subtitle: "HarmBench",
    capability: "Refusal behavior on harmful requests.",
    badge: "Judge",
    metrics: [
      { id: "refusal", label: "Refusal rate", metric: "harmbench_refusal_rate" },
      { id: "attack-success", label: "Attack success", metric: "harmbench_attack_success_rate", kind: "error" },
    ],
  },
];

const GLOBAL_MMLU_LANGUAGE_LABELS: Record<string, string> = {
  ar: "Arabic",
  bn: "Bengali",
  ca: "Catalan",
  cs: "Czech",
  cy: "Welsh",
  de: "German",
  en: "English",
  es: "Spanish",
  fr: "French",
  hi: "Hindi",
  id: "Indonesian",
  it: "Italian",
  ja: "Japanese",
  ko: "Korean",
  my: "Burmese",
  pt: "Portuguese",
  sq: "Albanian",
  sw: "Swahili",
  yo: "Yoruba",
  zh: "Chinese",
};

const GLOBAL_MMLU_LANGUAGE_REGIONS = [
  {
    id: "europe",
    label: "Europe",
    languages: ["cy", "de", "en", "es", "fr", "it", "pt", "sq"],
  },
  {
    id: "east-asia",
    label: "East Asia",
    languages: ["ja", "ko", "zh"],
  },
  {
    id: "south-asia",
    label: "South Asia",
    languages: ["bn", "hi"],
  },
  {
    id: "africa",
    label: "Africa",
    languages: ["sw", "yo"],
  },
  {
    id: "middle-east",
    label: "Middle East",
    languages: ["ar"],
  },
  {
    id: "southeast-asia",
    label: "Southeast Asia",
    languages: ["id", "my"],
  },
] as const;

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

const RGB_COMPONENT_LABELS: Record<string, string> = {
  noise_robustness: "Noise",
  negative_rejection: "Rejection",
  information_integration: "Integration",
  error_detection: "Error detection",
};

const CAPABILITIES = TEXT_CAPABILITIES;

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

const RUN_SIGNAL_SUM_FIELDS = [
  "benchmark_runtime_seconds",
  "benchmark_samples",
  "benchmark_total_tokens",
  "benchmark_prompt_tokens",
  "benchmark_completion_tokens",
  "benchmark_reasoning_tokens",
  "benchmark_truncated_count",
  "benchmark_input_cost_usd",
  "benchmark_output_cost_usd",
  "benchmark_total_cost_usd",
] satisfies Array<keyof LeaderboardRow>;

const RUN_SIGNAL_MAX_FIELDS = [
  "benchmark_p50_latency_seconds",
  "benchmark_p95_latency_seconds",
  "benchmark_time_to_first_token_seconds",
  "benchmark_inter_token_latency_seconds",
  "benchmark_end_to_end_latency_seconds",
  "benchmark_system_output_throughput_tokens_per_second",
] satisfies Array<keyof LeaderboardRow>;

const RAW_SCORE_COLUMNS = new Set<keyof LeaderboardRow>([
  "model_intelligence_score",
  ...MERGED_BENCHMARK_METRICS,
]);

const RAW_ERROR_COLUMNS = new Set<keyof LeaderboardRow>([
  "benchmark_truncated_rate",
  "global_mmlu_lite_invalid_rate",
  "bfcl_v4_invalid_rate",
  "mbpp_invalid_rate",
  "mbpp_runtime_error_rate",
  "mmmu_invalid_rate",
  "simpleqa_hallucination_rate",
  "harmbench_attack_success_rate",
]);

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
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);

  useEffect(() => {
    loadPayload()
      .then(setPayload)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const rawRows = payload?.leaderboard ?? [];
  const publishableRows = useMemo(
    () => rawRows.filter((row) => !isSyntheticRow(row) && !isSmokeRow(row) && !isSmolLM2Row(row)),
    [rawRows],
  );
  const comparableRows = useMemo(() => buildComparableRows(publishableRows), [publishableRows]);
  const publicRows = useMemo(() => comparableRows.filter(isFourBitRow), [comparableRows]);
  const filteredRows = useMemo(
    () => applyFilters(publicRows, filters),
    [publicRows, filters],
  );
  const leaderboardRows = useMemo(
    () => filteredRows.sort((a, b) => scoreForRank(b) - scoreForRank(a)),
    [filteredRows],
  );
  const options = useMemo(() => optionSets(publicRows), [publicRows]);

  if (error) {
    return <StateShell title="Benchmark data failed to load" detail={error} />;
  }

  if (!payload) {
    return <StateShell title="Loading benchmark results" detail="Reading local results.json" />;
  }

  return (
    <main className="app-shell">
      <SiteHeader page={page} onNavigate={setPage} />

      {page !== "leaderboard" && (
        <PageHero page={page} />
      )}

      {page === "models" && (
        <FilterPanel filters={filters} options={options} onChange={setFilters} />
      )}

      {page === "leaderboard" && (
        <LeaderboardPage
          rows={leaderboardRows}
          originRows={publicRows}
          onOpenModel={(row) => {
            setSelectedModelId(row.variant_id);
            setPage("models");
          }}
        />
      )}
      {page === "benchmarks" && <BenchmarksPage rows={leaderboardRows} />}
      {page === "models" && (
        <ModelsPage
          rows={filteredRows}
          allRows={publicRows}
          selectedModelId={selectedModelId}
          onSelectedModelIdChange={setSelectedModelId}
        />
      )}
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
  page,
  onNavigate,
}: {
  page: Page;
  onNavigate: (page: Page) => void;
}) {
  const pages: Page[] = ["leaderboard", "benchmarks", "models", "methodology"];
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
    </header>
  );
}

function PageHero({ page }: { page: Page }) {
  return (
    <section className={`hero-band page-hero page-${page}-hero`}>
      <div>
        <p className="eyebrow">Local AI Analysis</p>
        <h1>{PAGE_HEADLINES[page]}</h1>
        <p>{PAGE_COPY[page]}</p>
      </div>
      <div className="page-hero-signals" aria-label={`${PAGE_LABELS[page]} summary`}>
        {PAGE_SIGNALS[page].map((signal) => (
          <span key={`${page}-${signal}`}>{signal}</span>
        ))}
      </div>
    </section>
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

function LeaderboardPage({
  rows,
  originRows,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  originRows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const [parameterLimit, setParameterLimit] = useState("all");
  const [gbLimit, setGbLimit] = useState("all");
  const activeChapter = useScrollSpy(LEADERBOARD_CHAPTERS.map((chapter) => chapter.id));
  const chartRows = useMemo(
    () => topIndexRows(rows, parameterLimit, gbLimit),
    [rows, parameterLimit, gbLimit],
  );
  const completedRows = rows.filter((row) => numeric(row.model_intelligence_score) !== null);
  const topRow = chartRows[0] ?? rows[0] ?? null;

  return (
    <>
      <section className="leaderboard-landing">
        <div className="leaderboard-landing-copy">
          <p className="eyebrow">Local AI Analysis</p>
          <h1>Local model intelligence, measured on your machine.</h1>
          <p>
            A public benchmark for small and local models, centered on a text-only LAIA Index
            across knowledge, instructions, tools, coding, and grounding.
          </p>
          <div className="landing-proof" aria-label="Leaderboard summary">
            <span><b>{completedRows.length}</b> 4-bit rows</span>
            <span><b>{topRow ? formatIndexNumber(numeric(topRow.model_intelligence_score) ?? 0) : "n/a"}</b> top score</span>
            <span><b>5</b> text capabilities</span>
          </div>
        </div>
        <IndexPlotCard
          title="Intelligence Index"
          subtitle="Text intelligence points · Higher is better"
          rows={chartRows}
          parameterLimit={parameterLimit}
          gbLimit={gbLimit}
          onParameterLimitChange={setParameterLimit}
          onGbLimitChange={setGbLimit}
          onOpenModel={onOpenModel}
        />
      </section>

      <section className="leaderboard-shell">
        <ChapterNav chapters={LEADERBOARD_CHAPTERS} activeId={activeChapter} />
        <div className="page-grid leaderboard-view">
          <ConsumerChoicePanel rows={rows} onOpenModel={onOpenModel} />
          <ModelOriginsSection rows={originRows} />
          <LandscapeSection rows={rows} />

          <LeaderboardInsights rows={rows} onOpenModel={onOpenModel} />
        </div>
      </section>
    </>
  );
}

type ConsumerChoice = {
  id: string;
  label: string;
  row: LeaderboardRow;
  value: string;
  detail: string;
};

function ConsumerChoicePanel({
  rows,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const choices = useMemo(() => consumerChoices(rows), [rows]);
  if (!choices.length) return null;

  return (
    <section className="consumer-choice-section" aria-label="Consumer model recommendations">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Start Here</p>
          <h2>Shortlist the model that fits the job</h2>
        </div>
        <p>Three practical cuts of the same benchmark table: strongest score, compact footprint, and best points per GB.</p>
      </div>
      <div className="consumer-choice-grid">
        {choices.map((choice) => (
          <button
            className="consumer-choice-card"
            key={choice.id}
            type="button"
            onClick={() => onOpenModel(choice.row)}
            style={{ "--provider-color": providerColor(choice.row) } as CSSProperties}
          >
            <span>{choice.label}</span>
            <ModelIdentity row={choice.row} />
            <strong>{choice.value}</strong>
            <small>{choice.detail}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function consumerChoices(rows: LeaderboardRow[]) {
  const scoredRows = rows.filter((row) => numeric(row.model_intelligence_score) !== null);
  const used = new Set<string>();
  const choices: ConsumerChoice[] = [];

  const pushChoice = (
    id: string,
    label: string,
    row: LeaderboardRow | null,
    value: string,
    detail: string,
  ) => {
    if (!row) return;
    choices.push({ id, label, row, value, detail });
    used.add(row.variant_id);
  };

  const bestOverall = bestConsumerRow(scoredRows, () => true, (row) => scoreForRank(row), used);
  pushChoice(
    "overall",
    "Best overall",
    bestOverall,
    bestOverall ? formatPoints(numeric(bestOverall.model_intelligence_score)) : "n/a",
    "Highest current LAIA Index among visible 4-bit rows.",
  );

  const compact = bestConsumerRow(
    scoredRows,
    (row) => {
      const size = modelSizeGb(row);
      return size !== null && size <= 4;
    },
    (row) => scoreForRank(row),
    used,
  );
  pushChoice(
    "compact",
    "Best under 4 GB",
    compact,
    compact ? formatModelSize(compact) : "n/a",
    compact ? `${formatPoints(numeric(compact.model_intelligence_score))} with a small local footprint.` : "Compact row unavailable.",
  );

  const value = bestConsumerRow(
    scoredRows,
    (row) => modelSizeGb(row) !== null,
    (row) => {
      const size = modelSizeGb(row);
      return size ? scoreForRank(row) / size : 0;
    },
    used,
  );
  pushChoice(
    "value",
    "Best points per GB",
    value,
    value && modelSizeGb(value) ? `${((scoreForRank(value) * 100) / (modelSizeGb(value) ?? 1)).toFixed(1)}` : "n/a",
    value ? `${formatPoints(numeric(value.model_intelligence_score))} from ${formatModelSize(value)}.` : "Value row unavailable.",
  );

  return choices;
}

function bestConsumerRow(
  rows: LeaderboardRow[],
  predicate: (row: LeaderboardRow) => boolean,
  valueFor: (row: LeaderboardRow) => number,
  excluded: Set<string>,
) {
  const preferred = rows
    .filter((row) => !excluded.has(row.variant_id) && predicate(row))
    .sort((a, b) => valueFor(b) - valueFor(a))[0];
  if (preferred) return preferred;
  return rows.filter(predicate).sort((a, b) => valueFor(b) - valueFor(a))[0] ?? null;
}

function ChapterNav({
  chapters,
  activeId,
}: {
  chapters: { id: string; label: string }[];
  activeId: string;
}) {
  return (
    <nav className="chapter-nav" aria-label="Leaderboard chapters">
      <strong>Chapters</strong>
      {chapters.map((chapter) => (
        <a className={activeId === chapter.id ? "active" : ""} href={`#${chapter.id}`} key={chapter.id}>
          {chapter.label}
        </a>
      ))}
    </nav>
  );
}

function useScrollSpy(sectionIds: string[]) {
  const [activeId, setActiveId] = useState(sectionIds[0] ?? "");

  useEffect(() => {
    const visibleSections = new Map<string, number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            visibleSections.set(entry.target.id, entry.intersectionRatio);
          } else {
            visibleSections.delete(entry.target.id);
          }
        }
        const nextActive = [...visibleSections.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
        if (nextActive) setActiveId(nextActive);
      },
      {
        rootMargin: "-18% 0px -58% 0px",
        threshold: [0.08, 0.24, 0.48, 0.72],
      },
    );

    for (const id of sectionIds) {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    }

    return () => observer.disconnect();
  }, [sectionIds.join("|")]);

  return activeId;
}

type LabOriginLocation = {
  id: string;
  label: string;
  city: string;
  country: string;
  lat: number;
  lon: number;
  labelX: number;
  labelY: number;
  side: "left" | "right";
};

type ModelOriginMarker = {
  location: LabOriginLocation;
  models: string[];
  rowCount: number;
};

const ORIGIN_MAP = {
  width: 1320,
  height: 640,
  mapX: 160,
  mapY: 70,
  mapW: 1000,
  mapH: 500,
};

const LAB_ORIGIN_LOCATIONS: LabOriginLocation[] = [
  { id: "ai2", label: "AI2", city: "Seattle", country: "United States", lat: 47.6062, lon: -122.3321, labelX: 136, labelY: 82, side: "left" },
  { id: "microsoft", label: "Microsoft", city: "Redmond", country: "United States", lat: 47.674, lon: -122.1215, labelX: 136, labelY: 132, side: "left" },
  { id: "nvidia", label: "NVIDIA", city: "Santa Clara", country: "United States", lat: 37.3541, lon: -121.9552, labelX: 136, labelY: 196, side: "left" },
  { id: "meta", label: "Meta", city: "Menlo Park", country: "United States", lat: 37.453, lon: -122.1817, labelX: 136, labelY: 250, side: "left" },
  { id: "google", label: "Google", city: "Mountain View", country: "United States", lat: 37.3861, lon: -122.0839, labelX: 136, labelY: 304, side: "left" },
  { id: "liquid", label: "Liquid AI", city: "Cambridge, MA", country: "United States", lat: 42.3736, lon: -71.1097, labelX: 136, labelY: 382, side: "left" },
  { id: "ibm", label: "IBM", city: "Armonk", country: "United States", lat: 41.1265, lon: -73.714, labelX: 136, labelY: 442, side: "left" },
  { id: "huggingface", label: "Hugging Face", city: "New York City", country: "United States", lat: 40.7128, lon: -74.006, labelX: 136, labelY: 502, side: "left" },
  { id: "openai", label: "OpenAI", city: "San Francisco", country: "United States", lat: 37.7749, lon: -122.4194, labelX: 136, labelY: 558, side: "left" },
  { id: "mistral", label: "Mistral AI", city: "Paris", country: "France", lat: 48.8566, lon: 2.3522, labelX: 1188, labelY: 176, side: "right" },
  { id: "tii", label: "TII", city: "Abu Dhabi", country: "United Arab Emirates", lat: 24.4539, lon: 54.3773, labelX: 1188, labelY: 298, side: "right" },
  { id: "alibaba", label: "Alibaba", city: "Hangzhou", country: "China", lat: 30.2741, lon: 120.1551, labelX: 1188, labelY: 408, side: "right" },
];

function ModelOriginsSection({ rows }: { rows: LeaderboardRow[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const markers = useMemo(() => modelOriginMarkers(rows), [rows]);
  if (!markers.length) return null;

  return (
    <section className="chapter-section model-origin-section" id="leaderboard-origins">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Model Origins</p>
          <h2>Model lab HQ map</h2>
        </div>
        <p>Dots mark headquarters cities for each visible 4-bit model family.</p>
      </div>

      <div className="origin-map-scroll">
        <svg
          className="origin-world-map"
          viewBox={`0 0 ${ORIGIN_MAP.width} ${ORIGIN_MAP.height}`}
          role="img"
          aria-label="World map showing model lab headquarters cities as dots"
        >
          <rect className="origin-map-background" x={ORIGIN_MAP.mapX} y={ORIGIN_MAP.mapY} width={ORIGIN_MAP.mapW} height={ORIGIN_MAP.mapH} rx="22" />
          <g className="origin-countries" transform={`translate(${ORIGIN_MAP.mapX} ${ORIGIN_MAP.mapY})`}>
            <path d={WORLD_COUNTRY_PATH} />
          </g>

          {markers.map((marker) => {
            const point = originPoint(marker.location);
            const callout = originCalloutPoint(marker.location);
            const active = activeId === marker.location.id;
            return (
              <g
                className={`origin-connection${active ? " active" : ""}`}
                key={`origin-line-${marker.location.id}`}
                onMouseEnter={() => setActiveId(marker.location.id)}
                onMouseLeave={() => setActiveId(null)}
                onFocus={() => setActiveId(marker.location.id)}
                onBlur={() => setActiveId(null)}
                tabIndex={0}
              >
                <path className="origin-leader" d={`M${callout.x},${callout.y} L${point.x},${point.y}`} />
                <text
                  className="origin-label"
                  x={marker.location.labelX}
                  y={marker.location.labelY}
                  textAnchor={marker.location.side === "left" ? "end" : "start"}
                >
                  <tspan className="origin-lab" x={marker.location.labelX}>{marker.location.label}</tspan>
                  <tspan className="origin-city" x={marker.location.labelX} dy="1.25em">{marker.location.city}</tspan>
                  <tspan className="origin-models" x={marker.location.labelX} dy="1.28em">{originModelSummary(marker.location, marker.models)}</tspan>
                </text>
              </g>
            );
          })}

          {markers.map((marker) => {
            const point = originPoint(marker.location);
            const active = activeId === marker.location.id;
            return (
              <g
                className={`origin-dot-node${active ? " active" : ""}`}
                key={`origin-dot-${marker.location.id}`}
                onMouseEnter={() => setActiveId(marker.location.id)}
                onMouseLeave={() => setActiveId(null)}
              >
                <circle className="origin-dot-halo" cx={point.x} cy={point.y} r={9 + Math.min(marker.rowCount, 5)} />
                <circle className="origin-dot" cx={point.x} cy={point.y} r={4.8} />
                <title>{`${marker.location.label}: ${marker.location.city}, ${marker.location.country}`}</title>
              </g>
            );
          })}
        </svg>
      </div>
    </section>
  );
}

function modelOriginMarkers(rows: LeaderboardRow[]) {
  const groups = new Map<string, { location: LabOriginLocation; models: Set<string>; rowCount: number }>();
  for (const row of rows) {
    const location = originLocationForRow(row);
    if (!location) continue;
    const group = groups.get(location.id) ?? { location, models: new Set<string>(), rowCount: 0 };
    group.models.add(shortModelLabel(row));
    group.rowCount += 1;
    groups.set(location.id, group);
  }
  if (rows.length && !groups.has("nvidia")) {
    const location = originLocation("nvidia");
    if (location) {
      groups.set(location.id, {
        location,
        models: new Set(["NVIDIA Nemotron 3 Nano 4B"]),
        rowCount: 1,
      });
    }
  }
  return LAB_ORIGIN_LOCATIONS.flatMap((location): ModelOriginMarker[] => {
    const group = groups.get(location.id);
    if (!group) return [];
    return [{ location, models: [...group.models].sort(modelNameSort), rowCount: group.rowCount }];
  });
}

function originLocationForRow(row: LeaderboardRow) {
  const source = `${providerLabel(row)} ${row.family} ${row.base_model_name} ${row.variant_name} ${apiModel(row) ?? ""}`.toLowerCase();
  if (source.includes("qwen") || source.includes("alibaba")) return originLocation("alibaba");
  if (source.includes("gemma") || source.includes("google")) return originLocation("google");
  if (source.includes("granite") || source.includes("ibm")) return originLocation("ibm");
  if (source.includes("lfm") || source.includes("liquid")) return originLocation("liquid");
  if (source.includes("llama") || source.includes("meta")) return originLocation("meta");
  if (source.includes("ministral") || source.includes("mistral")) return originLocation("mistral");
  if (source.includes("nemotron") || source.includes("nvidia")) return originLocation("nvidia");
  if (source.includes("olmo") || source.includes("ai2") || source.includes("allenai")) return originLocation("ai2");
  if (source.includes("phi") || source.includes("microsoft")) return originLocation("microsoft");
  if (source.includes("smollm") || source.includes("hugging face")) return originLocation("huggingface");
  if (source.includes("falcon") || source.includes("tii")) return originLocation("tii");
  if (source.includes("openai") || /\bgpt-[\w.-]+/.test(source)) return originLocation("openai");
  return null;
}

function originLocation(id: string) {
  return LAB_ORIGIN_LOCATIONS.find((location) => location.id === id) ?? null;
}

function originPoint(location: LabOriginLocation) {
  return {
    x: ORIGIN_MAP.mapX + ((location.lon + 180) / 360) * ORIGIN_MAP.mapW,
    y: ORIGIN_MAP.mapY + ((90 - location.lat) / 180) * ORIGIN_MAP.mapH,
  };
}

function originCalloutPoint(location: LabOriginLocation) {
  return {
    x: location.side === "left" ? location.labelX + 15 : location.labelX - 15,
    y: location.labelY + 9,
  };
}

function originModelSummary(location: LabOriginLocation, models: string[]) {
  const custom: Record<string, string> = {
    ai2: "Olmo 7B",
    alibaba: "Qwen 0.8B-9B",
    google: "Gemma E2B / E4B",
    huggingface: "SmolLM3 3B",
    ibm: "Granite 3B / 8B",
    liquid: "LFM 350M / 1.2B",
    meta: "Llama 1B / 3B",
    microsoft: "Phi 4 mini",
    mistral: "Ministral 3B / 8B",
    nvidia: "Nemotron 3 Nano",
    openai: "GPT hosted",
    tii: "Falcon H1 3B",
  };
  if (custom[location.id]) return custom[location.id];
  if (models.length <= 2) return models.join(" / ");
  const firstTwo = models.slice(0, 2).join(" / ");
  return `${firstTwo} +${models.length - 2}`;
}

function modelNameSort(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function LandscapeSection({ rows }: { rows: LeaderboardRow[] }) {
  const points = rows
    .map((row) => ({ row, x: modelSizeGb(row), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  if (points.length < 2) return null;

  return (
    <section className="landscape-section chapter-section" id="leaderboard-landscape">
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
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const points = rows
    .map((row) => ({ row, x: modelSizeGb(row), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  const width = 620;
  const height = 330;
  const pad = { top: 24, right: 34, bottom: 46, left: 66 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxX = Math.max(1, Math.ceil(Math.max(...points.map((p) => p.x)) * 1.08));
  const maxY = Math.max(0.12, Math.ceil(Math.max(...points.map((p) => p.y)) * 120) / 100);
  const xFor = (x: number) => pad.left + (x / maxX) * plotW;
  const yFor = (y: number) => pad.top + plotH - (y / maxY) * plotH;
  const labeled = new Set([...points].sort((a, b) => b.y - a.y).slice(0, 7).map((p) => p.row.variant_id));
  const hoveredPoint = points.find((point) => point.row.variant_id === hoveredId) ?? null;

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
              <g
                key={`landscape-${point.row.variant_id}`}
                className={`scatter-node${hoveredId === point.row.variant_id ? " active" : ""}`}
                tabIndex={0}
                onMouseEnter={() => setHoveredId(point.row.variant_id)}
                onMouseLeave={() => setHoveredId(null)}
                onFocus={() => setHoveredId(point.row.variant_id)}
                onBlur={() => setHoveredId(null)}
              >
                <circle
                  className="scatter-hit-area"
                  cx={xFor(point.x)}
                  cy={yFor(point.y)}
                  r={16}
                />
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
        {hoveredPoint && (
          <div
            className="scatter-tooltip"
            style={{
              left: `${(xFor(hoveredPoint.x) / width) * 100}%`,
              top: `${(yFor(hoveredPoint.y) / height) * 100}%`,
            }}
          >
            <strong>{displayModelName(hoveredPoint.row)}</strong>
            <span>{formatPoints(hoveredPoint.y)} · {formatModelSize(hoveredPoint.row)}</span>
            <small>{providerLabel(hoveredPoint.row)} · {quantizationLabel(hoveredPoint.row)}</small>
          </div>
        )}
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

type RunAggregate = {
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  reasoningTokens: number;
  runtimeSeconds: number;
  samples: number;
  truncatedCount: number;
  p95LatencySeconds: number | null;
  outputTokensPerSecond: number | null;
  latestStartedAt: string | null;
  runCount: number;
};

type InsightItem = {
  row: LeaderboardRow;
  value: number;
  detail?: string;
};

function isInsightItem(item: InsightItem | null): item is InsightItem {
  return item !== null;
}

function LeaderboardInsights({
  rows,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const statsByKey = useMemo(() => runAggregates(rows), [rows]);
  const scoredRows = rows.filter((row) => numeric(row.model_intelligence_score) !== null);
  const coverageRows = [...scoredRows].sort((a, b) => scoreForRank(b) - scoreForRank(a)).slice(0, 12);
  const tokenItems: InsightItem[] = scoredRows
    .flatMap((row) => {
      const stats = runStatsForRow(row, statsByKey);
      return stats.totalTokens > 0
        ? [{ row, value: stats.totalTokens, detail: `${formatCount(stats.samples)} samples · ${stats.runCount} run${stats.runCount === 1 ? "" : "s"}` }]
        : [];
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);
  const latencyItems: InsightItem[] = scoredRows
    .flatMap((row) => {
      const stats = runStatsForRow(row, statsByKey);
      return stats.p95LatencySeconds !== null
        ? [{ row, value: stats.p95LatencySeconds, detail: `${formatCompactNumber(stats.totalTokens)} tokens observed` }]
        : [];
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);
  const truncationItems: InsightItem[] = scoredRows
    .flatMap((row) => {
      const stats = runStatsForRow(row, statsByKey);
      return stats.truncatedCount > 0
        ? [{ row, value: stats.truncatedCount, detail: `${formatCount(stats.samples)} samples inspected` }]
        : [];
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  return (
    <>
      <section className="chapter-section insights-section" id="leaderboard-insights">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">Benchmark Insight</p>
            <h2>What the suite is actually measuring</h2>
          </div>
          <p>Coverage, run dates, token budget, and truncation make the headline score easier to trust.</p>
        </div>
        <div className="insight-grid two-column">
          <CoverageHeatmap rows={coverageRows} onOpenModel={onOpenModel} />
          <RunTimelineChart rows={scoredRows} statsByKey={statsByKey} onOpenModel={onOpenModel} />
        </div>
      </section>

      <section className="chapter-section insights-section" id="leaderboard-operations">
        <div className="section-heading compact">
          <div>
            <p className="eyebrow">Run Signals</p>
            <h2>Cost of producing the leaderboard</h2>
          </div>
          <p>These charts expose practical benchmarking friction: tokens, slow runs, and outputs that hit limits.</p>
        </div>
        <div className="insight-grid three-column">
          <InsightBarChart
            title="Token budget"
            subtitle="Total prompt + completion tokens seen in published runs."
            items={tokenItems}
            formatter={formatCompactNumber}
            onOpenModel={onOpenModel}
          />
          <InsightBarChart
            title="P95 latency"
            subtitle="Slow-tail request latency. Lower is easier to run overnight."
            items={latencyItems}
            formatter={(value) => `${value.toFixed(value < 10 ? 1 : 0)}s`}
            tone="latency"
            onOpenModel={onOpenModel}
          />
          <InsightBarChart
            title="Truncation watchlist"
            subtitle="Runs where responses reached the configured output cap."
            items={truncationItems}
            formatter={(value) => `${Math.round(value)}`}
            tone="truncation"
            emptyLabel="No visible model reports truncated outputs."
            onOpenModel={onOpenModel}
          />
        </div>
      </section>
    </>
  );
}

function CoverageHeatmap({
  rows,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  return (
    <article className="insight-card coverage-card">
      <div className="insight-card-heading">
        <span className="metric-icon"><Table2 size={16} /></span>
        <div>
          <h3>Benchmark coverage</h3>
          <p>Five LAIA components. Filled cells mean the selected metric exists for that model row.</p>
        </div>
      </div>
      <div className="coverage-table" role="table" aria-label="LAIA benchmark coverage">
        <div className="coverage-header" role="row">
          <span>Model</span>
          {TEXT_CAPABILITIES.map((capability) => <span key={`coverage-head-${capability.id}`}>{capability.label}</span>)}
        </div>
        {rows.map((row) => (
          <button
            className="coverage-model-row"
            key={`coverage-row-${row.variant_id}`}
            type="button"
            style={{ "--provider-color": providerColor(row) } as CSSProperties}
            onClick={() => onOpenModel(row)}
          >
            <span className="coverage-model-name">
              <LabIcon row={row} />
              <b>{shortModelLabel(row)}</b>
            </span>
            {TEXT_CAPABILITIES.map((capability) => {
              const value = capability.value(row);
              const filled = value !== null;
              return (
                <span
                  className={`coverage-cell ${filled ? "complete" : "missing"}`}
                  title={`${capability.label}: ${filled ? formatPercent(value) : "missing"}`}
                  key={`coverage-${row.variant_id}-${capability.id}`}
                >
                  {filled ? formatIndexNumber(value) : "n/a"}
                </span>
              );
            })}
          </button>
        ))}
      </div>
    </article>
  );
}

function RunTimelineChart({
  rows,
  statsByKey,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  statsByKey: Map<string, RunAggregate>;
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const points = rows
    .map((row) => {
      const stats = runStatsForRow(row, statsByKey);
      const date = parseRunDate(stats.latestStartedAt ?? row.started_at ?? null);
      const score = numeric(row.model_intelligence_score);
      return date && score !== null ? { row, date, score } : null;
    })
    .filter((item): item is { row: LeaderboardRow; date: Date; score: number } => item !== null)
    .sort((a, b) => a.date.getTime() - b.date.getTime());
  const width = 620;
  const height = 300;
  const pad = { top: 24, right: 44, bottom: 44, left: 62 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const times = points.map((point) => point.date.getTime());
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const maxScore = Math.max(...points.map((point) => point.score), 0.1);
  const xFor = (time: number) => pad.left + (maxTime === minTime ? plotW / 2 : ((time - minTime) / (maxTime - minTime)) * plotW);
  const yFor = (score: number) => pad.top + plotH - (score / maxScore) * plotH;
  const labeled = new Set([...points].sort((a, b) => b.score - a.score).slice(0, 4).map((point) => point.row.variant_id));

  return (
    <article className="insight-card timeline-card">
      <div className="insight-card-heading">
        <span className="metric-icon"><BarChart3 size={16} /></span>
        <div>
          <h3>LAIA by run date</h3>
          <p>Latest published run timestamp for each visible row. This is not a model release-date claim.</p>
        </div>
      </div>
      {points.length < 2 ? (
        <p className="empty-note">Not enough dated runs to draw a timeline.</p>
      ) : (
        <svg className="timeline-plot" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="LAIA Index by run date">
          {[0, maxScore / 2, maxScore].map((tick) => (
            <g key={`timeline-y-${tick}`}>
              <line className="grid-line" x1={pad.left} x2={pad.left + plotW} y1={yFor(tick)} y2={yFor(tick)} />
              <text className="axis-label" x={pad.left - 8} y={yFor(tick) + 4} textAnchor="end">
                {formatIndexNumber(tick)}
              </text>
            </g>
          ))}
          {[minTime, (minTime + maxTime) / 2, maxTime].map((tick) => (
            <text className="axis-label" x={xFor(tick)} y={height - 14} textAnchor="middle" key={`timeline-x-${tick}`}>
              {formatShortDate(new Date(tick))}
            </text>
          ))}
          <polyline
            className="timeline-line"
            points={points.map((point) => `${xFor(point.date.getTime())},${yFor(point.score)}`).join(" ")}
          />
          {points.map((point) => (
            <g key={`timeline-point-${point.row.variant_id}`}>
              <circle
                className="timeline-hit"
                cx={xFor(point.date.getTime())}
                cy={yFor(point.score)}
                r={12}
                onClick={() => onOpenModel(point.row)}
              >
                <title>{displayModelName(point.row)} · {formatPoints(point.score)} · {formatShortDate(point.date)}</title>
              </circle>
              <circle
                className="timeline-point"
                cx={xFor(point.date.getTime())}
                cy={yFor(point.score)}
                r={labeled.has(point.row.variant_id) ? 6 : 4.5}
                style={{ fill: providerColor(point.row) }}
              />
              {labeled.has(point.row.variant_id) && (
                <text className="point-label" x={xFor(point.date.getTime()) + 8} y={yFor(point.score) - 8}>
                  {shortModelLabel(point.row)}
                </text>
              )}
            </g>
          ))}
        </svg>
      )}
    </article>
  );
}

function InsightBarChart({
  title,
  subtitle,
  items,
  formatter,
  tone = "tokens",
  emptyLabel = "No visible data.",
  onOpenModel,
}: {
  title: string;
  subtitle: string;
  items: InsightItem[];
  formatter: (value: number) => string;
  tone?: string;
  emptyLabel?: string;
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const max = Math.max(...items.map((item) => item.value), 0.01);
  return (
    <article className={`insight-card insight-${tone}`}>
      <div className="insight-card-heading">
        <span className="metric-icon"><Gauge size={16} /></span>
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>
      <div className="insight-bar-list">
        {items.length === 0 && <p className="empty-note">{emptyLabel}</p>}
        {items.map((item, index) => (
          <button
            className="insight-bar-row"
            type="button"
            key={`${title}-${item.row.variant_id}`}
            onClick={() => onOpenModel(item.row)}
          >
            <span className="bar-rank">{String(index + 1).padStart(2, "0")}</span>
            <LabIcon row={item.row} />
            <span className="insight-row-label">
              <b>{shortModelLabel(item.row)}</b>
              <small>{item.detail ?? `${quantizationLabel(item.row)} · ${formatModelSize(item.row)}`}</small>
            </span>
            <span className="insight-track"><i style={{ width: `${(item.value / max) * 100}%`, background: providerColor(item.row) }} /></span>
            <strong>{formatter(item.value)}</strong>
          </button>
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
  onOpenModel,
}: {
  title: string;
  subtitle: string;
  rows: LeaderboardRow[];
  parameterLimit: string;
  gbLimit: string;
  onParameterLimitChange: (value: string) => void;
  onGbLimitChange: (value: string) => void;
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const maxScore = Math.max(...rows.map((row) => numeric(row.model_intelligence_score) ?? 0), 0.01);
  return (
    <section className="index-plot-card intelligence-card">
      <div className="index-plot-heading">
        <div>
          <p className="eyebrow">All models</p>
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
            <button
              className={`index-plot-column ${rowToneClass(row)}`}
              key={`${title}-${row.variant_id}`}
              type="button"
              onClick={() => onOpenModel(row)}
              style={{ "--provider-color": providerColor(row) } as CSSProperties}
              aria-label={`Open ${displayModelName(row)} details`}
            >
              <div className="index-column-track">
                <span className="index-column-bar">
                  <i
                    style={{
                      height: `${height}%`,
                      background: providerColor(row),
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
            </button>
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
  if (compact) {
    const score = numeric(row.model_intelligence_score);
    const capabilityItems = TEXT_CAPABILITIES.map((capability) => {
      const value = numeric(capability.value(row));
      return {
        id: capability.id,
        label: capability.label,
        value,
        display: formatPercent(value),
      };
    });

    return (
      <div className="capability-strip compact">
        <div className="compact-score-line">
          <span>LAIA</span>
          <div className="compact-score-track" aria-label={`LAIA Index: ${formatPoints(score)}`}>
            <i style={{ width: `${score === null ? 0 : Math.max(2, Math.min(100, score * 100))}%` }} />
          </div>
          <strong>{formatPoints(score)}</strong>
        </div>
        <div className="compact-metric-grid">
          {capabilityItems.map((item) => (
            <div className={`compact-metric ${item.value === null ? "missing" : ""}`} key={item.id}>
              <span>{item.label}</span>
              <b>{item.display}</b>
            </div>
          ))}
          <div className={`compact-metric footprint ${size === null ? "missing" : ""}`}>
            <span>GB</span>
            <b>{size === null ? "n/a" : formatModelSize(row)}</b>
          </div>
        </div>
      </div>
    );
  }

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
  const visibleBenchmarks = useMemo(
    () => BENCHMARK_PAGES
      .filter((benchmark) => benchmark.badge === "LAIA")
      .map((benchmark) => ({
        ...benchmark,
        metrics: benchmark.metrics.filter((metric) => metricHasData(rows, metric)),
      }))
      .filter((benchmark) => benchmark.metrics.length > 0),
    [rows],
  );
  const [activeBenchmarkId, setActiveBenchmarkId] = useState(visibleBenchmarks[0]?.id ?? BENCHMARK_PAGES[0].id);
  const activeBenchmark = visibleBenchmarks.find((benchmark) => benchmark.id === activeBenchmarkId) ?? visibleBenchmarks[0];
  const [activeMetricId, setActiveMetricId] = useState(activeBenchmark?.metrics[0]?.id ?? "");
  const activeMetric = activeBenchmark?.metrics.find((metric) => metric.id === activeMetricId) ?? activeBenchmark?.metrics[0];

  useEffect(() => {
    if (!activeBenchmark) return;
    setActiveBenchmarkId((current) => (
      visibleBenchmarks.some((benchmark) => benchmark.id === current)
        ? current
        : activeBenchmark.id
    ));
  }, [activeBenchmark?.id, visibleBenchmarks]);

  useEffect(() => {
    if (!activeBenchmark) return;
    setActiveMetricId((current) => {
      const stillAvailable = activeBenchmark.metrics.some((metric) => metric.id === current);
      return stillAvailable ? current : activeBenchmark.metrics[0].id;
    });
  }, [activeBenchmark?.id, activeBenchmark?.metrics]);

  if (!activeBenchmark || !activeMetric) {
    return (
      <section className="page-grid benchmarks-page">
        <p className="empty-note">No completed LAIA benchmark metrics are available for the current filters.</p>
      </section>
    );
  }

  return (
    <section className="page-grid benchmarks-page">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Benchmark Pages</p>
          <h2>Evidence behind the ranking</h2>
        </div>
        <p>Each tab isolates one consumer-facing capability and the diagnostics that explain outliers.</p>
      </div>
      <BenchmarkReadinessStrip benchmarks={visibleBenchmarks} rows={rows} />
      <div className="benchmark-tabs" aria-label="Benchmark pages">
        {visibleBenchmarks.map((benchmark) => (
          <button
            className={benchmark.id === activeBenchmark.id ? "active" : ""}
            key={benchmark.id}
            type="button"
            onClick={() => setActiveBenchmarkId(benchmark.id)}
          >
            <span>{benchmark.title}</span>
            <small>{benchmark.subtitle}</small>
          </button>
        ))}
      </div>
      <BenchmarkDetailPage
        benchmark={activeBenchmark}
        activeMetric={activeMetric}
        onMetricChange={setActiveMetricId}
        rows={rows}
      />
    </section>
  );
}

function BenchmarkReadinessStrip({
  benchmarks,
  rows,
}: {
  benchmarks: BenchmarkPageConfig[];
  rows: LeaderboardRow[];
}) {
  const scoredModels = rows.filter((row) => numeric(row.model_intelligence_score) !== null).length;
  const coverage = benchmarks.map((benchmark) => ({
    benchmark,
    completeRows: rows.filter((row) => benchmark.metrics.some((metric) => numeric(row[metric.metric]) !== null)).length,
  }));

  return (
    <section className="benchmark-readiness-strip" aria-label="Benchmark readiness summary">
      <article>
        <span>Compared models</span>
        <strong>{scoredModels}</strong>
        <small>Rows with a current LAIA Index.</small>
      </article>
      {coverage.slice(0, 5).map(({ benchmark, completeRows }) => (
        <article key={`readiness-${benchmark.id}`}>
          <span>{benchmark.title}</span>
          <strong>{completeRows}</strong>
          <small>{benchmark.subtitle}</small>
        </article>
      ))}
    </section>
  );
}

function BenchmarkDetailPage({
  benchmark,
  activeMetric,
  onMetricChange,
  rows,
}: {
  benchmark: BenchmarkPageConfig;
  activeMetric: BenchmarkMetric;
  onMetricChange: (id: string) => void;
  rows: LeaderboardRow[];
}) {
  return (
    <section className="benchmark-detail">
      <div className="benchmark-detail-heading">
        <div>
          <p className="eyebrow">{benchmark.badge}</p>
          <h3>{benchmark.title}</h3>
          <p>{benchmark.subtitle} · {benchmark.capability}</p>
        </div>
        <div className="metric-tabs" aria-label={`${benchmark.title} metrics`}>
          {benchmark.metrics.map((metric) => (
            <button
              className={metric.id === activeMetric.id ? "active" : ""}
              type="button"
              key={metric.id}
              onClick={() => onMetricChange(metric.id)}
            >
              {metric.label}
            </button>
          ))}
        </div>
      </div>
      <BenchmarkTopPlot benchmark={benchmark} metric={activeMetric} rows={rows} />
      <div className="benchmark-small-multiples">
        {benchmark.metrics
          .filter((metric) => metric.id !== activeMetric.id)
          .filter((metric) => metricHasData(rows, metric))
          .map((metric) => (
            <BenchmarkMiniPlot benchmark={benchmark} metric={metric} rows={rows} key={`${benchmark.id}-${metric.id}`} />
          ))}
      </div>
      {benchmark.id === "global-mmlu-lite" && <GlobalMMLULanguageSection rows={rows} />}
      {benchmark.id === "rgb" && <RGBLanguageSection rows={rows} />}
    </section>
  );
}

function GlobalMMLULanguageSection({ rows }: { rows: LeaderboardRow[] }) {
  const languageCodes = useMemo(() => {
    const codes = new Set<string>();
    rows.forEach((row) => {
      row.global_mmlu_lite_language_scores?.forEach((score) => {
        if (score.language && score.accuracy !== null) {
          codes.add(score.language);
        }
      });
    });
    return Array.from(codes).sort((a, b) => languageLabel(a).localeCompare(languageLabel(b)));
  }, [rows]);
  const [selectedLanguage, setSelectedLanguage] = useState("all");
  const [rowLimit, setRowLimit] = useState("16");
  const [breakdownMode, setBreakdownMode] = useState<"region" | "language">("region");
  const visibleLimit = rowLimit === "all" ? null : Number(rowLimit);
  const availableRegions = GLOBAL_MMLU_LANGUAGE_REGIONS
    .map((region) => ({
      ...region,
      languages: region.languages.filter((language) => languageCodes.includes(language)),
    }))
    .filter((region) => region.languages.length > 0);
  const shownLanguages =
    selectedLanguage === "all" ? languageCodes : languageCodes.filter((language) => language === selectedLanguage);

  useEffect(() => {
    if (selectedLanguage !== "all" && !languageCodes.includes(selectedLanguage)) {
      setSelectedLanguage("all");
    }
  }, [languageCodes, selectedLanguage]);

  if (!languageCodes.length) {
    return (
      <p className="benchmark-data-note">
        No exported language-level Global MMLU Lite rows are available yet for the current results file.
      </p>
    );
  }

  return (
    <section className="language-section">
      <div className="benchmark-plot-title">
        <div>
          <h4>Language breakdown</h4>
          <p>
            Global MMLU Lite language accuracy. Regional averages are diagnostic geographic groupings, not claims about
            speakers or cultures.
          </p>
        </div>
        <div className="language-controls">
          <label className="language-filter">
            <span>View</span>
            <select
              value={breakdownMode}
              onChange={(event) => setBreakdownMode(event.target.value as "region" | "language")}
            >
              <option value="region">Regions</option>
              <option value="language">Languages</option>
            </select>
          </label>
          <label className="language-filter">
            <span>Language</span>
            <select
              value={selectedLanguage}
              onChange={(event) => setSelectedLanguage(event.target.value)}
              disabled={breakdownMode === "region"}
            >
              <option value="all">All languages</option>
              {languageCodes.map((language) => (
                <option value={language} key={language}>
                  {languageLabel(language)}
                </option>
              ))}
            </select>
          </label>
          <label className="language-filter">
            <span>Rows</span>
            <select value={rowLimit} onChange={(event) => setRowLimit(event.target.value)}>
              <option value="8">Top 8</option>
              <option value="16">Top 16</option>
              <option value="all">All rows</option>
            </select>
          </label>
        </div>
      </div>
      {breakdownMode === "region" ? (
        <div className="language-grid region-grid">
          {availableRegions.map((region) => (
            <GlobalMMLURegionPlot
              rows={rows}
              region={region}
              limit={visibleLimit}
              key={region.id}
            />
          ))}
        </div>
      ) : (
        <div className={`language-grid ${selectedLanguage !== "all" ? "single-language" : ""}`}>
          {shownLanguages.map((language) => (
            <GlobalMMLULanguagePlot rows={rows} language={language} limit={visibleLimit} key={language} />
          ))}
        </div>
      )}
    </section>
  );
}

function GlobalMMLURegionPlot({
  rows,
  region,
  limit,
}: {
  rows: LeaderboardRow[];
  region: { id: string; label: string; languages: string[] };
  limit: number | null;
}) {
  const allItems = rows
    .flatMap((row) => {
      const scores = region.languages
        .map((language) => row.global_mmlu_lite_language_scores?.find((item) => item.language === language) ?? null)
        .filter((score): score is GlobalMMLULanguageScore => score !== null && numeric(score.accuracy) !== null);
      const values = scores.map((score) => numeric(score.accuracy)).filter((value): value is number => value !== null);
      const total = scores.reduce((sum, score) => sum + (numeric(score.total) ?? 0), 0);
      const correct = scores.reduce((sum, score) => sum + (numeric(score.correct) ?? 0), 0);
      const invalid = scores.reduce((sum, score) => sum + (numeric(score.invalid) ?? 0), 0);
      if (!values.length) return [];
      const average = values.reduce((sum, value) => sum + value, 0) / values.length;
      return [{
        row,
        value: average,
        score: {
          language: region.id,
          accuracy: average,
          correct,
          total,
          invalid,
          invalid_rate: total ? invalid / total : null,
        } satisfies GlobalMMLULanguageScore,
        coveredLanguages: scores.length,
      }];
    })
    .sort((a, b) => b.value - a.value);
  const items = limit === null ? allItems : allItems.slice(0, limit);
  const subtitle = region.languages.map(languageLabel).join(", ");
  const title = `${region.label} (${region.languages.length})`;

  return (
    <GlobalMMLUColumnPlot
      code={`${items.length}/${allItems.length}`}
      items={items}
      totalItems={allItems.length}
      max={Math.max(...items.map((item) => item.value), 0.01)}
      subtitle={subtitle}
      title={title}
    />
  );
}

function GlobalMMLULanguagePlot({
  rows,
  language,
  limit,
}: {
  rows: LeaderboardRow[];
  language: string;
  limit: number | null;
}) {
  const allItems = rows
    .map((row) => {
      const score = row.global_mmlu_lite_language_scores?.find((item) => item.language === language);
      return { row, score, value: numeric(score?.accuracy) };
    })
    .filter((item): item is { row: LeaderboardRow; score: GlobalMMLULanguageScore; value: number } => item.value !== null)
    .sort((a, b) => b.value - a.value);
  const items = limit === null ? allItems : allItems.slice(0, limit);

  return (
    <GlobalMMLUColumnPlot
      code={`${language.toUpperCase()} · ${items.length}/${allItems.length}`}
      items={items}
      totalItems={allItems.length}
      max={Math.max(...items.map((item) => item.value), 0.01)}
      title={languageLabel(language)}
    />
  );
}

function GlobalMMLUColumnPlot({
  title,
  code,
  subtitle,
  items,
  totalItems,
  max,
}: {
  title: string;
  code: string;
  subtitle?: string;
  items: Array<{ row: LeaderboardRow; score: LanguageBreakdownScore; value: number }>;
  totalItems: number;
  max: number;
}) {
  return (
    <article className="language-card">
      <div className="language-card-heading">
        <div>
          <h5>{title}</h5>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <span>{code}</span>
      </div>
      <div className="language-column-plot">
        {items.length ? items.map(({ row, score, value }) => (
          <button
            className={`language-column ${rowToneClass(row)}`}
            key={`${title}-${row.variant_id}`}
            type="button"
            title={`${displayModelName(row)} · ${formatPercent(value)} · ${formatLanguageCounts(score)}`}
          >
            <div className="language-column-track">
              <i
                style={{
                  height: `${Math.max(5, (value / max) * 100)}%`,
                  background: providerColor(row),
                }}
              />
              <b>{Math.round(value * 100)}</b>
            </div>
            <LabIcon row={row} />
            <small>{shortModelLabel(row)}</small>
          </button>
        )) : <span className="empty-note">n/a</span>}
      </div>
    </article>
  );
}

function RGBLanguageSection({ rows }: { rows: LeaderboardRow[] }) {
  const languageCodes = useMemo(() => {
    const codes = new Set<string>();
    rows.forEach((row) => {
      row.rgb_language_scores?.forEach((score) => {
        if (score.language && numeric(score.accuracy) !== null) {
          codes.add(score.language);
        }
      });
    });
    return Array.from(codes).sort((a, b) => languageLabel(a).localeCompare(languageLabel(b)));
  }, [rows]);
  const [selectedLanguage, setSelectedLanguage] = useState("all");
  const [rowLimit, setRowLimit] = useState("16");
  const visibleLimit = rowLimit === "all" ? null : Number(rowLimit);
  const shownLanguages =
    selectedLanguage === "all" ? languageCodes : languageCodes.filter((language) => language === selectedLanguage);

  useEffect(() => {
    if (selectedLanguage !== "all" && !languageCodes.includes(selectedLanguage)) {
      setSelectedLanguage("all");
    }
  }, [languageCodes, selectedLanguage]);

  if (!languageCodes.length) {
    return (
      <p className="benchmark-data-note">
        No exported language-level RGB rows are available yet for the current results file.
      </p>
    );
  }

  return (
    <section className="language-section">
      <div className="benchmark-plot-title">
        <div>
          <h4>Language breakdown</h4>
          <p>
            RGB language score across English and Chinese grounding cases. Scores combine noise robustness, rejection,
            information integration, and error detection with the RGB suite weights.
          </p>
        </div>
        <div className="language-controls">
          <label className="language-filter">
            <span>Language</span>
            <select value={selectedLanguage} onChange={(event) => setSelectedLanguage(event.target.value)}>
              <option value="all">All languages</option>
              {languageCodes.map((language) => (
                <option value={language} key={language}>
                  {languageLabel(language)}
                </option>
              ))}
            </select>
          </label>
          <label className="language-filter">
            <span>Rows</span>
            <select value={rowLimit} onChange={(event) => setRowLimit(event.target.value)}>
              <option value="8">Top 8</option>
              <option value="16">Top 16</option>
              <option value="all">All rows</option>
            </select>
          </label>
        </div>
      </div>
      <div className={`language-grid ${selectedLanguage !== "all" ? "single-language" : ""}`}>
        {shownLanguages.map((language) => (
          <RGBLanguagePlot rows={rows} language={language} limit={visibleLimit} key={language} />
        ))}
      </div>
    </section>
  );
}

function RGBLanguagePlot({
  rows,
  language,
  limit,
}: {
  rows: LeaderboardRow[];
  language: string;
  limit: number | null;
}) {
  const allItems = rows
    .map((row) => {
      const score = row.rgb_language_scores?.find((item) => item.language === language);
      return { row, score, value: numeric(score?.accuracy) };
    })
    .filter((item): item is { row: LeaderboardRow; score: RGBLanguageScore; value: number } => item.value !== null)
    .sort((a, b) => b.value - a.value);
  const items = limit === null ? allItems : allItems.slice(0, limit);

  return (
    <GlobalMMLUColumnPlot
      code={`${language.toUpperCase()} · ${items.length}/${allItems.length}`}
      items={items}
      totalItems={allItems.length}
      max={Math.max(...items.map((item) => item.value), 0.01)}
      subtitle={rgbLanguageSubtitle(items)}
      title={languageLabel(language)}
    />
  );
}

function BenchmarkTopPlot({
  benchmark,
  metric,
  rows,
}: {
  benchmark: BenchmarkPageConfig;
  metric: BenchmarkMetric;
  rows: LeaderboardRow[];
}) {
  const items = benchmarkMetricItems(rows, metric).slice(0, 12);
  const max = Math.max(...items.map((item) => item.value), 0.01);
  return (
    <section className="benchmark-top-plot">
      <div className="benchmark-plot-title">
        <div>
          <h4>{metric.label}</h4>
          <p>{benchmark.subtitle} · Top completed rows</p>
        </div>
        <span>{metric.kind === "error" ? "Lower is better" : "Higher is better"}</span>
      </div>
      <div className="benchmark-column-plot">
        {items.length ? items.map(({ row, value }) => (
          <button
            className={`benchmark-column ${rowToneClass(row)}`}
            type="button"
            key={`${benchmark.id}-${metric.id}-${row.variant_id}`}
            style={{ "--provider-color": providerColor(row) } as CSSProperties}
            title={`${displayModelName(row)} · ${formatPercent(value)}`}
          >
            <div className="benchmark-column-track">
              <span
                style={{
                  height: `${Math.max(4, (value / max) * 100)}%`,
                  background: metric.kind === "error" ? "var(--orange)" : providerColor(row),
                }}
              />
              <b>{formatPercent(value)}</b>
            </div>
            <LabIcon row={row} />
            <strong>{shortModelLabel(row)}</strong>
            <small>{quantizationLabel(row)}</small>
          </button>
        )) : <p className="empty-note">No completed rows for this metric yet.</p>}
      </div>
    </section>
  );
}

function BenchmarkMiniPlot({
  benchmark,
  metric,
  rows,
}: {
  benchmark: BenchmarkPageConfig;
  metric: BenchmarkMetric;
  rows: LeaderboardRow[];
}) {
  const items = benchmarkMetricItems(rows, metric).slice(0, 8);
  const max = Math.max(...items.map((item) => item.value), 0.01);
  return (
    <article className="benchmark-mini-card">
      <h4>{metric.label}</h4>
      <p>{benchmark.subtitle}</p>
      <div className="mini-column-plot">
        {items.length ? items.map(({ row, value }) => (
          <span
            className={`mini-column ${rowToneClass(row)}`}
            key={`${benchmark.id}-${metric.id}-mini-${row.variant_id}`}
            title={`${displayModelName(row)} · ${formatPercent(value)} · ${quantizationLabel(row)}`}
            style={{ "--provider-color": providerColor(row) } as CSSProperties}
          >
            <span className="mini-column-track">
              <i
                style={{
                  height: `${Math.max(5, (value / max) * 100)}%`,
                  background: metric.kind === "error" ? "var(--orange)" : providerColor(row),
                }}
              />
              <b>{metric.kind === "error" ? formatPercent(value) : Math.round(value * 100)}</b>
            </span>
            <LabIcon row={row} />
            <small>{shortModelLabel(row)}</small>
            <em>{quantizationLabel(row)}</em>
          </span>
        )) : <span className="empty-note">n/a</span>}
      </div>
    </article>
  );
}

function benchmarkMetricItems(rows: LeaderboardRow[], metric: BenchmarkMetric) {
  return rows
    .map((row) => ({ row, value: numeric(row[metric.metric]) }))
    .filter((item): item is { row: LeaderboardRow; value: number } => item.value !== null)
    .sort((a, b) => metric.kind === "error" ? a.value - b.value : b.value - a.value);
}

function metricHasData(rows: LeaderboardRow[], metric: BenchmarkMetric) {
  return rows.some((row) => numeric(row[metric.metric]) !== null);
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

function ModelsPage({
  rows,
  allRows,
  selectedModelId,
  onSelectedModelIdChange,
}: {
  rows: LeaderboardRow[];
  allRows: LeaderboardRow[];
  selectedModelId: string | null;
  onSelectedModelIdChange: (value: string | null) => void;
}) {
  const tableRows = rows.length ? rows : allRows;
  const rankedRows = [...tableRows].sort((a, b) => scoreForRank(b) - scoreForRank(a));
  const maxModelSize = Math.max(...rankedRows.map(modelSizeGb).filter((size): size is number => size !== null), 0.01);
  const selectedRow = selectedModelId
    ? tableRows.find((row) => row.variant_id === selectedModelId) ?? null
    : null;
  return (
    <section className="page-grid models-page">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Model Catalog</p>
          <h2>Pick from tested 4-bit models</h2>
        </div>
        <p>Consumer rows stay compact first, then expand into sources, coverage, runtime, and raw metric details.</p>
      </div>
      <ModelPageSummary rows={rankedRows} />

      <div className="ranking-list models-ranking-list">
        {rankedRows.map((row, index) => (
          <LeaderboardRowCard
            row={row}
            rank={index + 1}
            maxModelSizeGb={maxModelSize}
            key={`models-rank-${row.normalized_result_id ?? row.variant_id}`}
          />
        ))}
      </div>

      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Model Registry</p>
          <h2>Coverage and source details</h2>
        </div>
        <p>Open the details panel for links, benchmark status, token counts, runtime, and truncation metadata.</p>
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
              <button className="details-button" type="button" onClick={() => onSelectedModelIdChange(row.variant_id)}>
                <Info size={14} aria-hidden="true" />
                Details
              </button>
            </div>
          </article>
        ))}
      </div>

      {selectedRow && <ModelDetailsDrawer row={selectedRow} onClose={() => onSelectedModelIdChange(null)} />}

      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Score Table</p>
          <h2>Latest successful metrics</h2>
        </div>
        <p>Column-level view after merging only the newest saved benchmark metric for each model version.</p>
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
                  <td
                    className={rawCellClass(column as keyof LeaderboardRow, row[column])}
                    key={column}
                    style={rawCellStyle(column as keyof LeaderboardRow, row[column])}
                  >
                    {formatCell(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelPageSummary({ rows }: { rows: LeaderboardRow[] }) {
  const top = rows.find((row) => numeric(row.model_intelligence_score) !== null) ?? null;
  const covered = rows.filter((row) => numeric(row.model_intelligence_coverage) !== null);
  const mostCovered = [...covered].sort((a, b) => (numeric(b.model_intelligence_coverage) ?? 0) - (numeric(a.model_intelligence_coverage) ?? 0))[0] ?? null;
  const smallest = [...rows]
    .filter((row) => modelSizeGb(row) !== null)
    .sort((a, b) => (modelSizeGb(a) ?? 0) - (modelSizeGb(b) ?? 0))[0] ?? null;

  return (
    <section className="consumer-summary-strip" aria-label="Model catalog summary">
      <article>
        <span>Current winner</span>
        <strong>{top ? displayModelName(top) : "n/a"}</strong>
        <small>{top ? formatPoints(numeric(top.model_intelligence_score)) : "No score yet"}</small>
      </article>
      <article>
        <span>Smallest footprint</span>
        <strong>{smallest ? displayModelName(smallest) : "n/a"}</strong>
        <small>{smallest ? formatModelSize(smallest) : "No size metadata"}</small>
      </article>
      <article>
        <span>Most complete row</span>
        <strong>{mostCovered ? displayModelName(mostCovered) : "n/a"}</strong>
        <small>{mostCovered ? coverageLabel(mostCovered) : "No coverage"}</small>
      </article>
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
            <DetailItem label="Merged runs" value={formatCount(numeric(row.merged_run_count))} />
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
        <h2>Readable scores, inspectable evidence.</h2>
        <p>
          LAIA Index is a 100-point text-model score built from five non-judge benchmarks.
          The public comparison keeps other evaluation modes separate so a consumer can compare
          local 4-bit models on one consistent surface.
        </p>
      </div>

      <section className="trust-strip" aria-label="Consumer trust summary">
        <article>
          <span>Scope</span>
          <strong>4-bit local rows</strong>
          <small>Hosted and larger precision rows stay outside the main comparison.</small>
        </article>
        <article>
          <span>Score</span>
          <strong>5 equal text capabilities</strong>
          <small>Knowledge, instructions, tools, coding, and grounding each carry 20 points.</small>
        </article>
        <article>
          <span>Evidence</span>
          <strong>Run signals exposed</strong>
          <small>Sample counts, tokens, latency, truncation, and sources remain visible.</small>
        </article>
      </section>

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
            <p>Vision, factuality, and safety have different evaluation assumptions. They are not part of the public LAIA pages or headline score.</p>
          </article>
          <article>
            <h4>Public scope</h4>
            <p>The public site shows 4-bit local model rows only. Other precision levels and hosted references stay out of the main comparison.</p>
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
  const sortedRows = newestRowsFirst(rows);
  const base = sortedRows[0] ?? bestComparableRun(rows);
  const merged: LeaderboardRow = { ...base };
  const runSignalSources = new Map<string, LeaderboardRow>();

  for (const key of MERGED_BENCHMARK_METRICS) {
    const source = sortedRows.find((row) => numeric(row[key]) !== null);
    if (source) {
      merged[key] = source[key];
      runSignalSources.set(rowIdentityKey(source), source);
    } else {
      merged[key] = null;
    }
  }

  const languageSource = sortedRows.find((row) => row.global_mmlu_lite_language_scores?.length);
  if (languageSource?.global_mmlu_lite_language_scores?.length) {
    merged.global_mmlu_lite_language_scores = languageSource.global_mmlu_lite_language_scores;
    runSignalSources.set(rowIdentityKey(languageSource), languageSource);
  } else {
    merged.global_mmlu_lite_language_scores = null;
  }

  const rgbLanguageSource = sortedRows.find((row) => row.rgb_language_scores?.length);
  if (rgbLanguageSource?.rgb_language_scores?.length) {
    merged.rgb_language_scores = rgbLanguageSource.rgb_language_scores;
    runSignalSources.set(rowIdentityKey(rgbLanguageSource), rgbLanguageSource);
  } else {
    merged.rgb_language_scores = null;
  }

  mergeLatestRunSignals(
    merged,
    runSignalSources.size ? Array.from(runSignalSources.values()) : [base],
  );
  const intelligence = laiaIndexValues(merged);
  merged.model_intelligence_score = intelligence.score;
  merged.model_intelligence_coverage = intelligence.coverage;
  merged.model_intelligence_available_score = intelligence.availableScore;
  return merged;
}

function newestRowsFirst(rows: LeaderboardRow[]) {
  return [...rows].sort((a, b) => {
    const dateDelta = rowTimestamp(b) - rowTimestamp(a);
    if (dateDelta !== 0) return dateDelta;
    return scoreForRank(b) - scoreForRank(a);
  });
}

function rowTimestamp(row: LeaderboardRow) {
  return parseRunDate(row.started_at)?.getTime() ?? 0;
}

function rowIdentityKey(row: LeaderboardRow) {
  return String(row.normalized_result_id ?? row.run_uuid ?? `${row.variant_id}-${row.started_at ?? ""}`);
}

function mergeLatestRunSignals(merged: LeaderboardRow, sourceRows: LeaderboardRow[]) {
  const sources = newestRowsFirst(uniqueRowsByIdentity(sourceRows));
  const latest = sources[0];
  if (latest) {
    merged.started_at = latest.started_at ?? null;
    merged.run_uuid = latest.run_uuid ?? null;
    merged.normalized_result_id = latest.normalized_result_id ?? merged.normalized_result_id;
  }
  merged.merged_run_count = sources.length;

  for (const key of RUN_SIGNAL_SUM_FIELDS) {
    const values = sources.map((row) => numeric(row[key])).filter((value): value is number => value !== null);
    merged[key] = values.length ? values.reduce((sum, value) => sum + value, 0) : null;
  }

  for (const key of RUN_SIGNAL_MAX_FIELDS) {
    const values = sources.map((row) => numeric(row[key])).filter((value): value is number => value !== null);
    merged[key] = values.length ? Math.max(...values) : null;
  }

  const samples = numeric(merged.benchmark_samples) ?? 0;
  const runtime = numeric(merged.benchmark_runtime_seconds) ?? 0;
  const completionTokens = numeric(merged.benchmark_completion_tokens) ?? 0;
  const totalTokens = numeric(merged.benchmark_total_tokens) ?? 0;
  const truncatedCount = numeric(merged.benchmark_truncated_count) ?? 0;
  merged.benchmark_truncated_rate = samples > 0 ? truncatedCount / samples : null;
  merged.benchmark_avg_latency_seconds = samples > 0 && runtime > 0 ? runtime / samples : null;
  merged.benchmark_output_tokens_per_second = runtime > 0 ? completionTokens / runtime : null;
  merged.benchmark_total_tokens_per_second = runtime > 0 ? totalTokens / runtime : null;
}

function uniqueRowsByIdentity(rows: LeaderboardRow[]) {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = rowIdentityKey(row);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
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

function topIndexRows(
  rows: LeaderboardRow[],
  parameterLimit: string,
  gbLimit: string,
) {
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
    .sort((a, b) => scoreForRank(b) - scoreForRank(a));
}

function comparableRowKey(row: LeaderboardRow) {
  return `${quantizationGroupKey(row)}|${quantizationKey(row)}`;
}

function emptyRunAggregate(): RunAggregate {
  return {
    totalTokens: 0,
    promptTokens: 0,
    completionTokens: 0,
    reasoningTokens: 0,
    runtimeSeconds: 0,
    samples: 0,
    truncatedCount: 0,
    p95LatencySeconds: null,
    outputTokensPerSecond: null,
    latestStartedAt: null,
    runCount: 0,
  };
}

function runAggregates(rows: LeaderboardRow[]) {
  const aggregates = new Map<string, RunAggregate>();
  for (const row of rows) {
    const key = comparableRowKey(row);
    const aggregate = aggregates.get(key) ?? emptyRunAggregate();
    aggregate.totalTokens += numeric(row.benchmark_total_tokens) ?? 0;
    aggregate.promptTokens += numeric(row.benchmark_prompt_tokens) ?? 0;
    aggregate.completionTokens += numeric(row.benchmark_completion_tokens) ?? 0;
    aggregate.reasoningTokens += numeric(row.benchmark_reasoning_tokens) ?? 0;
    aggregate.runtimeSeconds += numeric(row.benchmark_runtime_seconds) ?? 0;
    aggregate.samples += numeric(row.benchmark_samples) ?? 0;
    aggregate.truncatedCount += numeric(row.benchmark_truncated_count) ?? 0;

    const p95 = numeric(row.benchmark_p95_latency_seconds);
    if (p95 !== null) {
      aggregate.p95LatencySeconds = aggregate.p95LatencySeconds === null ? p95 : Math.max(aggregate.p95LatencySeconds, p95);
    }

    const outputTokensPerSecond = numeric(row.benchmark_output_tokens_per_second);
    if (outputTokensPerSecond !== null) {
      aggregate.outputTokensPerSecond = aggregate.outputTokensPerSecond === null
        ? outputTokensPerSecond
        : Math.max(aggregate.outputTokensPerSecond, outputTokensPerSecond);
    }

    const currentDate = parseRunDate(aggregate.latestStartedAt);
    const nextDate = parseRunDate(row.started_at);
    if (nextDate && (!currentDate || nextDate.getTime() > currentDate.getTime())) {
      aggregate.latestStartedAt = row.started_at ?? null;
    }

    aggregate.runCount += 1;
    aggregates.set(key, aggregate);
  }
  return aggregates;
}

function runStatsForRow(row: LeaderboardRow, aggregates: Map<string, RunAggregate>) {
  const aggregate = aggregates.get(comparableRowKey(row));
  if (aggregate && aggregate.runCount > 0) return aggregate;
  return {
    ...emptyRunAggregate(),
    totalTokens: numeric(row.benchmark_total_tokens) ?? 0,
    promptTokens: numeric(row.benchmark_prompt_tokens) ?? 0,
    completionTokens: numeric(row.benchmark_completion_tokens) ?? 0,
    reasoningTokens: numeric(row.benchmark_reasoning_tokens) ?? 0,
    runtimeSeconds: numeric(row.benchmark_runtime_seconds) ?? 0,
    samples: numeric(row.benchmark_samples) ?? 0,
    truncatedCount: numeric(row.benchmark_truncated_count) ?? 0,
    p95LatencySeconds: numeric(row.benchmark_p95_latency_seconds),
    outputTokensPerSecond: numeric(row.benchmark_output_tokens_per_second),
    latestStartedAt: row.started_at ?? null,
    runCount: 1,
  };
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

function isSmolLM2Row(row: LeaderboardRow) {
  const source = `${row.variant_name} ${row.base_model_name} ${apiModel(row) ?? ""} ${row.model_repo ?? ""}`.toLowerCase();
  return source.includes("smollm2") || source.includes("smol lm2") || source.includes("smol lm 2");
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

function formatCompactNumber(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`;
  return Math.round(value).toLocaleString();
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

function rawCellClass(column: keyof LeaderboardRow, value: unknown) {
  const numericValue = numeric(value);
  if (!RAW_SCORE_COLUMNS.has(column) && !RAW_ERROR_COLUMNS.has(column)) return undefined;
  return numericValue === null ? "score-heat-cell missing" : "score-heat-cell";
}

function rawCellStyle(column: keyof LeaderboardRow, value: unknown): CSSProperties | undefined {
  const numericValue = numeric(value);
  if (numericValue === null) return undefined;
  if (RAW_SCORE_COLUMNS.has(column)) return heatmapStyle(numericValue);
  if (RAW_ERROR_COLUMNS.has(column)) return heatmapStyle(numericValue, true);
  return undefined;
}

function heatmapStyle(value: number, invert = false): CSSProperties {
  const clamped = Math.max(0, Math.min(1, value));
  const quality = invert ? 1 - clamped : clamped;
  const hue = 10 + quality * 95;
  const background = `hsl(${hue}, 92%, 88%)`;
  const border = `hsl(${hue}, 70%, 62%)`;
  return {
    background,
    boxShadow: `inset 0 0 0 1px ${border}`,
  };
}

function humanizeColumn(value: string) {
  return value.replace(/_/g, " ");
}

function formatPercent(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function languageLabel(language: string) {
  return GLOBAL_MMLU_LANGUAGE_LABELS[language] ?? language.toUpperCase();
}

function formatLanguageCounts(score: LanguageBreakdownScore) {
  if ((score.correct === null || score.correct === undefined) && score.total) {
    return `${Math.round(score.total)} samples`;
  }
  if (score.correct === null || score.correct === undefined || !score.total) return "sample count n/a";
  const invalid = score.invalid ? ` · ${score.invalid} invalid` : "";
  return `${score.correct}/${score.total}${invalid}`;
}

function rgbLanguageSubtitle(items: Array<{ score: RGBLanguageScore }>) {
  const components = items[0]?.score.components;
  if (!components) return "RGB language suite";
  return Object.entries(components)
    .filter(([, value]) => numeric(value) !== null)
    .map(([key, value]) => `${RGB_COMPONENT_LABELS[key] ?? humanizeColumn(key)} ${Math.round((numeric(value) ?? 0) * 100)}%`)
    .join(" · ");
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

function formatShortDate(value: Date, includeYear = false) {
  return value.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(includeYear ? { year: "numeric" as const } : {}),
  });
}

function parseRunDate(value?: string | null) {
  if (!value) return null;
  const normalized = value.includes(" ") && !value.includes("T") ? value.replace(" ", "T") : value;
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
}

function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

function slugForAsset(value: string) {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
