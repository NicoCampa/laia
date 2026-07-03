import {
  Bot,
  BookOpen,
  Braces,
  Code2,
  Cpu,
  Database,
  FileText,
  Flag,
  Info,
  Leaf,
  ExternalLink,
  Lightbulb,
  LightbulbOff,
  Search,
  Shield,
  WifiOff,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { WORLD_COUNTRY_PATH } from "./worldMapPaths";

const WORLD_COUNTRY_PATH_WITHOUT_ANTARCTICA = WORLD_COUNTRY_PATH
  .replace(/M1000,485\.3[\s\S]*?L1000,485\.3Z/, "")
  .replace(/M0,485\.3[\s\S]*?L0,485\.3Z/, "");

const LAIA_LOGO_SRC = "/laia_primary_logo_transparent.png";

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
  model_release_date?: string | null;
  model_release_source?: string | null;
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
  benchmark_output_cap_hit_count?: number | null;
  benchmark_output_cap_hit_samples?: number | null;
  benchmark_output_cap_hit_rate?: number | null;
  benchmark_output_cap_breakdown?: OutputCapBreakdown[] | string | null;
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

type OutputCapBreakdown = {
  benchmark: string;
  max_output_tokens: number;
  hits: number;
  samples: number;
  rate?: number | null;
};

type Payload = {
  generated_at: string;
  tagline: string;
  output_cap_policy?: Record<string, number>;
  leaderboard: LeaderboardRow[];
};

type Page = "leaderboard" | "benchmarks" | "models" | "methodology" | "mission";

type Filters = {
  query: string;
  family: string;
  parameterSize: string;
  memoryFootprint: string;
  sortBy: string;
};

type SelectOption = {
  value: string;
  label: string;
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

type WeightedMetric = {
  metric: keyof LeaderboardRow;
  label: string;
  weight: number;
};

type UseCaseLeaderboard = {
  id: string;
  label: string;
  summary: string;
  minCoverage: number;
  icon: ReactNode;
  weights: WeightedMetric[];
};

const PAGE_LABELS: Record<Page, string> = {
  leaderboard: "Leaderboard",
  benchmarks: "Benchmarks",
  models: "Models",
  methodology: "Methodology",
  mission: "Mission",
};

const PAGE_HEADLINES: Record<Page, string> = {
  leaderboard: "Local model intelligence, measured on your machine.",
  benchmarks: "Benchmarks",
  models: "A practical catalog of 4-bit local models and OpenAI references.",
  methodology: "A score built for transparent comparison.",
  mission: "Why local models matter.",
};

const PAGE_COPY: Record<Page, string> = {
  leaderboard: "Rank 4-bit local models and OpenAI references by the text-only LAIA Index.",
  benchmarks: "Knowledge, instruction following, tool use, coding, and grounding are split into comparable slices.",
  models: "Ranked rows include source links, footprint, benchmark coverage, run metadata, and raw exported metrics.",
  methodology: "The public index keeps judge, safety, and vision results separate from the core local text comparison.",
  mission: "The project exists to make local model capability easier to see, compare, and trust on consumer and edge hardware.",
};

const PAGE_SIGNALS: Record<Page, string[]> = {
  leaderboard: ["4-bit rows + OpenAI refs", "Merged benchmark runs", "Text-only LAIA Index"],
  benchmarks: [],
  models: ["Source and backend metadata", "Coverage status per benchmark", "Raw metric table"],
  methodology: ["100-point formula", "No external judge in the score", "4-bit public scope"],
  mission: ["Consumer hardware", "Edge deployment", "Private by default"],
};

const emptyFilters: Filters = {
  query: "",
  family: "all",
  parameterSize: "all",
  memoryFootprint: "all",
  sortBy: "laia",
};

const LEADERBOARD_CHAPTERS = [
  { id: "leaderboard-use-cases", label: "Use cases" },
  { id: "leaderboard-landscape", label: "Pareto" },
  { id: "leaderboard-release", label: "Release" },
  { id: "leaderboard-top-benchmarks", label: "Top 5 by category" },
  { id: "leaderboard-origins", label: "Origins" },
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

type MethodologyChapter = {
  id: string;
  label: string;
  title: string;
  intro: string;
};

type MethodologyBenchmark = {
  id: string;
  title: string;
  subtitle: string;
  measures: string;
  metric: string;
  scope: string;
  evaluator: string;
  inclusion: string;
  caveat: string;
};

type MethodologyNamedItem = {
  label: string;
  detail: string;
};

type MethodologyDefinition = {
  term: string;
  description: string;
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

const BENCHMARK_METRIC_INFO: Record<string, string> = {
  global_mmlu_lite_pass_at_1: "Accuracy on the exported Global MMLU Lite questions across the multilingual knowledge set.",
  global_mmlu_lite_invalid_rate: "Share of Global MMLU Lite questions where the output did not yield a valid answer.",
  ifbench_prompt_level_loose: "IFBench prompt-level pass rate with loose matching to the requested format and constraints.",
  ifbench_prompt_level_strict: "IFBench prompt-level pass rate with strict matching to the requested format and constraints.",
  ifbench_instruction_level_loose: "IFBench instruction-level pass rate with loose matching across the prompt's instructions.",
  ifbench_instruction_level_strict: "IFBench instruction-level pass rate with strict matching across the prompt's instructions.",
  bfcl_v4_selected_accuracy: "Accuracy on the BFCL v4 selected subset used in the text-only comparison.",
  bfcl_v4_non_live_accuracy: "Accuracy on BFCL v4 non-live tool calls.",
  bfcl_v4_live_accuracy: "Accuracy on BFCL v4 live tool calls.",
  bfcl_v4_multi_turn_accuracy: "Accuracy on BFCL v4 multi-turn tool-use tasks.",
  bfcl_v4_agentic_accuracy: "Accuracy on BFCL v4 agentic tool-use tasks.",
  bfcl_v4_invalid_rate: "Share of BFCL v4 outputs that did not produce a valid tool call.",
  mbpp_pass_at_1: "Pass@1 on MBPP: first-attempt code that passes the benchmark tests.",
  mbpp_compile_rate: "Share of MBPP generations that compile successfully.",
  mbpp_runtime_error_rate: "Share of MBPP generations that fail with a runtime error.",
  mbpp_invalid_rate: "Share of MBPP generations that produce an invalid benchmark output.",
  rgb_all_rate: "Overall RGB grounding score across the exported evidence-based tasks.",
  rgb_rejection_rate: "RGB negative-rejection rate: rejecting unsupported or distractor evidence.",
  rgb_fact_check_rate: "RGB fact-check rate: verifying claims correctly against retrieved evidence.",
  rgb_error_correction_rate: "RGB error-correction rate: fixing false statements using the retrieved evidence.",
  ocrbench_v2_micro_score: "Micro average across the exported OCRBench v2 tasks.",
  ocrbench_v2_en_score: "OCRBench v2 score on English OCR tasks.",
  ocrbench_v2_cn_score: "OCRBench v2 score on Chinese OCR tasks.",
  mmmu_accuracy: "Overall MMMU accuracy across the exported multimodal questions.",
  mmmu_multiple_choice_accuracy: "MMMU accuracy on multiple-choice questions only.",
  mmmu_open_accuracy: "MMMU accuracy on open-ended questions only.",
  simpleqa_f1: "SimpleQA answer overlap score using token-level F1.",
  simpleqa_correct_rate: "Share of SimpleQA answers judged correct.",
  simpleqa_hallucination_rate: "Share of SimpleQA answers judged hallucinated.",
  simpleqa_not_attempted_rate: "Share of SimpleQA questions left unanswered or declined.",
  harmbench_refusal_rate: "Share of HarmBench harmful prompts the model refuses.",
  harmbench_attack_success_rate: "Share of HarmBench harmful prompts that bypass refusal.",
};

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

const MODEL_SORT_OPTIONS: SelectOption[] = [
  { value: "laia", label: "LAIA Index" },
  ...TEXT_CAPABILITIES.map((capability) => ({
    value: capability.id,
    label: capability.label,
  })),
  { value: "parameter-size", label: "Parameter size" },
  { value: "memory-footprint", label: "Memory footprint" },
];

const RGB_COMPONENT_LABELS: Record<string, string> = {
  noise_robustness: "Noise",
  negative_rejection: "Rejection",
  information_integration: "Integration",
  error_detection: "Error detection",
};

const CAPABILITIES = TEXT_CAPABILITIES;

const OUTPUT_CAP_ORDER = ["global_mmlu_lite", "ifbench", "bfcl_v4", "mbpp", "rgb"];

const LAIA_INDEX_WEIGHTS = {
  global_mmlu_lite_pass_at_1: 0.2,
  ifbench_prompt_level_loose: 0.2,
  bfcl_v4_selected_accuracy: 0.2,
  mbpp_pass_at_1: 0.2,
  rgb_all_rate: 0.2,
} satisfies Partial<Record<keyof LeaderboardRow, number>>;

const USE_CASE_LEADERBOARDS: UseCaseLeaderboard[] = [
  {
    id: "assistant",
    label: "Local Assistant",
    summary: "Everyday assistant quality with instruction following, knowledge, grounding, tools, and coding.",
    minCoverage: 0.5,
    icon: <Bot size={16} />,
    weights: [
      { metric: "ifbench_prompt_level_loose", label: "IFBench", weight: 0.30 },
      { metric: "global_mmlu_lite_pass_at_1", label: "Global MMLU", weight: 0.225 },
      { metric: "rgb_all_rate", label: "RGB", weight: 0.225 },
      { metric: "bfcl_v4_selected_accuracy", label: "BFCL", weight: 0.15 },
      { metric: "mbpp_pass_at_1", label: "MBPP", weight: 0.10 },
    ],
  },
  {
    id: "coding-assistant",
    label: "Coding Assistant",
    summary: "Short Python program synthesis with enough instruction and tool reliability to be useful.",
    minCoverage: 0.5,
    icon: <Code2 size={16} />,
    weights: [
      { metric: "mbpp_pass_at_1", label: "MBPP", weight: 0.60 },
      { metric: "ifbench_prompt_level_loose", label: "IFBench", weight: 0.15 },
      { metric: "bfcl_v4_selected_accuracy", label: "BFCL", weight: 0.15 },
      { metric: "global_mmlu_lite_pass_at_1", label: "Global MMLU", weight: 0.075 },
      { metric: "rgb_all_rate", label: "RGB", weight: 0.025 },
    ],
  },
  {
    id: "tool-agent",
    label: "Tool Agent",
    summary: "Function selection and argument generation, supported by instruction and grounding signals.",
    minCoverage: 0.5,
    icon: <Braces size={16} />,
    weights: [
      { metric: "bfcl_v4_selected_accuracy", label: "BFCL", weight: 0.55 },
      { metric: "ifbench_prompt_level_loose", label: "IFBench", weight: 0.20 },
      { metric: "rgb_all_rate", label: "RGB", weight: 0.15 },
      { metric: "mbpp_pass_at_1", label: "MBPP", weight: 0.075 },
      { metric: "global_mmlu_lite_pass_at_1", label: "Global MMLU", weight: 0.025 },
    ],
  },
  {
    id: "rag-assistant",
    label: "RAG Assistant",
    summary: "Evidence use, noise rejection, and grounded answers with supporting knowledge and instructions.",
    minCoverage: 0.5,
    icon: <Database size={16} />,
    weights: [
      { metric: "rgb_all_rate", label: "RGB", weight: 0.50 },
      { metric: "global_mmlu_lite_pass_at_1", label: "Global MMLU", weight: 0.25 },
      { metric: "ifbench_prompt_level_loose", label: "IFBench", weight: 0.15 },
      { metric: "bfcl_v4_selected_accuracy", label: "BFCL", weight: 0.075 },
      { metric: "mbpp_pass_at_1", label: "MBPP", weight: 0.025 },
    ],
  },
  {
    id: "knowledge-slm",
    label: "Knowledge SLM",
    summary: "Text knowledge breadth with smaller supporting weights for following and grounding.",
    minCoverage: 0.5,
    icon: <BookOpen size={16} />,
    weights: [
      { metric: "global_mmlu_lite_pass_at_1", label: "Global MMLU", weight: 0.70 },
      { metric: "ifbench_prompt_level_loose", label: "IFBench", weight: 0.15 },
      { metric: "rgb_all_rate", label: "RGB", weight: 0.10 },
      { metric: "bfcl_v4_selected_accuracy", label: "BFCL", weight: 0.025 },
      { metric: "mbpp_pass_at_1", label: "MBPP", weight: 0.025 },
    ],
  },
  {
    id: "classifier-proxy",
    label: "Classifier Proxy",
    summary: "A proxy for labeling and routing tasks until a dedicated classification suite is added.",
    minCoverage: 0.55,
    icon: <Flag size={16} />,
    weights: [
      { metric: "global_mmlu_lite_pass_at_1", label: "Global MMLU", weight: 0.40 },
      { metric: "ifbench_prompt_level_loose", label: "IFBench", weight: 0.30 },
      { metric: "rgb_fact_check_rate", label: "RGB fact-check", weight: 0.15 },
      { metric: "rgb_rejection_rate", label: "RGB rejection", weight: 0.15 },
    ],
  },
];

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
  "benchmark_output_cap_hit_count",
  "benchmark_output_cap_hit_samples",
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

const METHODOLOGY_CHAPTERS: MethodologyChapter[] = [
  {
    id: "method-overview",
    label: "Overview",
    title: "Overview",
    intro: "What the public site measures, what LAIA is for, and why separate benchmark families stay outside the headline score.",
  },
  {
    id: "method-public-scope",
    label: "Scope",
    title: "Public scope",
    intro: "The main public comparison is intentionally narrower than the full benchmark suite.",
  },
  {
    id: "method-rows",
    label: "Rows",
    title: "How rows are built",
    intro: "Comparable runs are merged into one public row before the LAIA columns are computed.",
  },
  {
    id: "method-laia-index",
    label: "LAIA Index",
    title: "LAIA Index",
    intro: "The headline score is stored as normalized data and rendered as points, alongside coverage and available-score fields.",
  },
  {
    id: "method-index-benchmarks",
    label: "Index benchmarks",
    title: "Benchmarks in the index",
    intro: "These five text-only, non-judge benchmark families feed the public LAIA score.",
  },
  {
    id: "method-separate-benchmarks",
    label: "Separate benchmarks",
    title: "Benchmarks reported separately",
    intro: "Vision, factuality, and safety stay visible in the project, but they do not change the headline public ranking.",
  },
  {
    id: "method-reproducibility",
    label: "Reproducibility",
    title: "Reproducibility",
    intro: "Every row is treated as an auditable local measurement with pinned revisions, deterministic defaults, and recorded run metadata.",
  },
  {
    id: "method-outputs",
    label: "Outputs",
    title: "Outputs and data model",
    intro: "The website is backed by exported normalized rows, but the raw artifacts and schema stay available under results/.",
  },
  {
    id: "method-limitations",
    label: "Limitations",
    title: "Limitations and comparison rules",
    intro: "Benchmark rows are only comparable when their benchmark-specific settings and evaluation assumptions match.",
  },
  {
    id: "method-definitions",
    label: "Definitions",
    title: "Definitions",
    intro: "Project-specific terms used throughout the site and export pipeline.",
  },
];

const METHODOLOGY_PUBLIC_SCOPE = {
  main: [
    "Publishable 4-bit local rows.",
    "OpenAI reference rows shown on the same score surface for context.",
    "The five text-only, non-judge benchmark families used by LAIA.",
  ],
  separate: [
    "OCRBench v2 and MMMU as separate vision and multimodal diagnostics.",
    "SimpleQA and HarmBench as separate judge-based factuality and safety diagnostics.",
    "Supporting benchmark metrics that explain a row without changing the headline LAIA ranking.",
  ],
  excluded: [
    "Non-4-bit local precision variants from the public main ranking.",
    "Smoke rows, synthetic rows, and other non-publishable export rows.",
    "Judge-dependent and multimodal metrics from the text-only LAIA score itself.",
  ],
};

const METHODOLOGY_ROW_BUILD_STEPS = [
  "Start from publishable normalized rows. Synthetic rows, smoke rows, and non-public comparison rows are filtered out before the public site is rendered.",
  "Group comparable rows by displayed model name, provider, parameter size, and quantization label so equivalent reruns can be merged instead of ranked separately.",
  "For each benchmark metric family, take the newest comparable run that actually contains that metric. Global MMLU Lite and RGB language breakdown arrays follow the newest run that exported them.",
  "Merge run signals across the contributing runs. Count-like values, runtime, tokens, costs, and cap-hit counts are summed; latency-style fields keep the maximum exposed value; derived rates are recalculated on the merged row.",
  "Carry forward the newest run identity fields (`started_at`, `run_uuid`, and `normalized_result_id`) and record how many rows contributed through `merged_run_count`.",
  "Compute `model_intelligence_score`, `model_intelligence_coverage`, and `model_intelligence_available_score` on the merged row that the website exports.",
];

const METHODOLOGY_LAIA_FIELDS: MethodologyDefinition[] = [
  {
    term: "model_intelligence_score",
    description: "Full-suite weighted text score with missing benchmark families counted as zero. The database stores a normalized 0-1 value; the site renders it as points out of 100.",
  },
  {
    term: "model_intelligence_coverage",
    description: "Total benchmark-family weight actually present in that row. Coverage tells you how much of the text suite was run before ranking the row by its full score.",
  },
  {
    term: "model_intelligence_available_score",
    description: "Weighted average over only the benchmark families present in the row. It removes the zero-for-missing-family penalty and is also rendered as points.",
  },
];

const METHODOLOGY_PUBLISHED_WEIGHTS: MethodologyNamedItem[] = [
  { label: "Global MMLU Lite", detail: "20%" },
  { label: "IFBench", detail: "20%" },
  { label: "BFCL v4", detail: "20%" },
  { label: "MBPP", detail: "20%" },
  { label: "RGB", detail: "20%" },
];

const METHODOLOGY_INDEX_BENCHMARKS: MethodologyBenchmark[] = [
  {
    id: "global-mmlu-lite",
    title: "Global MMLU Lite",
    subtitle: "Knowledge",
    measures: "Multilingual academic and factual breadth through multiple-choice question answering across the supported language configs.",
    metric: "global_mmlu_lite_pass_at_1",
    scope: "Full runs use the supported language configs on the test split. Smoke mode uses 5 English questions.",
    evaluator: "The runner extracts a single answer letter and reports generation pass@1. When configured, thinking blocks are stripped before parsing.",
    inclusion: "Included in LAIA because it measures broad text knowledge without an external judge.",
    caveat: "Compare it with other generation pass@1 Global MMLU Lite rows, not with log-likelihood MMLU variants.",
  },
  {
    id: "ifbench",
    title: "IFBench",
    subtitle: "Instructions",
    measures: "Precise instruction following with verifiable formatting and constraint checks.",
    metric: "ifbench_prompt_level_loose",
    scope: "The default run uses the full 300-prompt `allenai/IFBench_test` set. Smoke mode evaluates the first 5 prompts.",
    evaluator: "The project uses the official AllenAI verification functions. The leaderboard-facing metric is prompt-level loose accuracy.",
    inclusion: "Included in LAIA because it is a text-only, non-judge measure of instruction-following reliability.",
    caveat: "Strict and instruction-level metrics are reported separately, but publishable comparisons should use prompt-level loose accuracy unless a stricter target is explicitly intended.",
  },
  {
    id: "bfcl-v4",
    title: "BFCL v4",
    subtitle: "Tools",
    measures: "Function selection and argument generation in prompt-mode tool calling.",
    metric: "bfcl_v4_selected_accuracy",
    scope: "The default shortcut category is `single_turn`. Shortcut-generated full runs use 1,000 deterministic stratified samples across the resolved BFCL categories.",
    evaluator: "Local AI Analysis uses Berkeley's prompt-mode BFCL scoring path and checks whether the emitted function name and arguments match the expected call. The default score does not execute arbitrary tools.",
    inclusion: "Included in LAIA because it supplies the tool-use benchmark family in the text-only, non-judge suite.",
    caveat: "Single-turn, non-live, live, multi-turn, agentic, sampled, and all-scoring BFCL runs are not interchangeable. Category set, sample limit, strategy, and seed must match.",
  },
  {
    id: "mbpp",
    title: "MBPP",
    subtitle: "Coding",
    measures: "Short Python program synthesis from natural-language task descriptions.",
    metric: "mbpp_pass_at_1",
    scope: "The default full run uses the standard MBPP test split. `sanitized` and challenge-test variants are separate configurations.",
    evaluator: "Generated Python is executed locally against the dataset assertions in an isolated-mode Python subprocess with a per-sample timeout.",
    inclusion: "Included in LAIA because it provides executable coding evidence without an external judge.",
    caveat: "Rows are only comparable when the MBPP config (`full` vs `sanitized`) and challenge-test setting match.",
  },
  {
    id: "rgb",
    title: "RGB",
    subtitle: "Grounding",
    measures: "Retrieval-grounded answering under noise, rejection, information integration, and factual-error detection.",
    metric: "rgb_all_rate",
    scope: "The default `suite` covers curated English and Chinese slices. Shortcut-generated full suite runs use 100 seeded random rows per slice, for 800 RGB calls total.",
    evaluator: "The runner builds the document prompt and applies RGB's local lexical scoring logic from `evalue.py`. No external judge is used for the public RGB score.",
    inclusion: "Included in LAIA because it measures evidence use and noise rejection inside the text-only, non-judge suite.",
    caveat: "Suite rows and single-dataset RGB rows should not be mixed unless dataset, noise rate, passage settings, and sample strategy match.",
  },
];

const METHODOLOGY_SEPARATE_BENCHMARKS: MethodologyBenchmark[] = [
  {
    id: "ocrbench-v2",
    title: "OCRBench v2",
    subtitle: "Vision",
    measures: "Bilingual OCR, document parsing, visual text understanding, reasoning, and Chinese OCR tasks for multimodal models.",
    metric: "ocrbench_v2_score",
    scope: "Default full runs use the English and Chinese aggregate configs, but cap them to a deterministic 1,000-example stratified subset instead of all 10,000 examples.",
    evaluator: "The local evaluator mirrors the official OCRBench v2 grouping and uses recorded evaluator version `ocrbench_v2_local_vqa_anls_iou_v1` for the local VQA, ANLS, counting, formula, and IoU checks.",
    inclusion: "Reported separately because it is a vision benchmark for multimodal models rather than a text-only LAIA component.",
    caveat: "Only compare rows that use the same evaluator version, config set, sample limit, sample strategy, and sample seed.",
  },
  {
    id: "mmmu",
    title: "MMMU",
    subtitle: "Vision",
    measures: "College-level multimodal reasoning across 30 subjects and six domains.",
    metric: "mmmu_accuracy",
    scope: "Default local runs use the validation split across all 30 subjects. Smoke mode always uses 5 Accounting samples.",
    evaluator: "The local evaluator follows the official MMMU response parsing and exact-match logic for multiple-choice and open questions, recorded as `mmmu_official_parse_local_v1`.",
    inclusion: "Reported separately because it is a multimodal benchmark and not part of the text-only LAIA ranking.",
    caveat: "MMMU rows are only comparable when the split and subject list match.",
  },
  {
    id: "simpleqa",
    title: "SimpleQA",
    subtitle: "Judge benchmark",
    measures: "Short-form factual accuracy, incorrect attempted answers, and not-attempted behavior.",
    metric: "simpleqa_f1",
    scope: "Shortcut-generated full runs use 500 deterministic stratified questions by topic and answer type.",
    evaluator: "By default, SimpleQA uses an LLM judge; a deterministic heuristic fallback exists for debugging, but it is not the official scoring method.",
    inclusion: "Reported separately because it requires a judge and is intentionally excluded from `model_intelligence_score`.",
    caveat: "Judge choice matters. Publishable comparisons should use a stronger pinned judge and record whether grading was LLM-based or heuristic.",
  },
  {
    id: "harmbench",
    title: "HarmBench",
    subtitle: "Judge benchmark",
    measures: "Refusal and harmful-completion behavior on the HarmBench text behavior set.",
    metric: "harmbench_refusal_rate",
    scope: "Default runs use the text behavior set with `standard,contextual` functional categories and skip the copyright slice.",
    evaluator: "The benchmark grades outputs with a local judge model. A deterministic heuristic exists for debugging, but it is not the official HarmBench classifier.",
    inclusion: "Reported separately because it requires a judge and measures safety/refusal behavior rather than general capability.",
    caveat: "Rows are only comparable when the selected functional categories and judge model match.",
  },
];

const METHODOLOGY_SUITE_ALIASES: MethodologyNamedItem[] = [
  { label: "text", detail: "Text-only, non-judge benchmarks: Global MMLU Lite, IFBench, BFCL v4, MBPP, and RGB." },
  { label: "vision", detail: "Multimodal, non-judge benchmarks: OCRBench v2 and MMMU." },
  { label: "judge", detail: "Judge-based benchmarks: SimpleQA and HarmBench." },
  { label: "suite", detail: "`text` plus `vision`, without judge-based benchmarks." },
  { label: "full", detail: "Every benchmark family, including judge-based benchmarks." },
];

const METHODOLOGY_DEFAULT_CAPS: MethodologyNamedItem[] = [
  { label: "BFCL v4", detail: "1,000 deterministic stratified prompt-mode samples across the resolved categories." },
  { label: "RGB", detail: "100 seeded random rows per curated suite slice, 800 RGB calls total." },
  { label: "OCRBench v2", detail: "1,000 deterministic stratified examples across the English and Chinese aggregate configs." },
  { label: "SimpleQA", detail: "500 deterministic stratified questions by topic and answer type." },
];

const METHODOLOGY_PINNED_REVISIONS: MethodologyNamedItem[] = [
  { label: "Global MMLU Lite", detail: "cbf2f73663ff201d4d56e891c8c2c18467aeea06" },
  { label: "IFBench", detail: "2e8a48de45ff3bf41242f927254ca81b59ca3ae2" },
  { label: "OCRBench v2", detail: "458b55b5f62bfd6eba7b5080da34fbc9a68c2626" },
  { label: "MMMU", detail: "4619a102cf5ad2da1abf7e220fde1258d2434cb7" },
  { label: "MBPP", detail: "4bb6404fdc6cacfda99d4ac4205087b89d32030c" },
  { label: "RGB", detail: "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615" },
  { label: "SimpleQA reference", detail: "652c89d0ca9df547706735883097e9537d40dc47" },
  { label: "HarmBench", detail: "8e1604d1171fe8a48d8febecd22f600e462bdcdd" },
];

const METHODOLOGY_RECORDED_METADATA: MethodologyNamedItem[] = [
  { label: "Provider and backend", detail: "Provider label, native API base URL, backend profile, hardware profile, and model id or tag." },
  { label: "Benchmark settings", detail: "Dataset name, pinned revision, split, languages, sample limits, strategy, seed, and benchmark-specific config such as BFCL categories or RGB dataset/noise settings." },
  { label: "Prompting and decoding", detail: "Prompt template, parser version, temperature, top-p, max tokens, seed, reasoning effort, and requested context length." },
  { label: "Per-sample logs", detail: "Raw prompt, raw output, extracted answer, gold answer, correctness, runtime, and API usage when the backend exposes it." },
  { label: "Run signals", detail: "Sample counts, token totals, latency, cap hits, throughput, and cost-like fields when they are available." },
];

const METHODOLOGY_OUTPUTS: MethodologyNamedItem[] = [
  { label: "DuckDB database", detail: "results/local_ai_analysis.duckdb" },
  { label: "Raw run events", detail: "results/raw_results.jsonl" },
  { label: "Per-benchmark artifacts", detail: "results/<benchmark>/ with sample JSONL and summary files for each benchmark family." },
  { label: "Generated configs", detail: "results/generated_configs/" },
];

const METHODOLOGY_AUDIT_FIELDS: string[] = [
  "provider label and backend name",
  "model id, variant name, quantization, and source link",
  "run UUID, latest started-at time, and merged run count",
  "benchmark sample counts, runtime, and output-cap hits",
  "prompt, completion, reasoning, and total token totals",
  "average and tail latency plus output throughput when available",
  "vision, judge, and supporting metrics that stay outside the LAIA headline score",
];

const METHODOLOGY_LIMITATIONS: MethodologyNamedItem[] = [
  { label: "Missing families are penalized", detail: "`model_intelligence_score` counts missing text benchmark families as zero, so check `model_intelligence_coverage` before treating rows as fully comparable." },
  { label: "Public rows are 4-bit and reasoning-off", detail: "The public local comparison is built from 4-bit rows with reasoning disabled. Early local results did not show a large performance drop versus full-precision variants, but cross-precision rows are still kept out of the public main ranking." },
  { label: "Global MMLU Lite is generation pass@1", detail: "Do not compare it directly with log-likelihood MMLU numbers from other leaderboards." },
  { label: "BFCL scope must match", detail: "Category set, sample limit, strategy, and seed all affect the BFCL result surface." },
  { label: "RGB scope must match", detail: "Suite rows and single-dataset RGB rows are not equivalent unless dataset, noise rate, and passage settings match." },
  { label: "Vision is separate", detail: "OCRBench v2 and MMMU are reported separately because they measure multimodal capability, not text-only LAIA performance." },
  { label: "Judge benchmarks are separate", detail: "SimpleQA and HarmBench depend on judge behavior, so they stay outside `model_intelligence_score`." },
  { label: "MBPP runs local code", detail: "Generated Python is executed locally in an isolated-mode subprocess with a timeout, but it is not a hardened security sandbox." },
];

const METHODOLOGY_DEFINITIONS: MethodologyDefinition[] = [
  {
    term: "Benchmark row",
    description: "A leaderboard-facing normalized result record that contains the benchmark metrics and metadata for one public row.",
  },
  {
    term: "Merged run",
    description: "A public row built by merging compatible benchmark metrics from multiple comparable runs of the same model/quantization surface.",
  },
  {
    term: "Provider / backend",
    description: "Provider is the lab or reference label shown on the site; backend is the native serving stack such as Ollama, LM Studio, or oMLX that produced the run.",
  },
  {
    term: "Judge benchmark",
    description: "A benchmark whose final label depends on another model or explicit judge logic, such as SimpleQA or HarmBench.",
  },
  {
    term: "OpenAI reference",
    description: "A closed-source API row shown on the same public surface for context, but not a 4-bit local model row.",
  },
  {
    term: "Coverage",
    description: "`model_intelligence_coverage`, the total benchmark-family weight present in the row.",
  },
  {
    term: "Available score",
    description: "`model_intelligence_available_score`, the weighted average over only the benchmark families present in the row.",
  },
];

const MISSION_PILLARS = [
  {
    icon: <Cpu size={18} />,
    title: "Consumer hardware first",
    copy: "The mission is to increase awareness of how much capability now fits on laptops, desktops, phones, and edge devices instead of only large cloud clusters.",
  },
  {
    icon: <Bot size={18} />,
    title: "Built for real-world edge use",
    copy: "Small and tiny language models are especially important where inference must live close to the machine: robotics, embedded systems, field devices, assistants, and other constrained environments.",
  },
  {
    icon: <Shield size={18} />,
    title: "Privacy and sovereignty",
    copy: "Local execution keeps sensitive prompts, documents, and actions on the device or inside the organization that owns the hardware, which matters for privacy, control, and sovereign deployment.",
  },
  {
    icon: <WifiOff size={18} />,
    title: "Offline resilience",
    copy: "When models run locally, useful systems can keep working with poor connectivity, limited bandwidth, or no internet access at all.",
  },
  {
    icon: <Leaf size={18} />,
    title: "Leaner deployment",
    copy: "Efficient local models can reduce infrastructure overhead and unnecessary remote inference. In many practical settings that can also mean lower operating cost and lower emissions than shipping everything to a large remote service.",
  },
  {
    icon: <BookOpen size={18} />,
    title: "Awareness through measurement",
    copy: "The project tries to replace vague impressions with auditable results, so people can see what local models can already do, where they still fail, and which tradeoffs are actually worth making.",
  },
];

const MISSION_FOCUS_AREAS = [
  "Make small and tiny model performance visible on one public comparison surface.",
  "Show when local rows are already competitive with familiar closed-source API models that otherwise come with usage cost.",
  "Show that practical local AI is not limited to hobby demos or isolated benchmarks.",
  "Help developers choose models that can run on everyday hardware and edge devices.",
  "Support robotics and embodied systems where latency, reliability, and on-device control matter.",
  "Keep the conversation focused on measurable capability instead of marketing scale alone.",
];

const MISSION_OUTCOMES = [
  "Total privacy for prompts, context, and outputs when deployment requires it.",
  "Greater deployment sovereignty for teams that do not want core inference tied to a remote provider.",
  "Offline and low-connectivity operation in field, industrial, education, and mobile settings.",
  "Lower friction to build assistants, tools, and controllers that live next to the application instead of behind an API hop.",
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
  const publicRows = useMemo(() => comparableRows.filter(isPublicLeaderboardRow), [comparableRows]);
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
      <SiteHeader
        page={page}
        onNavigate={setPage}
      />

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
          filters={filters}
          options={options}
          onFiltersChange={setFilters}
          onClearFilters={() => setFilters(emptyFilters)}
        />
      )}
      {page === "methodology" && <MethodologyPage />}
      {page === "mission" && <MissionPage />}

      <SiteFooter />
    </main>
  );
}

function SiteFooter() {
  const links = {
    repo: "https://github.com/NicoCampa/laia",
    issues: "https://github.com/NicoCampa/laia/issues",
    owner: "https://github.com/NicoCampa",
    readme: "https://github.com/NicoCampa/laia#readme",
    license: "https://github.com/NicoCampa/laia/blob/main/LICENSE",
  };
  return (
    <footer className="footer">
      <div className="footer-brand">
        <span className="footer-eyebrow">Local AI Analysis</span>
        <strong>Independent local-model benchmarking, with closed-source references for context.</strong>
        <p className="footer-note">Questions, corrections, and benchmark requests should go through the project repository.</p>
      </div>
      <div className="footer-columns">
        <section className="footer-column" aria-labelledby="footer-contact-title">
          <h3 id="footer-contact-title">Contact</h3>
          <a href={links.issues} target="_blank" rel="noreferrer">
            Open an issue
            <ExternalLink size={14} />
          </a>
          <a href={links.owner} target="_blank" rel="noreferrer">
            GitHub profile
            <ExternalLink size={14} />
          </a>
        </section>
        <section className="footer-column" aria-labelledby="footer-project-title">
          <h3 id="footer-project-title">Project</h3>
          <a href={links.repo} target="_blank" rel="noreferrer">
            Repository
            <ExternalLink size={14} />
          </a>
          <a href={links.readme} target="_blank" rel="noreferrer">
            README
            <ExternalLink size={14} />
          </a>
          <a href={links.license} target="_blank" rel="noreferrer">
            Apache-2.0 license
            <ExternalLink size={14} />
          </a>
        </section>
      </div>
    </footer>
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

function LaiaLogo({ className }: { className: string }) {
  return <img className={className} src={LAIA_LOGO_SRC} alt="" aria-hidden="true" />;
}

function PlotBrandStamp({ className = "" }: { className?: string }) {
  return <LaiaLogo className={`plot-brand-stamp${className ? ` ${className}` : ""}`} />;
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
      <div className="site-header-left">
        <button className="site-mark" type="button" onClick={() => onNavigate("leaderboard")}>
          <LaiaLogo className="site-mark-logo" />
          <strong>Local AI Analysis</strong>
        </button>
      </div>
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
      <div className="site-header-right">
        <button
          className={`site-mission-button ${page === "mission" ? "nav-active" : ""}`}
          type="button"
          onClick={() => onNavigate("mission")}
        >
          <span className="site-mission-icon" aria-hidden="true">
            <Flag size={13} />
          </span>
          <span>Mission</span>
        </button>
      </div>
    </header>
  );
}

function PageHero({ page }: { page: Page }) {
  const signals = PAGE_SIGNALS[page];
  return (
    <section className={`hero-band page-hero page-${page}-hero`}>
      <div>
        {page !== "benchmarks" && <p className="eyebrow">Local AI Analysis</p>}
        <h1>{PAGE_HEADLINES[page]}</h1>
        <p>{PAGE_COPY[page]}</p>
      </div>
      {signals.length > 0 && (
        <div className="page-hero-signals" aria-label={`${PAGE_LABELS[page]} summary`}>
          {signals.map((signal) => (
            <span key={`${page}-${signal}`}>{signal}</span>
          ))}
        </div>
      )}
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
        label="Lab"
        value={filters.family}
        options={options.families}
        allLabel="All labs"
        onChange={(family) => onChange({ ...filters, family })}
      />
      <Select
        label="Parameter Size"
        value={filters.parameterSize}
        options={options.parameterSizes}
        allLabel="All parameter sizes"
        onChange={(parameterSize) => onChange({ ...filters, parameterSize })}
      />
      <Select
        label="Memory Footprint"
        value={filters.memoryFootprint}
        options={options.memoryFootprints}
        allLabel="All memory"
        onChange={(memoryFootprint) => onChange({ ...filters, memoryFootprint })}
      />
      <Select
        label="Order By"
        value={filters.sortBy}
        options={options.sortOptions}
        allLabel="Default order"
        includeAllOption={false}
        onChange={(sortBy) => onChange({ ...filters, sortBy })}
      />
    </section>
  );
}

function Select({
  label,
  value,
  options,
  allLabel,
  includeAllOption = true,
  onChange,
}: {
  label: string;
  value: string;
  options: SelectOption[];
  allLabel: string;
  includeAllOption?: boolean;
  onChange: (value: string) => void;
}) {
  const upToLabel = label === "Parameter Size" || label === "Memory Footprint";
  return (
    <label className="field">
      <span>{upToLabel ? `${label} (Up To)` : label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {includeAllOption ? <option value="all">{allLabel}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
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
  const chartRows = useMemo(() => topIndexRows(rows), [rows]);

  return (
    <>
      <section className="leaderboard-landing">
        <div className="leaderboard-landing-copy">
          <h1>Independently benchmarked SML and TML</h1>
          <p>SML/TML means small and tiny language models, compared with the same text-only LAIA score.</p>
        </div>
        <IndexPlotCard
          title="LAIA score"
          subtitle="Higher is better"
          rows={chartRows}
          onOpenModel={onOpenModel}
        />
      </section>

      <section className="leaderboard-shell">
        <ChapterNav chapters={LEADERBOARD_CHAPTERS} />
        <div className="page-grid leaderboard-view">
          <UseCaseLeaderboardsSection rows={rows} onOpenModel={onOpenModel} />
          <LandscapeSection rows={rows} />
          <ReleaseDateSection rows={rows} />
          <TopBenchmarkSection rows={rows} onOpenModel={onOpenModel} />
          <ModelOriginsSection rows={originRows} />
        </div>
      </section>
    </>
  );
}

function ChapterNav({
  chapters,
  title = "Chapters",
  ariaLabel = "Section chapters",
}: {
  chapters: { id: string; label: string }[];
  title?: string;
  ariaLabel?: string;
}) {
  const [activeId, setActiveId] = useState<string>(chapters[0]?.id ?? "");

  useEffect(() => {
    if (!chapters.length) return;

    const updateActiveChapter = () => {
      const offset = 132;
      let currentId = chapters[0]?.id ?? "";

      for (const chapter of chapters) {
        const section = document.getElementById(chapter.id);
        if (!section) continue;
        const { top } = section.getBoundingClientRect();
        if (top - offset <= 0) {
          currentId = chapter.id;
        } else {
          break;
        }
      }

      setActiveId(currentId);
    };

    updateActiveChapter();
    window.addEventListener("scroll", updateActiveChapter, { passive: true });
    window.addEventListener("resize", updateActiveChapter);
    return () => {
      window.removeEventListener("scroll", updateActiveChapter);
      window.removeEventListener("resize", updateActiveChapter);
    };
  }, [chapters]);

  return (
    <nav className="chapter-nav" aria-label={ariaLabel}>
      <strong>{title}</strong>
      {chapters.map((chapter) => (
        <a
          className={activeId === chapter.id ? "active" : ""}
          href={`#${chapter.id}`}
          key={chapter.id}
          aria-current={activeId === chapter.id ? "true" : undefined}
        >
          {chapter.label}
        </a>
      ))}
    </nav>
  );
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
  height: 660,
  mapX: 160,
  mapY: 103,
  mapW: 1000,
  mapH: 500,
  mapVisibleH: 455,
};
const ORIGIN_MAP_DISPLAY_SCALE = 0.72;

const LAB_ORIGIN_LOCATIONS: LabOriginLocation[] = [
  { id: "ai2", label: "AI2", city: "Seattle", country: "United States", lat: 47.6062, lon: -122.3321, labelX: 136, labelY: 110, side: "left" },
  { id: "microsoft", label: "Microsoft", city: "Redmond", country: "United States", lat: 47.674, lon: -122.1215, labelX: 136, labelY: 178, side: "left" },
  { id: "nvidia", label: "NVIDIA", city: "Santa Clara", country: "United States", lat: 37.3541, lon: -121.9552, labelX: 136, labelY: 246, side: "left" },
  { id: "meta", label: "Meta", city: "Menlo Park", country: "United States", lat: 37.453, lon: -122.1817, labelX: 136, labelY: 314, side: "left" },
  { id: "google", label: "Google", city: "Mountain View", country: "United States", lat: 37.3861, lon: -122.0839, labelX: 136, labelY: 382, side: "left" },
  { id: "liquid", label: "Liquid AI", city: "Cambridge, MA", country: "United States", lat: 42.3736, lon: -71.1097, labelX: 136, labelY: 458, side: "left" },
  { id: "ibm", label: "IBM", city: "Armonk", country: "United States", lat: 41.1265, lon: -73.714, labelX: 136, labelY: 526, side: "left" },
  { id: "huggingface", label: "Hugging Face", city: "New York City", country: "United States", lat: 40.7128, lon: -74.006, labelX: 136, labelY: 594, side: "left" },
  { id: "mistral", label: "Mistral AI", city: "Paris", country: "France", lat: 48.8566, lon: 2.3522, labelX: 1188, labelY: 212, side: "right" },
  { id: "tii", label: "TII", city: "Abu Dhabi", country: "United Arab Emirates", lat: 24.4539, lon: 54.3773, labelX: 1188, labelY: 346, side: "right" },
  { id: "alibaba", label: "Alibaba", city: "Hangzhou", country: "China", lat: 30.2741, lon: 120.1551, labelX: 1188, labelY: 474, side: "right" },
];

function ModelOriginsSection({ rows }: { rows: LeaderboardRow[] }) {
  const [activeId, setActiveId] = useState<string | null>(null);
  const viewportWidth = useViewportWidth();
  const markers = useMemo(() => modelOriginMarkers(rows), [rows]);
  if (!markers.length) return null;
  const mapScale = Math.min(
    ORIGIN_MAP_DISPLAY_SCALE,
    Math.max(0.26, (viewportWidth - 44) / ORIGIN_MAP.width),
  );

  return (
    <section className="chapter-section model-origin-section" id="leaderboard-origins">
      <div className="section-heading compact">
        <div>
          <h2>Model origins</h2>
        </div>
        <p>Lab headquarters for the compared models.</p>
      </div>

      <div className="origin-map-scroll">
        <div
          className="origin-map-stage-shell"
          style={{
            width: `${ORIGIN_MAP.width * mapScale}px`,
            height: `${ORIGIN_MAP.height * mapScale}px`,
          }}
        >
          <div
            className="origin-map-stage"
            style={{
              transform: `scale(${mapScale})`,
              transformOrigin: "top left",
            }}
          >
          <svg
            className="origin-world-map"
            viewBox={`0 0 ${ORIGIN_MAP.width} ${ORIGIN_MAP.height}`}
            role="img"
            aria-label="World map showing model lab headquarters cities as dots"
          >
            <defs>
              <clipPath id="origin-map-land-clip">
                <rect x="0" y="0" width={ORIGIN_MAP.mapW} height={ORIGIN_MAP.mapVisibleH} />
              </clipPath>
            </defs>
            <rect className="origin-map-background" x={ORIGIN_MAP.mapX} y={ORIGIN_MAP.mapY} width={ORIGIN_MAP.mapW} height={ORIGIN_MAP.mapVisibleH} rx="22" />
            <g className="origin-countries" transform={`translate(${ORIGIN_MAP.mapX} ${ORIGIN_MAP.mapY})`} clipPath="url(#origin-map-land-clip)">
              <path d={WORLD_COUNTRY_PATH_WITHOUT_ANTARCTICA} />
            </g>

            {markers.map((marker) => {
              const point = originPoint(marker.location);
              const callout = originCalloutPoint(marker.location);
              const active = activeId === marker.location.id;
              return (
                <g className={`origin-connection${active ? " active" : ""}`} key={`origin-line-${marker.location.id}`}>
                  <path className="origin-leader" d={`M${callout.x},${callout.y} L${point.x},${point.y}`} />
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

          <div className="origin-map-overlay" aria-hidden="true">
            {markers.map((marker) => {
              const box = originLabelBox(marker.location);
              const active = activeId === marker.location.id;
              return (
                <div
                  className="origin-map-overlay-item"
                  key={`origin-card-${marker.location.id}`}
                  style={{ left: `${box.x}px`, top: `${box.y}px`, width: `${box.width}px` }}
                >
                  <div
                    className={`pareto-row origin-map-card${active ? " active" : ""}`}
                    style={{ "--provider-color": originAccentColor(marker.location.id) } as CSSProperties}
                    onMouseEnter={() => setActiveId(marker.location.id)}
                    onMouseLeave={() => setActiveId(null)}
                  >
                    <span className="lab-icon" aria-hidden="true">
                      <img src={originLogoSrc(marker.location.id)} alt="" />
                    </span>
                    <span>
                      <b>{marker.location.label}</b>
                      <small>{marker.location.city}</small>
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          </div>
        </div>
      </div>
      <div className="origin-mobile-list" aria-label="Model lab headquarters">
        {markers.map((marker) => (
          <div
            className="pareto-row origin-map-card"
            key={`origin-mobile-card-${marker.location.id}`}
            style={{ "--provider-color": originAccentColor(marker.location.id) } as CSSProperties}
          >
            <span className="lab-icon" aria-hidden="true">
              <img src={originLogoSrc(marker.location.id)} alt="" />
            </span>
            <span>
              <b>{marker.location.label}</b>
              <small>{marker.location.city}</small>
            </span>
          </div>
        ))}
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
        models: new Set(["Nemotron 3 Nano 4B"]),
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
  const box = originLabelBox(location);
  return {
    x: location.side === "left" ? box.x + box.width : box.x,
    y: box.y + box.height / 2,
  };
}

function originLabelBox(location: LabOriginLocation) {
  const width = 194;
  const height = 62;
  const x = location.side === "left" ? 20 : ORIGIN_MAP.width - width - 20;
  const y = location.labelY - height / 2;
  return {
    x,
    y,
    width,
    height,
  };
}

function useViewportWidth() {
  const [width, setWidth] = useState(() => (typeof window === "undefined" ? 1280 : window.innerWidth));

  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return width;
}

function originLogoSrc(id: string) {
  const logos: Record<string, string> = {
    ai2: "/labs/ai2.png",
    alibaba: "/labs/qwen.png",
    google: "/labs/google.png",
    huggingface: "/labs/huggingface.svg",
    ibm: "/labs/ibm.png",
    liquid: "/labs/liquidAI.jpeg",
    meta: "/labs/meta.png",
    microsoft: "/labs/microsoft.png",
    mistral: "/labs/mistral.png",
    nvidia: "/labs/nvidia.webp",
    openai: "/labs/openai.png",
    tii: "/labs/TechnologyInnovationINstitute.PNG",
  };
  return logos[id] ?? "/labs/ai2.png";
}

function originAccentColor(id: string) {
  const providers: Record<string, string> = {
    ai2: "AI2",
    alibaba: "Alibaba",
    google: "Google",
    huggingface: "Hugging Face",
    ibm: "IBM",
    liquid: "Liquid AI",
    meta: "Meta",
    microsoft: "Microsoft",
    mistral: "Mistral AI",
    nvidia: "NVIDIA",
    openai: "OpenAI",
    tii: "TII",
  };
  return providerColorName(providers[id] ?? id);
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
    openai: "GPT reference",
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
  const [xAxis, setXAxis] = useState<"footprint" | "parameters">("footprint");
  const localRows = useMemo(() => rows.filter((row) => !isHostedOpenAIRow(row)), [rows]);
  const sizePoints = localRows
    .map((row) => ({ row, x: paretoXAxisValue(row, "footprint"), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  const parameterPoints = localRows
    .map((row) => ({ row, x: paretoXAxisValue(row, "parameters"), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  const hasSizeView = sizePoints.length >= 2;
  const hasParameterView = parameterPoints.length >= 2;
  if (!hasSizeView && !hasParameterView) return null;

  const effectiveXAxis = xAxis === "parameters" && hasParameterView ? "parameters" : "footprint";

  return (
    <section className="landscape-section chapter-section" id="leaderboard-landscape">
      <div className="section-heading compact">
        <div>
          <h2>Pareto frontier</h2>
          <p>No smaller, higher-scoring row exists.</p>
        </div>
        <div className="section-heading-actions" aria-label="Pareto axis">
          <button
            className={`plot-toggle${effectiveXAxis === "footprint" ? " active" : ""}`}
            type="button"
            onClick={() => setXAxis("footprint")}
          >
            By memory footprint
          </button>
          <button
            className={`plot-toggle${effectiveXAxis === "parameters" ? " active" : ""}`}
            type="button"
            onClick={() => setXAxis("parameters")}
            disabled={!hasParameterView}
          >
            By parameter size
          </button>
        </div>
      </div>
      <div className="landscape-grid">
        <SizeIntelligencePlot rows={localRows} xAxis={effectiveXAxis} />
      </div>
    </section>
  );
}

function SizeIntelligencePlot({
  rows,
  xAxis,
}: {
  rows: LeaderboardRow[];
  xAxis: "footprint" | "parameters";
}) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const points = rows
    .map((row) => ({ row, x: paretoXAxisValue(row, xAxis), y: numeric(row.model_intelligence_score) }))
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point.x !== null && point.y !== null);
  const width = 720;
  const height = 460;
  const pad = { top: 28, right: 28, bottom: 42, left: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxX = Math.max(1, Math.ceil(Math.max(...points.map((p) => p.x)) * 1.08));
  const maxY = 1;
  const xFor = (x: number) => pad.left + (x / maxX) * plotW;
  const yFor = (y: number) => pad.top + plotH - (y / maxY) * plotH;
  const frontier = efficiencyFrontierRows(rows, xAxis);
  const frontierIds = new Set(frontier.map((item) => item.row.variant_id));
  const paretoList = frontier.slice().sort((a, b) => a.size - b.size);
  const frontierPolyline = frontier
    .slice()
    .sort((a, b) => a.size - b.size)
    .map((point) => `${xFor(point.size)},${yFor(point.score)}`)
    .join(" ");
  const hoveredPoint = points.find((point) => point.row.variant_id === hoveredId) ?? null;
  const hoveredCallout = hoveredPoint
    ? {
        left: (xFor(hoveredPoint.x) / width) * 100,
        top: (yFor(hoveredPoint.y) / height) * 100,
        anchor: hoveredPoint.x > maxX * 0.58 ? "-100%" : "0px",
        offset: hoveredPoint.x > maxX * 0.58 ? { x: -8, y: -10 } : { x: 8, y: -10 },
      }
    : null;

  return (
    <article className="landscape-panel">
      <div className="landscape-scatter">
        <PlotBrandStamp className="floating" />
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={xAxis === "parameters" ? "LAIA Index versus parameter size in billions" : "LAIA Index versus model size in GB"}
        >
          {[0, maxX / 2, maxX].map((tick) => (
            <g key={`landscape-x-${tick}`}>
              <line className="grid-line" x1={xFor(tick)} x2={xFor(tick)} y1={pad.top} y2={pad.top + plotH} />
              <text className="axis-label" x={xFor(tick)} y={height - 16} textAnchor="middle">
                {formatParetoTick(tick, xAxis)}
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
          {frontierPolyline && <polyline className="frontier-line" points={frontierPolyline} />}
          {points.map((point) => {
            const isFrontier = frontierIds.has(point.row.variant_id);
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
                {isFrontier && (
                  <circle
                    className="scatter-point-halo"
                    cx={xFor(point.x)}
                    cy={yFor(point.y)}
                    r={10}
                  />
                )}
                <circle
                  className={`scatter-point quant-${quantizationTone(point.row)}${isFrontier ? " frontier" : ""}`}
                  cx={xFor(point.x)}
                  cy={yFor(point.y)}
                  r={isFrontier ? 6 : 4.5}
                  style={{ fill: providerColor(point.row) }}
                >
                  <title>{displayModelName(point.row)} · {formatPoints(point.y)} · {formatParetoXAxisValue(point.row, xAxis)}</title>
                </circle>
              </g>
            );
          })}
        </svg>
        <div className="frontier-callouts" aria-hidden="true">
          {hoveredPoint && hoveredCallout && (
            <div
              className="frontier-callout hover-callout"
              style={{
                left: `${hoveredCallout.left}%`,
                top: `${hoveredCallout.top}%`,
                "--callout-anchor-x": hoveredCallout.anchor,
                "--callout-x": `${hoveredCallout.offset.x}px`,
                "--callout-y": `${hoveredCallout.offset.y}px`,
                "--provider-color": providerColor(hoveredPoint.row),
              } as CSSProperties}
            >
              <LabIcon row={hoveredPoint.row} />
              <span>{shortModelLabel(hoveredPoint.row)}</span>
            </div>
          )}
        </div>
      </div>
      <aside className="pareto-list" aria-label="Pareto frontier models">
        <h4>Pareto models</h4>
        <div>
          {paretoList.map((point) => (
            <button
              className={`pareto-row ${rowToneClass(point.row)}`}
              key={`pareto-row-${point.row.variant_id}`}
              type="button"
              onMouseEnter={() => setHoveredId(point.row.variant_id)}
              onMouseLeave={() => setHoveredId(null)}
              onFocus={() => setHoveredId(point.row.variant_id)}
              onBlur={() => setHoveredId(null)}
              style={{ "--provider-color": providerColor(point.row) } as CSSProperties}
            >
              <LabIcon row={point.row} />
              <span>
                <b>{shortModelLabel(point.row)}</b>
                <small>{formatPoints(point.score)} · {formatParetoXAxisValue(point.row, xAxis)}</small>
              </span>
            </button>
          ))}
        </div>
      </aside>
    </article>
  );
}

function efficiencyFrontierRows(rows: LeaderboardRow[], xAxis: "footprint" | "parameters" = "footprint") {
  const candidates = rows
    .map((row) => {
      const size = paretoXAxisValue(row, xAxis);
      const score = numeric(row.model_intelligence_score);
      return size !== null && score !== null ? { row, size, score } : null;
    })
    .filter((item): item is { row: LeaderboardRow; size: number; score: number } => item !== null);
  return candidates
    .filter((candidate) => !candidates.some((other) =>
      other.row.variant_id !== candidate.row.variant_id
      && other.size <= candidate.size
      && other.score >= candidate.score
      && (other.size < candidate.size || other.score > candidate.score)
    ))
    .sort((a, b) => a.size - b.size || b.score - a.score);
}

function paretoXAxisValue(row: LeaderboardRow, xAxis: "footprint" | "parameters") {
  return xAxis === "parameters" ? numeric(row.parameter_size_b) : modelSizeGb(row);
}

function formatParetoXAxisValue(row: LeaderboardRow, xAxis: "footprint" | "parameters") {
  if (xAxis === "parameters") {
    const value = numeric(row.parameter_size_b);
    return value === null ? "n/a" : `${formatBillionSize(value)}B params`;
  }
  return formatModelSize(row);
}

function formatParetoTick(value: number, xAxis: "footprint" | "parameters") {
  if (value === 0) return "0";
  if (xAxis === "parameters") return `${formatBillionSize(value)}B`;
  return `${value.toFixed(value < 10 ? 1 : 0)} GB`;
}

function ReleaseDateSection({ rows }: { rows: LeaderboardRow[] }) {
  const points = releaseDatePoints(rows);
  if (points.length < 2) return null;

  return (
    <section className="landscape-section chapter-section" id="leaderboard-release">
      <div className="section-heading compact">
        <div>
          <h2>Score by release date</h2>
          <p>Each dot uses the model release date recorded in the public model registry.</p>
        </div>
      </div>
      <div className="landscape-grid">
        <ReleaseDatePlot rows={rows} />
      </div>
    </section>
  );
}

function TopBenchmarkSection({
  rows,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const groups = useMemo(() => topRowsByCapability(rows), [rows]);
  const visibleGroups = groups.filter((group) => group.rows.length);
  if (!visibleGroups.length) return null;

  return (
    <section className="chapter-section benchmark-top-section" id="leaderboard-top-benchmarks">
      <div className="section-heading compact">
        <div>
          <h2>Top 5 by category</h2>
        </div>
        <p>Highest-scoring models in each LAIA category.</p>
      </div>
      <div className="benchmark-top-grid">
        {visibleGroups.map(({ capability, rows: topRows }) => (
          <article className="benchmark-top-card" key={`top-capability-${capability.id}`}>
            <div className="benchmark-top-card-head">
              <span className="metric-icon">{capability.icon}</span>
              <div>
                <h3>{capability.label}</h3>
                <p>{capability.benchmark} · {capability.metricLabel}</p>
              </div>
            </div>
            <div className="benchmark-top-list">
              {topRows.map(({ row, value }, index) => (
                <button
                  className={`benchmark-top-row ${rowToneClass(row)}`}
                  key={`top-${capability.id}-${row.variant_id}`}
                  type="button"
                  style={{ "--provider-color": providerColor(row) } as CSSProperties}
                  onClick={() => onOpenModel(row)}
                >
                  <span className="benchmark-top-rank">{index + 1}</span>
                  <LabIcon row={row} />
                  <span>
                    <b>{shortModelLabel(row)}</b>
                    <small>{formatPercent(value)}</small>
                  </span>
                </button>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function UseCaseLeaderboardsSection({
  rows,
  onOpenModel,
}: {
  rows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const groups = useMemo(
    () => USE_CASE_LEADERBOARDS.map((leaderboard) => ({
      leaderboard,
      rows: topRowsByUseCase(rows, leaderboard),
    })).filter((group) => group.rows.length),
    [rows],
  );
  if (!groups.length) return null;

  return (
    <section className="chapter-section use-case-section" id="leaderboard-use-cases">
      <div className="section-heading compact">
        <div>
          <h2>Use-case leaderboards</h2>
        </div>
        <p>Local SLM rankings by benchmark outcomes only. Runtime, latency, throughput, and hardware cost are excluded.</p>
      </div>
      <div className="use-case-grid">
        {groups.map(({ leaderboard, rows: topRows }) => (
          <article className="use-case-card" key={`use-case-${leaderboard.id}`}>
            <div className="benchmark-top-card-head">
              <span className="metric-icon">{leaderboard.icon}</span>
              <div>
                <h3>{leaderboard.label}</h3>
                <p>{leaderboard.summary}</p>
              </div>
            </div>
            <div className="use-case-weight-list" aria-label={`${leaderboard.label} weights`}>
              {leaderboard.weights.map((weight) => (
                <span key={`${leaderboard.id}-${weight.metric}`}>
                  <b>{formatWeightPercent(weight.weight)}</b>
                  {weight.label}
                </span>
              ))}
            </div>
            <div className="benchmark-top-list">
              {topRows.map(({ row, score, coverage }, index) => (
                <button
                  className={`benchmark-top-row use-case-row ${rowToneClass(row)}`}
                  key={`use-case-${leaderboard.id}-${row.variant_id}`}
                  type="button"
                  style={{ "--provider-color": providerColor(row) } as CSSProperties}
                  onClick={() => onOpenModel(row)}
                >
                  <span className="benchmark-top-rank">{index + 1}</span>
                  <LabIcon row={row} />
                  <span>
                    <b>{shortModelLabel(row)}</b>
                    <small>{formatPoints(score)}</small>
                  </span>
                </button>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function ReleaseDatePlot({ rows }: { rows: LeaderboardRow[] }) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const points = releaseDatePoints(rows);
  const width = 720;
  const height = 460;
  const pad = { top: 28, right: 28, bottom: 42, left: 58 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const minTime = Math.min(...points.map((point) => point.x));
  const maxTime = Math.max(...points.map((point) => point.x));
  const span = Math.max(1, maxTime - minTime);
  const xMin = minTime - span * 0.04;
  const xMax = maxTime + span * 0.04;
  const maxY = 1;
  const xFor = (x: number) => pad.left + ((x - xMin) / (xMax - xMin)) * plotW;
  const yFor = (y: number) => pad.top + plotH - (y / maxY) * plotH;
  const latestRows = points
    .slice()
    .sort((a, b) => b.x - a.x || b.y - a.y)
    .slice(0, 8);
  const hoveredPoint = points.find((point) => point.row.variant_id === hoveredId) ?? null;
  const hoveredCallout = hoveredPoint
    ? {
        left: (xFor(hoveredPoint.x) / width) * 100,
        top: (yFor(hoveredPoint.y) / height) * 100,
        anchor: hoveredPoint.x > xMin + (xMax - xMin) * 0.58 ? "-100%" : "0px",
        offset: hoveredPoint.x > xMin + (xMax - xMin) * 0.58 ? { x: -8, y: -10 } : { x: 8, y: -10 },
      }
    : null;
  const xTicks = dateTicks(xMin, xMax);

  return (
    <article className="landscape-panel">
      <div className="landscape-scatter">
        <PlotBrandStamp className="floating" />
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="LAIA Index versus model release date">
          {xTicks.map((tick) => (
            <g key={`release-x-${tick}`}>
              <line className="grid-line" x1={xFor(tick)} x2={xFor(tick)} y1={pad.top} y2={pad.top + plotH} />
              <text className="axis-label" x={xFor(tick)} y={height - 16} textAnchor="middle">
                {formatYearTick(tick)}
              </text>
            </g>
          ))}
          {[0, maxY / 2, maxY].map((tick) => (
            <g key={`release-y-${tick}`}>
              <line className="grid-line" x1={pad.left} x2={pad.left + plotW} y1={yFor(tick)} y2={yFor(tick)} />
              <text className="axis-label" x={pad.left - 9} y={yFor(tick) + 4} textAnchor="end">
                {formatPoints(tick)}
              </text>
            </g>
          ))}
          {points.map((point) => (
            <g
              key={`release-${point.row.variant_id}`}
              className={`scatter-node${hoveredId === point.row.variant_id ? " active" : ""}`}
              tabIndex={0}
              onMouseEnter={() => setHoveredId(point.row.variant_id)}
              onMouseLeave={() => setHoveredId(null)}
              onFocus={() => setHoveredId(point.row.variant_id)}
              onBlur={() => setHoveredId(null)}
            >
              <circle className="scatter-hit-area" cx={xFor(point.x)} cy={yFor(point.y)} r={16} />
              <circle
                className="scatter-point"
                cx={xFor(point.x)}
                cy={yFor(point.y)}
                r={5}
                style={{ fill: providerColor(point.row) }}
              >
                <title>{displayModelName(point.row)} · {formatPoints(point.y)} · {formatReleaseDate(point.row)}</title>
              </circle>
            </g>
          ))}
        </svg>
        <div className="frontier-callouts" aria-hidden="true">
          {hoveredPoint && hoveredCallout && (
            <div
              className="frontier-callout hover-callout"
              style={{
                left: `${hoveredCallout.left}%`,
                top: `${hoveredCallout.top}%`,
                "--callout-anchor-x": hoveredCallout.anchor,
                "--callout-x": `${hoveredCallout.offset.x}px`,
                "--callout-y": `${hoveredCallout.offset.y}px`,
                "--provider-color": providerColor(hoveredPoint.row),
              } as CSSProperties}
            >
              <LabIcon row={hoveredPoint.row} />
              <span>{shortModelLabel(hoveredPoint.row)}</span>
            </div>
          )}
        </div>
      </div>
      <aside className="pareto-list release-list" aria-label="Latest model releases">
        <h4>Latest releases</h4>
        <div>
          {latestRows.map((point) => (
            <button
              className={`pareto-row ${rowToneClass(point.row)}`}
              key={`release-row-${point.row.variant_id}`}
              type="button"
              onMouseEnter={() => setHoveredId(point.row.variant_id)}
              onMouseLeave={() => setHoveredId(null)}
              onFocus={() => setHoveredId(point.row.variant_id)}
              onBlur={() => setHoveredId(null)}
              style={{ "--provider-color": providerColor(point.row) } as CSSProperties}
            >
              <LabIcon row={point.row} />
              <span>
                <b>{shortModelLabel(point.row)}</b>
                <small>{formatPoints(point.y)} · {formatReleaseDate(point.row)}</small>
              </span>
            </button>
          ))}
        </div>
      </aside>
    </article>
  );
}

function releaseDatePoints(rows: LeaderboardRow[]) {
  return rows
    .map((row) => {
      const releaseTime = parseReleaseDate(row)?.getTime();
      const score = numeric(row.model_intelligence_score);
      return releaseTime !== undefined && score !== null ? { row, x: releaseTime, y: score } : null;
    })
    .filter((point): point is { row: LeaderboardRow; x: number; y: number } => point !== null);
}

function parseReleaseDate(row: LeaderboardRow) {
  if (!row.model_release_date) return null;
  const date = new Date(`${row.model_release_date}T00:00:00Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

function dateTicks(minTime: number, maxTime: number) {
  const minYear = new Date(minTime).getUTCFullYear();
  const maxYear = new Date(maxTime).getUTCFullYear();
  const ticks: number[] = [];
  for (let year = minYear; year <= maxYear; year += 1) {
    const tick = Date.UTC(year, 0, 1);
    if (tick >= minTime && tick <= maxTime) ticks.push(tick);
  }
  if (ticks.length < 2) return [minTime, maxTime];
  return ticks;
}

function formatYearTick(time: number) {
  return String(new Date(time).getUTCFullYear());
}

function IndexPlotCard({
  title,
  subtitle,
  rows,
  onOpenModel,
}: {
  title: string;
  subtitle: string;
  rows: LeaderboardRow[];
  onOpenModel: (row: LeaderboardRow) => void;
}) {
  const maxScore = Math.max(...rows.map((row) => numeric(row.model_intelligence_score) ?? 0), 0.01);
  return (
    <section className="index-plot-card intelligence-card">
      <div className="index-plot-heading">
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
        <PlotBrandStamp />
      </div>
      <div className="index-plot-list" aria-label={title}>
        {rows.length === 0 && <p className="empty-note">No rows available for the chart.</p>}
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
                <span>{indexColumnMetaLabel(row)}</span>
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
  onOpenDetails,
}: {
  row: LeaderboardRow;
  rank: number;
  maxModelSizeGb: number;
  onOpenDetails?: (row: LeaderboardRow) => void;
}) {
  return (
    <article
      className={`leaderboard-row ${rowToneClass(row)}`}
      style={{ "--provider-color": providerColor(row) } as CSSProperties}
    >
      <div className="rank-number">{String(rank).padStart(2, "0")}</div>
      <ModelIdentity row={row} showClosedSourceBadge={false}>
        {onOpenDetails && (
          <div className="leaderboard-row-actions">
            <ModelSourceLink row={row} />
            <button className="details-button" type="button" onClick={() => onOpenDetails(row)}>
              <Info size={14} aria-hidden="true" />
              Details
            </button>
          </div>
        )}
      </ModelIdentity>
      <div className="model-barplot">
        <CapabilityStrip row={row} compact maxModelSizeGb={maxModelSizeGb} />
      </div>
    </article>
  );
}

function ModelIdentity({
  row,
  children,
  showClosedSourceBadge = true,
}: {
  row: LeaderboardRow;
  children?: ReactNode;
  showClosedSourceBadge?: boolean;
}) {
  const reasoningEnabled = reasoningKey(row) !== "off";
  const ReasoningIcon = reasoningEnabled ? Lightbulb : LightbulbOff;
  const openai = isHostedOpenAIRow(row);
  return (
    <div className="model-identity">
      <LabIcon row={row} />
      <div>
        <div className="model-title-line">
          <strong>{displayModelName(row)}</strong>
          {!openai && <span className={`model-badge quant-${quantizationTone(row)}`}>{quantizationLabel(row)}</span>}
          <span
            className={`model-badge icon-only reasoning-badge ${reasoningEnabled ? "on" : "off"}`}
            title={reasoningEnabled ? `Reasoning ${reasoningValue(row)}` : "Reasoning disabled"}
          >
            <ReasoningIcon size={13} aria-hidden="true" />
          </span>
          {openai && showClosedSourceBadge && <span className="model-badge closed-source">Closed source</span>}
        </div>
        <span className="model-meta-line">{modelMetaLine(row)}</span>
        {children}
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
    const capabilityItems = [
      {
        id: "laia",
        label: "LAIA Index",
        value: score,
        display: formatPoints(score),
        accentClass: "laia-summary",
      },
      ...TEXT_CAPABILITIES.map((capability) => {
        const value = numeric(capability.value(row));
        return {
          id: capability.id,
          label: capability.label,
          value,
          display: formatPercent(value),
          accentClass: "",
        };
      }),
    ];

    return (
      <div className="capability-strip compact">
        <div className="compact-metric-grid">
          {capabilityItems.map((item) => (
            <div
              className={`compact-metric ${item.accentClass} ${item.value === null ? "missing" : ""}`}
              key={item.id}
              title={item.label}
              aria-label={`${item.label}: ${item.display}`}
            >
              <span>{item.label}</span>
              <b>{item.display}</b>
            </div>
          ))}
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

  useEffect(() => {
    if (!activeBenchmark) return;
    setActiveBenchmarkId((current) => (
      visibleBenchmarks.some((benchmark) => benchmark.id === current)
        ? current
        : activeBenchmark.id
    ));
  }, [activeBenchmark?.id, visibleBenchmarks]);

  if (!activeBenchmark) {
    return (
      <section className="page-grid benchmarks-page">
        <p className="empty-note">No completed LAIA benchmark metrics are available for the current filters.</p>
      </section>
    );
  }

  return (
    <section className="page-grid benchmarks-page">
      <header className="benchmarks-landing">
        <div className="benchmarks-landing-copy">
          <h1>Benchmarks</h1>
          <p>Open one capability at a time and inspect the metric slices behind the score.</p>
        </div>
        <div className="benchmark-tabs benchmarks-hero-tabs" aria-label="Benchmark pages">
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
      </header>
      <BenchmarkDetailPage
        benchmark={activeBenchmark}
        rows={rows}
      />
    </section>
  );
}

function BenchmarkDetailPage({
  benchmark,
  rows,
}: {
  benchmark: BenchmarkPageConfig;
  rows: LeaderboardRow[];
}) {
  return (
    <section className="benchmark-detail">
      <div className="benchmark-small-multiples">
        {benchmark.metrics
          .filter((metric) => metricHasData(rows, metric))
          .map((metric) => (
            <BenchmarkTopPlot benchmark={benchmark} metric={metric} rows={rows} key={`${benchmark.id}-${metric.id}`} />
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
  const [breakdownMode, setBreakdownMode] = useState<"region" | "language">("region");
  const availableRegions = GLOBAL_MMLU_LANGUAGE_REGIONS
    .map((region) => ({
      ...region,
      languages: region.languages.filter((language) => languageCodes.includes(language)),
    }))
    .filter((region) => region.languages.length > 0);

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
          <div className="benchmark-plot-title-row">
            <h4>Language breakdown</h4>
            <PlotInfoButton
              label="Language breakdown info"
              detail="Region view averages the exported Global MMLU Lite language accuracies inside each region. Language view shows those same exported scores one language at a time."
            />
          </div>
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
        </div>
      </div>
      {breakdownMode === "region" ? (
        <div className="language-grid region-grid">
          {availableRegions.map((region) => (
            <GlobalMMLURegionPlot
              rows={rows}
              region={region}
              limit={null}
              key={region.id}
            />
          ))}
        </div>
      ) : (
        <div className="language-grid">
          {languageCodes.map((language) => (
            <GlobalMMLULanguagePlot rows={rows} language={language} limit={null} key={language} />
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
      items={items}
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
      items={items}
      max={Math.max(...items.map((item) => item.value), 0.01)}
      title={languageLabel(language)}
    />
  );
}

function GlobalMMLUColumnPlot({
  title,
  subtitle,
  items,
  max,
}: {
  title: string;
  subtitle?: string;
  items: Array<{ row: LeaderboardRow; score: LanguageBreakdownScore; value: number }>;
  max: number;
}) {
  return (
    <article className="language-card">
      <div className="language-card-heading">
        <div>
          <h5>{title}</h5>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <PlotBrandStamp />
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
          <div className="benchmark-plot-title-row">
            <h4>Language breakdown</h4>
            <PlotInfoButton
              label="RGB language breakdown info"
              detail="Each card is the exported RGB score for one language slice."
            />
          </div>
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
        </div>
      </div>
      <div className={`language-grid ${selectedLanguage !== "all" ? "single-language" : ""}`}>
        {shownLanguages.map((language) => (
          <RGBLanguagePlot rows={rows} language={language} limit={null} key={language} />
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
      items={items}
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
  const items = benchmarkMetricItems(rows, metric);
  const max = Math.max(...items.map((item) => item.value), 0.01);
  return (
    <section className="benchmark-top-plot">
      <div className="benchmark-plot-title">
        <div>
          <div className="benchmark-plot-title-row">
            <h4>{metric.label}</h4>
            <PlotInfoButton
              label={`${metric.label} info`}
              detail={benchmarkMetricInfo(benchmark, metric)}
            />
          </div>
          <p>{benchmark.subtitle}</p>
        </div>
        <div className="benchmark-plot-meta">
          <PlotBrandStamp />
          <span>{metric.kind === "error" ? "Lower is better" : "Higher is better"}</span>
        </div>
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
  const items = benchmarkMetricItems(rows, metric);
  const max = Math.max(...items.map((item) => item.value), 0.01);
  return (
    <article className="benchmark-mini-card">
      <div className="benchmark-mini-heading">
        <div>
          <h4>{metric.label}</h4>
          <p>{benchmark.subtitle}</p>
        </div>
        <PlotBrandStamp />
      </div>
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

function benchmarkMetricInfo(benchmark: BenchmarkPageConfig, metric: BenchmarkMetric) {
  return BENCHMARK_METRIC_INFO[String(metric.metric)] ?? `${benchmark.subtitle}. ${benchmark.capability}`;
}

function PlotInfoButton({ label, detail }: { label: string; detail: string }) {
  return (
    <button className="plot-info-button" type="button" aria-label={label}>
      <span className="plot-info-glyph" aria-hidden="true">
        (i)
      </span>
      <span className="plot-info-tooltip">{detail}</span>
    </button>
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
        <div className="chart-card-meta">
          <PlotBrandStamp />
          {capability.weight ? <b>{capability.weight} LAIA pts</b> : <b>Separate</b>}
        </div>
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
      <span>{formatOutputCapHits(row)}</span>
    </div>
  );
}

function ModelsPage({
  rows,
  allRows,
  selectedModelId,
  onSelectedModelIdChange,
  filters,
  options,
  onFiltersChange,
  onClearFilters,
}: {
  rows: LeaderboardRow[];
  allRows: LeaderboardRow[];
  selectedModelId: string | null;
  onSelectedModelIdChange: (value: string | null) => void;
  filters: Filters;
  options: ReturnType<typeof optionSets>;
  onFiltersChange: (filters: Filters) => void;
  onClearFilters: () => void;
}) {
  const hasActiveFilters = (
    filters.query !== ""
    || filters.family !== "all"
    || filters.parameterSize !== "all"
    || filters.memoryFootprint !== "all"
  );
  const tableRows = hasActiveFilters ? rows : allRows;
  const rankedRows = sortModelRows(tableRows, filters.sortBy);
  const maxModelSize = rankedRows.length
    ? Math.max(...rankedRows.map(modelSizeGb).filter((size): size is number => size !== null), 0.01)
    : 0.01;
  const selectedRow = selectedModelId
    ? tableRows.find((row) => row.variant_id === selectedModelId) ?? null
    : null;

  return (
    <section className="page-grid models-page">
      <header className="models-landing">
        <div className="models-landing-copy">
          <h1>Models</h1>
          <p>Search, filter, and inspect the tested rows, source links, benchmark coverage, and runtime metadata.</p>
        </div>
      </header>

      <FilterPanel
        filters={filters}
        options={options}
        onChange={onFiltersChange}
      />

      {tableRows.length === 0 ? (
        <div className="empty-state-card">
          <Info size={40} className="empty-state-icon" aria-hidden="true" />
          <h3>No models match your criteria</h3>
          <p>
            No models were found matching your current filters:
            {filters.query !== "" && (
              <span>
                {" "}
                Search: <strong>"{filters.query}"</strong>
              </span>
            )}
            {filters.family !== "all" && (
              <span>
                {" "}
                Lab: <strong>{filters.family}</strong>
              </span>
            )}
            {filters.parameterSize !== "all" && (
              <span>
                {" "}
                Parameter size: <strong>{`Up to ${formatBillionSize(Number(filters.parameterSize))}B`}</strong>
              </span>
            )}
            {filters.memoryFootprint !== "all" && (
              <span>
                {" "}
                Memory footprint: <strong>{`Up to ${formatGigabyteSize(Number(filters.memoryFootprint))} GB`}</strong>
              </span>
            )}
            .
          </p>
          <button className="clear-filters-btn" type="button" onClick={onClearFilters}>
            Clear Filters
          </button>
        </div>
      ) : (
        <>
          <div className="ranking-list models-ranking-list">
            {rankedRows.map((row, index) => (
              <LeaderboardRowCard
                row={row}
                rank={index + 1}
                maxModelSizeGb={maxModelSize}
                onOpenDetails={(selected) => onSelectedModelIdChange(selected.variant_id)}
                key={`models-rank-${row.normalized_result_id ?? row.variant_id}`}
              />
            ))}
          </div>

          {selectedRow && <ModelDetailsDrawer row={selectedRow} onClose={() => onSelectedModelIdChange(null)} />}
        </>
      )}
    </section>
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
            <DetailItem label="Released" value={formatReleaseDate(row)} />
            <DetailItem label="Release source" value={row.model_release_source ?? "n/a"} />
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
            <DetailItem label="Output cap hits" value={formatOutputCapHits(row).replace("cap hits ", "")} />
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

function MethodologySectionHeader({ section }: { section: MethodologyChapter }) {
  return (
    <header className="methodology-section-head">
      <h2>{section.title}</h2>
      <p>{section.intro}</p>
    </header>
  );
}

function MethodologySchema({
  items,
}: {
  items: MethodologyNamedItem[];
}) {
  return (
    <dl className="methodology-schema">
      {items.map((item) => (
        <div className="methodology-schema-row" key={`${item.label}-${item.detail}`}>
          <dt>{item.label}</dt>
          <dd>{item.detail}</dd>
        </div>
      ))}
    </dl>
  );
}

function MethodologyBenchmarkCard({ benchmark }: { benchmark: MethodologyBenchmark }) {
  return (
    <article className="methodology-benchmark">
      <div className="methodology-benchmark-head">
        <span>{benchmark.subtitle}</span>
        <h3>{benchmark.title}</h3>
      </div>
      <MethodologySchema
        items={[
          { label: "What it measures", detail: benchmark.measures },
          { label: "Leaderboard metric", detail: benchmark.metric },
          { label: "Default evaluation scope / cap", detail: benchmark.scope },
          { label: "Evaluator / scoring note", detail: benchmark.evaluator },
          { label: "Inclusion", detail: benchmark.inclusion },
          { label: "Comparison caveat", detail: benchmark.caveat },
        ]}
      />
    </article>
  );
}

function MethodologyPage() {
  const [
    overviewSection,
    publicScopeSection,
    rowsSection,
    laiaSection,
    indexBenchmarksSection,
    separateBenchmarksSection,
    reproducibilitySection,
    outputsSection,
    limitationsSection,
    definitionsSection,
  ] = METHODOLOGY_CHAPTERS;

  return (
    <>
      <section className="methodology-landing">
        <div className="methodology-landing-copy">
          <p className="eyebrow">Methodology</p>
          <h1>How LAIA works</h1>
          <p>
            How Local AI Analysis builds comparable rows, computes the LAIA Index,
            and keeps benchmark evidence auditable for local 4-bit models and
            OpenAI references.
          </p>
        </div>
      </section>

      <section className="methodology-shell">
        <ChapterNav
          chapters={METHODOLOGY_CHAPTERS}
          title="On this page"
          ariaLabel="Methodology contents"
        />

        <div className="methodology-view">
          <section className="chapter-section methodology-section" id={overviewSection.id}>
            <MethodologySectionHeader section={overviewSection} />
            <div className="methodology-prose">
              <p>
                Local AI Analysis publishes a public text-only comparison surface
                for 4-bit local rows plus OpenAI reference rows. The headline
                LAIA score is meant to answer one question: how strong is this
                row on a comparable text benchmark mix without relying on an
                external judge.
              </p>
              <p>
                Vision, factuality, and safety are still part of the project, but
                they stay outside the main ranking because they introduce
                different evaluation assumptions. The public site keeps those
                metrics visible as separate diagnostics instead of blending them
                into one opaque score.
              </p>
              <p>
                The methodology is deliberately auditable. Benchmark metrics,
                sample caps, runtime, token usage, cap hits, backend metadata,
                run identifiers, and source links remain visible in the export and
                on the site.
              </p>
            </div>
          </section>

          <section className="chapter-section methodology-section" id={publicScopeSection.id}>
            <MethodologySectionHeader section={publicScopeSection} />
            <div className="methodology-split-grid">
              <article className="methodology-summary-card">
                <h3>Main comparison</h3>
                <ul>
                  {METHODOLOGY_PUBLIC_SCOPE.main.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
              <article className="methodology-summary-card">
                <h3>Reported separately</h3>
                <ul>
                  {METHODOLOGY_PUBLIC_SCOPE.separate.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
              <article className="methodology-summary-card">
                <h3>Excluded from the public ranking</h3>
                <ul>
                  {METHODOLOGY_PUBLIC_SCOPE.excluded.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
            </div>
          </section>

          <section className="chapter-section methodology-section" id={rowsSection.id}>
            <MethodologySectionHeader section={rowsSection} />
            <ol className="methodology-step-list">
              {METHODOLOGY_ROW_BUILD_STEPS.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
            <p className="methodology-note">
              In practice, the public row is not just the newest single run. It is
              a merged export surface built from the latest comparable benchmark
              evidence that exists for that model and quantization.
            </p>
          </section>

          <section className="chapter-section methodology-section" id={laiaSection.id}>
            <MethodologySectionHeader section={laiaSection} />
            <div className="methodology-prose">
              <p>
                LAIA Index is the project&apos;s headline text-only score. The
                stored values are normalized from 0 to 1 in the database and export,
                while the website renders them as points out of 100.
              </p>
              <p>
                The three intelligence fields serve different jobs: the full
                ranking score, the benchmark-family coverage check, and the
                within-coverage average for rows that are still missing parts of
                the text suite.
              </p>
            </div>
            <div className="methodology-score-grid">
              {METHODOLOGY_LAIA_FIELDS.map((field) => (
                <article className="methodology-score-card" key={field.term}>
                  <code>{field.term}</code>
                  <p>{field.description}</p>
                </article>
              ))}
            </div>
            <div className="methodology-weight-strip" aria-label="Published LAIA weights">
              {METHODOLOGY_PUBLISHED_WEIGHTS.map((item) => (
                <article className="methodology-weight-item" key={item.label}>
                  <span>{item.label}</span>
                  <strong>{item.detail}</strong>
                </article>
              ))}
            </div>
            <p className="methodology-note">
              Published LAIA weighting is shown once here. OCRBench v2 and MMMU
              stay separate as vision metrics, while SimpleQA and HarmBench stay
              separate because they require a judge.
            </p>
          </section>

          <section className="chapter-section methodology-section" id={indexBenchmarksSection.id}>
            <MethodologySectionHeader section={indexBenchmarksSection} />
            <div className="methodology-benchmark-list">
              {METHODOLOGY_INDEX_BENCHMARKS.map((benchmark) => (
                <MethodologyBenchmarkCard benchmark={benchmark} key={benchmark.id} />
              ))}
            </div>
          </section>

          <section className="chapter-section methodology-section" id={separateBenchmarksSection.id}>
            <MethodologySectionHeader section={separateBenchmarksSection} />
            <div className="methodology-benchmark-list">
              {METHODOLOGY_SEPARATE_BENCHMARKS.map((benchmark) => (
                <MethodologyBenchmarkCard benchmark={benchmark} key={benchmark.id} />
              ))}
            </div>
          </section>

          <section className="chapter-section methodology-section" id={reproducibilitySection.id}>
            <MethodologySectionHeader section={reproducibilitySection} />
            <div className="methodology-prose">
              <p>
                Local AI Analysis treats each row as an auditable local measurement.
                Shortcut-generated configs pin upstream dataset revisions, use
                deterministic caps where the suite would otherwise become too
                heavy, and keep benchmark-specific settings in the generated
                config and exported artifacts.
              </p>
            </div>
            <div className="methodology-dual-grid">
              <article className="methodology-summary-card">
                <h3>Suite aliases</h3>
                <MethodologySchema items={METHODOLOGY_SUITE_ALIASES} />
              </article>
              <article className="methodology-summary-card">
                <h3>Deterministic default caps</h3>
                <MethodologySchema items={METHODOLOGY_DEFAULT_CAPS} />
              </article>
            </div>
            <div className="methodology-dual-grid">
              <article className="methodology-summary-card">
                <h3>Reasoning and context defaults</h3>
                <ul>
                  <li>Local shortcut commands default to <code>reasoning-effort none</code>.</li>
                  <li>OpenAI defaults to <code>reasoning-effort auto</code>, which resolves by model family.</li>
                  <li>Shortcut commands default to <code>context-length 8192</code> for Ollama and LM Studio.</li>
                  <li>Changing prompts or benchmark settings creates new sample rows instead of silently reusing incompatible ones.</li>
                </ul>
              </article>
              <article className="methodology-summary-card">
                <h3>Resume behavior</h3>
                <p>
                  <code>--resume-samples</code> reuses matching sample rows from an
                  existing benchmark&apos;s <code>samples.jsonl</code>. The match
                  includes dataset identity, split, sample id, and rendered prompt,
                  so changing prompts or benchmark settings starts new sample rows
                  instead of mixing incompatible evidence.
                </p>
              </article>
            </div>
            <article className="methodology-summary-card">
              <h3>Pinned dataset revisions</h3>
              <MethodologySchema items={METHODOLOGY_PINNED_REVISIONS} />
            </article>
            <article className="methodology-summary-card">
              <h3>Recorded metadata and run signals</h3>
              <MethodologySchema items={METHODOLOGY_RECORDED_METADATA} />
            </article>
          </section>

          <section className="chapter-section methodology-section" id={outputsSection.id}>
            <MethodologySectionHeader section={outputsSection} />
            <div className="methodology-prose">
              <p>
                The website is fed from exported normalized rows, but the full local
                measurement stack stays on disk under <code>results/</code>. That
                includes the DuckDB database, raw run events, generated configs,
                and benchmark-specific sample and summary artifacts.
              </p>
              <p>
                In the schema, the auditable pipeline runs from <code>base_model</code>
                {" "}and <code>model_variant</code> through <code>benchmark_run</code>,
                {" "}<code>benchmark_task</code>, and <code>benchmark_result</code>,
                {" "}then into the leaderboard-facing <code>normalized_result</code>
                {" "}row that the site exports.
              </p>
            </div>
            <div className="methodology-dual-grid">
              <article className="methodology-summary-card">
                <h3>What is written under results/</h3>
                <MethodologySchema items={METHODOLOGY_OUTPUTS} />
              </article>
              <article className="methodology-summary-card">
                <h3>What a normalized row represents</h3>
                <p>
                  A normalized row is the leaderboard-facing record for one public
                  model surface. It contains the exported benchmark metrics,
                  intelligence fields, source and backend metadata, and the merged
                  run signals used by the site.
                </p>
              </article>
            </div>
            <article className="methodology-summary-card">
              <h3>Auditable fields still visible</h3>
              <ul>
                {METHODOLOGY_AUDIT_FIELDS.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          </section>

          <section className="chapter-section methodology-section" id={limitationsSection.id}>
            <MethodologySectionHeader section={limitationsSection} />
            <div className="methodology-key-grid">
              {METHODOLOGY_LIMITATIONS.map((item) => (
                <article className="methodology-key-item" key={item.label}>
                  <h3>{item.label}</h3>
                  <p>{item.detail}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="chapter-section methodology-section" id={definitionsSection.id}>
            <MethodologySectionHeader section={definitionsSection} />
            <dl className="methodology-definition-grid">
              {METHODOLOGY_DEFINITIONS.map((item) => (
                <div className="methodology-definition" key={item.term}>
                  <dt>{item.term}</dt>
                  <dd>{item.description}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </section>
    </>
  );
}

function MissionPage() {
  return (
    <section className="page-grid mission-page">
      <header className="mission-landing">
        <div className="mission-landing-copy">
          <p className="eyebrow">Mission</p>
          <h1>Make local model capability impossible to ignore.</h1>
          <p>
            Local AI Analysis exists to increase awareness of what small and tiny
            language models can already do on consumer hardware and edge devices,
            to show when they are already competitive with famous closed-source
            API models that come with per-call cost, and to make that progress
            legible through public, comparable evidence.
          </p>
        </div>
      </header>

      <section className="mission-section">
        <div className="mission-section-head">
          <h2>Why this project exists</h2>
          <p>
            Too much of the conversation around AI still assumes that capability
            only lives in the cloud. That misses what becomes possible when
            performant models fit directly into products, machines, and devices
            that people actually own and operate.
          </p>
        </div>
        <div className="mission-pillar-grid">
          {MISSION_PILLARS.map((item) => (
            <article className="mission-pillar" key={item.title}>
              <span className="mission-pillar-icon" aria-hidden="true">
                {item.icon}
              </span>
              <h3>{item.title}</h3>
              <p>{item.copy}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="mission-section mission-section-split">
        <article className="mission-panel">
          <h2>What we want to push forward</h2>
          <ul className="mission-list">
            {MISSION_FOCUS_AREAS.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>
        <article className="mission-panel">
          <h2>What local SLMs and TLMs unlock</h2>
          <ul className="mission-list">
            {MISSION_OUTCOMES.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>
      </section>

      <section className="mission-statement">
        <p>
          The long-term goal is simple: make it easier for builders, researchers,
          and organizations to take local models seriously as real deployment
          infrastructure, especially when privacy, sovereignty, offline use,
          latency, or edge integration matter more than raw scale.
        </p>
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
  const outputCapHitCount = numeric(merged.benchmark_output_cap_hit_count) ?? 0;
  const outputCapHitSamples = numeric(merged.benchmark_output_cap_hit_samples) ?? 0;
  merged.benchmark_truncated_rate = samples > 0 ? truncatedCount / samples : null;
  merged.benchmark_output_cap_hit_rate = outputCapHitSamples > 0 ? outputCapHitCount / outputCapHitSamples : null;
  merged.benchmark_output_cap_breakdown = mergeOutputCapBreakdowns(sources);
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

function mergeOutputCapBreakdowns(rows: LeaderboardRow[]) {
  const byBenchmark = new Map<string, OutputCapBreakdown>();
  for (const row of rows) {
    for (const item of outputCapBreakdown(row)) {
      const existing = byBenchmark.get(item.benchmark) ?? {
        benchmark: item.benchmark,
        max_output_tokens: item.max_output_tokens,
        hits: 0,
        samples: 0,
        rate: null,
      };
      existing.hits += item.hits;
      existing.samples += item.samples;
      existing.max_output_tokens = item.max_output_tokens || existing.max_output_tokens;
      existing.rate = existing.samples > 0 ? existing.hits / existing.samples : null;
      byBenchmark.set(item.benchmark, existing);
    }
  }
  return Array.from(byBenchmark.values()).sort(
    (a, b) => outputCapSortIndex(a.benchmark) - outputCapSortIndex(b.benchmark),
  );
}

function outputCapBreakdown(row: LeaderboardRow) {
  const raw = row.benchmark_output_cap_breakdown;
  let items: unknown = raw;
  if (typeof raw === "string") {
    try {
      items = JSON.parse(raw);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(items)) return [];
  return items.flatMap((item): OutputCapBreakdown[] => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const value = item as Record<string, unknown>;
    const benchmark = typeof value.benchmark === "string" ? value.benchmark : "";
    const maxTokens = numeric(value.max_output_tokens);
    const hits = numeric(value.hits);
    const samples = numeric(value.samples);
    if (!benchmark || maxTokens === null || hits === null || samples === null) return [];
    return [{
      benchmark,
      max_output_tokens: maxTokens,
      hits,
      samples,
      rate: samples > 0 ? hits / samples : null,
    }];
  });
}

function outputCapSortIndex(benchmark: string) {
  const index = OUTPUT_CAP_ORDER.indexOf(benchmark);
  return index === -1 ? OUTPUT_CAP_ORDER.length : index;
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
  const parameterLimit = filters.parameterSize === "all" ? null : Number(filters.parameterSize);
  const memoryLimit = filters.memoryFootprint === "all" ? null : Number(filters.memoryFootprint);
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
    const parameterSize = numeric(row.parameter_size_b);
    const memoryFootprint = modelSizeGb(row);
    return (
      (!query || searchable.includes(query)) &&
      (filters.family === "all" || providerLabel(row) === filters.family) &&
      (parameterLimit === null || (parameterSize !== null && parameterSize > 0 && parameterSize <= parameterLimit)) &&
      (memoryLimit === null || (memoryFootprint !== null && memoryFootprint > 0 && memoryFootprint <= memoryLimit))
    );
  });
}

function topIndexRows(rows: LeaderboardRow[]) {
  return rows
    .filter((row) => {
      const score = numeric(row.model_intelligence_score);
      return score !== null && score > 0;
    })
    .sort((a, b) => scoreForRank(b) - scoreForRank(a));
}

function topRowsByCapability(rows: LeaderboardRow[], limit = 5) {
  return TEXT_CAPABILITIES.map((capability) => ({
    capability,
    rows: rows
      .map((row) => ({ row, value: capability.value(row) }))
      .filter((item): item is { row: LeaderboardRow; value: number } => item.value !== null)
      .sort((a, b) => b.value - a.value || scoreForRank(b.row) - scoreForRank(a.row))
      .slice(0, limit),
  }));
}

function topRowsByUseCase(rows: LeaderboardRow[], leaderboard: UseCaseLeaderboard, limit = 5) {
  return rows
    .filter((row) => !isHostedOpenAIRow(row))
    .map((row) => ({ row, ...useCaseLeaderboardScore(row, leaderboard) }))
    .filter((item): item is { row: LeaderboardRow; score: number; coverage: number } =>
      item.score !== null && item.coverage >= leaderboard.minCoverage
    )
    .sort((a, b) => b.score - a.score || scoreForRank(b.row) - scoreForRank(a.row))
    .slice(0, limit);
}

function useCaseLeaderboardScore(row: LeaderboardRow, leaderboard: UseCaseLeaderboard) {
  let weightedSum = 0;
  let coveredWeight = 0;
  for (const weight of leaderboard.weights) {
    const value = numeric(row[weight.metric]);
    if (value === null) continue;
    weightedSum += Math.max(0, Math.min(1, value)) * weight.weight;
    coveredWeight += weight.weight;
  }
  if (coveredWeight <= 0) return { score: null, coverage: 0 };
  return { score: weightedSum / coveredWeight, coverage: coveredWeight };
}

function useCaseSortValue(row: LeaderboardRow, sortBy: string) {
  const leaderboard = useCaseLeaderboardForSort(sortBy);
  if (!leaderboard) return null;
  if (isHostedOpenAIRow(row)) return null;
  const score = useCaseLeaderboardScore(row, leaderboard);
  return score.coverage >= leaderboard.minCoverage ? score.score : null;
}

function useCaseLeaderboardForSort(sortBy: string) {
  const id = sortBy.startsWith("use-case:") ? sortBy.slice("use-case:".length) : "";
  return USE_CASE_LEADERBOARDS.find((leaderboard) => leaderboard.id === id) ?? null;
}

function comparableRowKey(row: LeaderboardRow) {
  return `${quantizationGroupKey(row)}|${quantizationKey(row)}`;
}

function optionSets(rows: LeaderboardRow[]) {
  const familyOptions = unique(rows.map((row) => providerLabel(row))).map((family) => ({
    value: family,
    label: family,
  }));
  const parameterSizeOptions = [...new Set(
    rows
      .map((row) => numeric(row.parameter_size_b))
      .filter((value): value is number => value !== null && value > 0)
      .map((value) => Number(value.toFixed(3))),
  )]
    .sort((a, b) => a - b)
    .map((value) => ({
      value: String(value),
      label: `Up to ${formatBillionSize(value)}B`,
    }));
  const memoryFootprintOptions = [...new Set(
    rows
      .map((row) => modelSizeGb(row))
      .filter((value): value is number => value !== null && value > 0)
      .map((value) => Number(value.toFixed(3))),
  )]
    .sort((a, b) => a - b)
    .map((value) => ({
      value: String(value),
      label: `Up to ${formatGigabyteSize(value)} GB`,
    }));
  return {
    families: familyOptions,
    parameterSizes: parameterSizeOptions,
    memoryFootprints: memoryFootprintOptions,
    sortOptions: [
      ...MODEL_SORT_OPTIONS,
      ...USE_CASE_LEADERBOARDS.map((leaderboard) => ({
        value: `use-case:${leaderboard.id}`,
        label: leaderboard.label,
      })),
    ],
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

function sortModelRows(rows: LeaderboardRow[], sortBy: string) {
  const capability = TEXT_CAPABILITIES.find((item) => item.id === sortBy) ?? null;
  const useCaseLeaderboard = useCaseLeaderboardForSort(sortBy);
  return [...rows].sort((a, b) => {
    if (sortBy === "parameter-size") {
      const delta = compareNullableNumbers(numeric(a.parameter_size_b), numeric(b.parameter_size_b), "asc");
      if (delta !== 0) return delta;
    } else if (sortBy === "memory-footprint") {
      const delta = compareNullableNumbers(modelSizeGb(a), modelSizeGb(b), "asc");
      if (delta !== 0) return delta;
    } else if (capability) {
      const delta = compareNullableNumbers(capability.value(a), capability.value(b), "desc");
      if (delta !== 0) return delta;
    } else if (useCaseLeaderboard) {
      const delta = compareNullableNumbers(useCaseSortValue(a, sortBy), useCaseSortValue(b, sortBy), "desc");
      if (delta !== 0) return delta;
    } else {
      const delta = compareNullableNumbers(numeric(a.model_intelligence_score), numeric(b.model_intelligence_score), "desc");
      if (delta !== 0) return delta;
    }

    const scoreDelta = scoreForRank(b) - scoreForRank(a);
    if (scoreDelta !== 0) return scoreDelta;
    return displayModelName(a).localeCompare(displayModelName(b), undefined, { numeric: true, sensitivity: "base" });
  });
}

function compareNullableNumbers(a: number | null, b: number | null, direction: "asc" | "desc") {
  if (a === null && b === null) return 0;
  if (a === null) return 1;
  if (b === null) return -1;
  return direction === "asc" ? a - b : b - a;
}

function rowToneClass(row: LeaderboardRow) {
  if (isHostedOpenAIRow(row)) return "hosted-openai-row";
  if (!isFourBitRow(row)) return "alternate-version-row";
  return "";
}

function isHostedOpenAIRow(row: LeaderboardRow) {
  return providerLabel(row) === "OpenAI";
}

function isPublicLeaderboardRow(row: LeaderboardRow) {
  return isFourBitRow(row) || isHostedOpenAIRow(row);
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
  if (/nemotron[-_\s]*3[-_\s]*nano/i.test(lastSegment)) return "Nemotron 3 Nano 4B";
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
  const falconH = clean.match(/\bfalcon\s*h\s*(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)([bm])\b/i);
  if (falconH) {
    const size = falconH[3].toLowerCase() === "m" ? Number(falconH[2]) / 1000 : Number(falconH[2]);
    return `Falcon H${falconH[1]} ${formatBillionSize(size)}B`;
  }
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

function modelMetaLine(row: LeaderboardRow) {
  const items = [providerLabel(row)];
  if (isHostedOpenAIRow(row)) {
    items.push("Closed source");
  } else {
    const parameter = displayParameter(row);
    const footprint = formatModelSize(row);
    if (parameter !== "n/a") items.push(parameter);
    if (footprint !== "n/a") items.push(footprint);
  }
  const release = formatReleaseDate(row);
  if (release !== "release n/a") items.push(release);
  return items.join(" · ");
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
  return "Closed source";
}

function indexColumnMetaLabel(row: LeaderboardRow) {
  if (isHostedOpenAIRow(row)) return "Closed source";
  return `${quantizationLabel(row)} · ${formatModelSize(row)}`;
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
  if (apiModelValue && apiModelValue in KNOWN_MODEL_FILE_SIZES) return KNOWN_MODEL_FILE_SIZES[apiModelValue];
  return null;
}

const KNOWN_MODEL_FILE_SIZES: Record<string, number> = {
  "lfm2.5-350m@bf16": 711_500_000,
  "lfm2.5-350m@q8_0": 379_200_000,
  "lfm2.5-350m@q4_k_m": 229_300_000,
  "nemotron-3-nano:4b": 3_006_477_107,
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

function formatReleaseDate(row: LeaderboardRow) {
  if (!row.model_release_date) return "release n/a";
  const date = new Date(`${row.model_release_date}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return row.model_release_date;
  return date.toLocaleDateString(undefined, { month: "short", year: "numeric" });
}

function formatSamples(row: LeaderboardRow) {
  const value = numeric(row.benchmark_samples);
  if (value === null) return "samples n/a";
  return `${value.toLocaleString()} samples`;
}

function formatOutputCapHits(row: LeaderboardRow) {
  const count = numeric(row.benchmark_output_cap_hit_count) ?? numeric(row.benchmark_truncated_count);
  const samples = numeric(row.benchmark_output_cap_hit_samples) ?? numeric(row.benchmark_samples);
  const rate = numeric(row.benchmark_output_cap_hit_rate) ?? (
    count !== null && samples !== null && samples > 0 ? count / samples : null
  );
  if (rate === null) return "cap hits n/a";
  return count === null ? `${(rate * 100).toFixed(1)}% cap hits` : `${(rate * 100).toFixed(1)}% cap hits · ${count}`;
}

function coverageLabel(row: LeaderboardRow) {
  const complete = CAPABILITIES.filter((capability) => capability.value(row) !== null).length;
  return `${complete}/${CAPABILITIES.length}`;
}

function modelSourceLink(row: LeaderboardRow) {
  if (isHostedOpenAIRow(row)) {
    return { label: "Closed source", href: "", kind: "hosted" };
  }
  const candidates = [
    row.model_repo,
    variantConfigValue(row, "model_repo"),
    apiModel(row),
  ].filter((value): value is string => typeof value === "string" && value.trim().length > 0);
  const exactRepo = candidates.find((value) => /^[\w.-]+\/[\w.-]+$/.test(value) && !value.includes("@"));
  if (exactRepo) {
    return { label: "HF repo", href: `https://huggingface.co/${exactRepo}`, kind: "exact" };
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

function humanizeColumn(value: string) {
  return value.replace(/_/g, " ");
}

function formatPercent(value?: number | null) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : `${(value * 100).toFixed(1)}%`;
}

function formatWeightPercent(value: number) {
  const percent = Math.round(value * 1000) / 10;
  return `${Number.isInteger(percent) ? percent.toFixed(0) : percent.toFixed(1)}%`;
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

function formatGigabyteSize(value: number) {
  if (value < 10) return value.toFixed(1).replace(/\.0$/, "");
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
