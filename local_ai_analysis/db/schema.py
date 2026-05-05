SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS base_model (
    id TEXT PRIMARY KEY,
    family TEXT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    parameter_size_b DOUBLE,
    architecture TEXT,
    license TEXT,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS quantization (
    id TEXT PRIMARY KEY,
    quantization_type TEXT NOT NULL UNIQUE,
    bits DOUBLE,
    scheme TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS model_variant (
    id TEXT PRIMARY KEY,
    base_model_id TEXT NOT NULL,
    variant_name TEXT NOT NULL,
    model_repo TEXT,
    local_path TEXT,
    file_name TEXT,
    quantization_id TEXT NOT NULL,
    precision TEXT,
    parameter_size_b DOUBLE,
    checksum_sha256 TEXT,
    file_size_bytes BIGINT,
    is_baseline BOOLEAN DEFAULT false,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp,
    UNIQUE (base_model_id, variant_name)
);

CREATE TABLE IF NOT EXISTS backend_profile (
    id TEXT PRIMARY KEY,
    backend_name TEXT NOT NULL,
    backend_type TEXT,
    backend_version TEXT,
    backend_commit TEXT,
    command TEXT,
    extra_json TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS hardware_profile (
    id TEXT PRIMARY KEY,
    hardware_hash TEXT NOT NULL UNIQUE,
    os_name TEXT,
    os_version TEXT,
    python_version TEXT,
    cpu_model TEXT,
    cpu_count INTEGER,
    ram_total_bytes BIGINT,
    gpu_name TEXT,
    gpu_memory_bytes BIGINT,
    accelerator TEXT,
    extra_json TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS benchmark_task (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    task_version TEXT,
    few_shot INTEGER,
    prompt_template TEXT,
    decoding_json TEXT,
    metric_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT current_timestamp,
    UNIQUE (name, task_type, task_version, few_shot, metric_name)
);

CREATE TABLE IF NOT EXISTS benchmark_run (
    id TEXT PRIMARY KEY,
    run_uuid TEXT NOT NULL UNIQUE,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    config_path TEXT,
    seed INTEGER,
    command_args_json TEXT,
    hardware_profile_id TEXT,
    backend_profile_id TEXT,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS benchmark_result (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE,
    unit TEXT,
    raw_json TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS normalized_result (
    id TEXT PRIMARY KEY,
    variant_id TEXT NOT NULL,
    run_id TEXT,
    base_model_id TEXT NOT NULL,
    baseline_variant_id TEXT,
    global_mmlu_lite_pass_at_1 DOUBLE,
    global_mmlu_lite_micro_pass_at_1 DOUBLE,
    global_mmlu_lite_invalid_rate DOUBLE,
    ifbench_prompt_level_loose DOUBLE,
    ifbench_instruction_level_loose DOUBLE,
    ifbench_prompt_level_strict DOUBLE,
    ifbench_instruction_level_strict DOUBLE,
    bfcl_v4_selected_accuracy DOUBLE,
    bfcl_v4_invalid_rate DOUBLE,
    bfcl_v4_non_live_accuracy DOUBLE,
    bfcl_v4_live_accuracy DOUBLE,
    bfcl_v4_multi_turn_accuracy DOUBLE,
    bfcl_v4_agentic_accuracy DOUBLE,
    benchmark_runtime_seconds DOUBLE,
    metadata_json TEXT,
    created_at TIMESTAMP DEFAULT current_timestamp,
    UNIQUE (variant_id, run_id)
);
"""
