/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx2webnn::model_validation::manifest::Selection;
use onnx2webnn::model_validation::{run_manifest_validation, RunOptions, WeightMode};
use onnx2webnn::{
    cache_onnx_model, convert_onnx_with_backend, export_onnx,
    validate_cached_model_with_inputs_on_backend, validate_cached_model_with_options_on_backend,
    BackendChoice, ConvertOptions,
};

#[derive(Parser)]
#[command(name = "onnx2webnn")]
#[command(
    about = "Convert ONNX models to WebNN via MLGraphBuilder",
    long_about = None
)]
struct Cli {
    /// Enable debug output
    #[arg(long, global = true)]
    debug: bool,

    /// Execution backend for graph builds and validation (default: ort)
    #[arg(long, global = true, default_value_t = BackendChoice::Ort)]
    backend: BackendChoice,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print the code-backed operator catalog as JSON
    OperatorCatalog,
    /// Lower ONNX and either build or export the WebNN graph
    Convert {
        /// Input ONNX model path
        #[arg(long)]
        input: String,

        /// Override a symbolic dimension, e.g. batch_size=1 (repeatable)
        #[arg(long = "override-dim")]
        override_dims: Vec<String>,

        /// JSON file with dimension overrides (freeDimensionOverrides object)
        #[arg(long = "override-dims-file")]
        override_dims_file: Option<String>,

        /// Freeze a graph input to a constant, e.g. use_cache_branch=false
        /// (repeatable). Constant `If` gates are then inlined.
        #[arg(long = "pin-input")]
        pin_inputs: Vec<String>,

        /// Zero-fill external tensors whose data file is missing (weight-stripped
        /// skeleton models; graph structure only)
        #[arg(long = "allow-missing-external-data")]
        allow_missing_external_data: bool,

        /// Enable constant folding (Shape/Gather/Concat/Reshape pipelines)
        #[arg(long)]
        optimize: bool,

        /// Preserve unresolved symbolic input dims as dynamic metadata (experimental)
        #[arg(long)]
        experimental_dynamic_inputs: bool,

        /// Export the graph and ONNX model to the OS cache without a backend build
        #[arg(long, conflicts_with = "validate")]
        no_backend_graph_build: bool,

        /// Export, reload, dispatch deterministic inputs, and compare with native ORT
        #[arg(long)]
        validate: bool,

        /// Validate existing cache artifacts only; does not read or convert the source ONNX.
        #[arg(long, conflicts_with_all = ["validate", "no_backend_graph_build"])]
        validate_cached: bool,
    },

    /// Numerically validate manifest models against reloaded WebNN artifacts
    ValidateModels {
        /// Manifest selection: all or match=<text>
        #[arg(long)]
        selection: String,

        /// Weight source. Only publisher weights are currently supported.
        #[arg(long, default_value_t = WeightMode::Real)]
        weights: WeightMode,

        /// Override tests/models/manifest.json
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Number of parallel light-model workers; heavy models remain sequential
        #[arg(long, default_value_t = 1)]
        jobs: usize,
    },

    /// Validate one or more local ONNX files against freshly exported WebNN artifacts
    ValidateFiles {
        /// Local ONNX file; repeat for each case
        #[arg(long = "model", required = true)]
        models: Vec<PathBuf>,
        /// Complete SafeTensors input file, paired by position with each --model
        #[arg(long = "inputs")]
        inputs: Vec<PathBuf>,
        /// Symbolic dimension binding shared by every model, NAME=VALUE
        #[arg(long = "override-dim")]
        override_dims: Vec<String>,
        /// Pin a graph input in every model, NAME=VALUE
        #[arg(long = "pin-input")]
        pin_inputs: Vec<String>,
    },
}

fn parse_overrides(specs: Vec<String>) -> anyhow::Result<HashMap<String, u32>> {
    let mut values = HashMap::new();
    for spec in specs {
        let (name, value) = spec.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("invalid --override-dim {spec:?}; expected NAME=VALUE")
        })?;
        if name.trim().is_empty() {
            anyhow::bail!("empty --override-dim name");
        }
        let parsed = value.trim().parse::<u32>()?;
        if values.insert(name.trim().to_string(), parsed).is_some() {
            anyhow::bail!("duplicate --override-dim {name}");
        }
    }
    Ok(values)
}

fn parse_pins(specs: Vec<String>) -> anyhow::Result<HashMap<String, i64>> {
    let mut values = HashMap::new();
    for spec in specs {
        let (name, value) = onnx2webnn::onnx::convert::parse_pinned_input(&spec)?;
        if values.insert(name.clone(), value).is_some() {
            anyhow::bail!("duplicate --pin-input {name}");
        }
    }
    Ok(values)
}

fn local_webnn_path(
    model: &Path,
    overrides: &HashMap<String, u32>,
    pins: &HashMap<String, i64>,
) -> anyhow::Result<PathBuf> {
    let mut hasher = Sha256::new();
    let mut field = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    field(b"onnx2webnn-local-v2");
    field(model.canonicalize()?.to_string_lossy().as_bytes());
    let mut dims: Vec<_> = overrides.iter().collect();
    dims.sort_unstable_by_key(|(key, _)| *key);
    for (key, value) in dims {
        field(b"dimension");
        field(key.as_bytes());
        field(&value.to_le_bytes());
    }
    let mut pinned: Vec<_> = pins.iter().collect();
    pinned.sort_unstable_by_key(|(key, _)| *key);
    for (key, value) in pinned {
        field(b"pin");
        field(key.as_bytes());
        field(&value.to_le_bytes());
    }
    drop(field);
    let key = format!("{:x}", hasher.finalize());
    Ok(onnx2webnn::cache::webnn_cache_dir()
        .map_err(anyhow::Error::msg)?
        .join(format!("local-{}.webnn", &key[..24])))
}

fn cache_paths(input: &Path) -> anyhow::Result<(PathBuf, PathBuf)> {
    let canonical = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
    let mut hash = 0xcbf29ce484222325u64;
    for byte in canonical.to_string_lossy().bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let safe_stem: String = stem
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let key = format!("{safe_stem}-{hash:016x}");
    Ok((
        onnx2webnn::cache::onnx_cache_dir()
            .map_err(anyhow::Error::msg)?
            .join(format!("{key}.onnx")),
        onnx2webnn::cache::webnn_cache_dir()
            .map_err(anyhow::Error::msg)?
            .join(format!("{key}.webnn")),
    ))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let backend = cli.backend;
    if !matches!(cli.cmd, Command::OperatorCatalog) {
        std::env::set_var("O2W_OPERATOR_REPORT", "1");
    }
    if cli.debug {
        onnx2webnn::debug::enable();
        std::env::set_var("O2W_OPERATOR_REPORT_VERBOSE", "1");
    }

    match cli.cmd {
        Command::OperatorCatalog => {
            println!(
                "{}",
                serde_json::to_string_pretty(
                    onnx2webnn::onnx::ops::catalog::OPERATOR_CAPABILITIES
                )?
            );
        }
        Command::Convert {
            input,
            override_dims,
            override_dims_file,
            pin_inputs,
            allow_missing_external_data,
            optimize,
            experimental_dynamic_inputs,
            no_backend_graph_build,
            validate,
            validate_cached,
        } => {
            if no_backend_graph_build && backend != BackendChoice::Ort {
                anyhow::bail!(
                    "convert --no-backend-graph-build is backend-independent; omit --backend or use --validate instead"
                );
            }
            if validate || validate_cached {
                std::env::set_var("O2W_OPERATOR_REPORT", "1");
            }
            let mut free_dim_overrides = if let Some(path) = override_dims_file {
                let content = std::fs::read_to_string(&path)?;
                let json: serde_json::Value = serde_json::from_str(&content)?;
                let overrides = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "override-dims-file must be a JSON object (optionally nested under freeDimensionOverrides)"
                        )
                    })?;

                let mut map = HashMap::new();
                for (name, value) in overrides {
                    let parsed = value.as_u64().ok_or_else(|| {
                        anyhow::anyhow!(
                            "override value for '{}' must be an integer, got {}",
                            name,
                            value
                        )
                    })?;
                    map.insert(name.to_string(), parsed as u32);
                }
                map
            } else {
                HashMap::new()
            };

            for override_dim in override_dims {
                let parts: Vec<&str> = override_dim.split('=').collect();
                if parts.len() != 2 {
                    return Err(anyhow::anyhow!(
                        "Invalid override-dim format: '{}'. Expected NAME=VALUE",
                        override_dim
                    ));
                }
                let name = parts[0].trim().to_string();
                let value: u32 = parts[1]
                    .trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("Invalid dimension value: '{}'", parts[1]))?;
                free_dim_overrides.insert(name, value);
            }

            let input_path = Path::new(&input);

            let mut pinned_inputs = HashMap::new();
            for spec in pin_inputs {
                let (name, value) = onnx2webnn::onnx::convert::parse_pinned_input(&spec)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                pinned_inputs.insert(name, value);
            }

            let validation_overrides = free_dim_overrides.clone();
            if validate_cached {
                let (onnx_cache, webnn_cache) = cache_paths(input_path)?;
                if !onnx_cache.exists() || !webnn_cache.exists() {
                    return Err(anyhow::anyhow!(
                        "cached validation requires {} and {}; run with --no-backend-graph-build or --validate first",
                        onnx_cache.display(),
                        webnn_cache.display()
                    ));
                }
                println!("Validating cached graph with {backend} backend against native CPU ORT");
                let summary = validate_cached_model_with_options_on_backend(
                    &onnx_cache,
                    &webnn_cache,
                    &validation_overrides,
                    &HashMap::new(),
                    backend,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                println!(
                    "✓ cached deterministic validation passed ({} inputs, {} outputs)",
                    summary.input_count, summary.output_count
                );
                return Ok(());
            }

            let cache_paths = (no_backend_graph_build || validate)
                .then(|| cache_paths(input_path))
                .transpose()?;
            if let Some((onnx_cache, _)) = &cache_paths {
                cache_onnx_model(input_path, onnx_cache, allow_missing_external_data)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            }

            let options = ConvertOptions {
                free_dim_overrides,
                optimize,
                experimental_dynamic_inputs,
                pinned_inputs,
                zero_fill_missing_external_data: allow_missing_external_data,
                output_path: cache_paths.as_ref().map(|(_, webnn)| webnn.clone()),
            };

            if no_backend_graph_build {
                export_onnx(input_path, options).map_err(|e| anyhow::anyhow!("{e}"))?;
                println!("✓ portable WebNN export succeeded for {input} (no backend build)");
            } else {
                println!("Building converted graph with {backend} backend");
                let validated = convert_onnx_with_backend(input_path, options, backend)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                println!(
                    "✓ {:?} graph build succeeded for {input}",
                    validated.context.rustnn_backend()
                );
            }
            if let Some((onnx_cache, webnn_cache)) = &cache_paths {
                println!("✓ cached ONNX at {}", onnx_cache.display());
                println!("✓ cached WebNN at {}", webnn_cache.display());
                if validate {
                    println!(
                        "Validating reloaded graph with {backend} backend against native CPU ORT"
                    );
                    let summary = validate_cached_model_with_options_on_backend(
                        onnx_cache,
                        webnn_cache,
                        &validation_overrides,
                        &HashMap::new(),
                        backend,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    println!(
                        "✓ deterministic validation passed ({} inputs, {} outputs)",
                        summary.input_count, summary.output_count
                    );
                }
            }
        }
        Command::ValidateModels {
            selection,
            weights,
            manifest,
            jobs,
        } => {
            std::env::set_var("O2W_OPERATOR_REPORT", "1");
            let selection = Selection::parse(&selection).map_err(anyhow::Error::msg)?;
            let mut options = RunOptions::new(selection, weights).map_err(anyhow::Error::msg)?;
            options.jobs = jobs;
            options.backend = backend;
            println!("Validating reloaded graphs with {backend} backend against native CPU ORT");
            if let Some(manifest) = manifest {
                options.manifest = manifest;
            }
            let summary = run_manifest_validation(options).map_err(anyhow::Error::msg)?;
            println!("{summary}");
            if summary.has_failures() {
                return Err(anyhow::anyhow!(
                    "{} of {} model validations failed",
                    summary.failed.len(),
                    summary.selected
                ));
            }
        }
        Command::ValidateFiles {
            models,
            inputs,
            override_dims,
            pin_inputs,
        } => {
            std::env::set_var("O2W_OPERATOR_REPORT", "1");
            if !inputs.is_empty() && inputs.len() != models.len() {
                anyhow::bail!("--inputs must be omitted or supplied once per --model ({} models, {} input files)",
                    models.len(), inputs.len());
            }
            let overrides = parse_overrides(override_dims)?;
            let pins = parse_pins(pin_inputs)?;
            println!("Validating reloaded graphs with {backend} backend against native CPU ORT");
            let mut failed = Vec::new();
            let total = models.len();
            for (index, model) in models.iter().enumerate() {
                let run = || -> anyhow::Result<onnx2webnn::ValidationSummary> {
                    let (effective, supplied) = if inputs.is_empty() {
                        (overrides.clone(), None)
                    } else {
                        let (dims, tensors) = onnx2webnn::load_safetensors_inputs(
                            model,
                            &inputs[index],
                            &overrides,
                            &pins,
                        )?;
                        (dims, Some(tensors))
                    };
                    let webnn = local_webnn_path(model, &effective, &pins)?;
                    convert_onnx_with_backend(
                        model,
                        ConvertOptions {
                            free_dim_overrides: effective.clone(),
                            optimize: true,
                            experimental_dynamic_inputs: false,
                            pinned_inputs: pins.clone(),
                            zero_fill_missing_external_data: false,
                            output_path: Some(webnn.clone()),
                        },
                        backend,
                    )?;
                    let result = if let Some(tensors) = supplied {
                        validate_cached_model_with_inputs_on_backend(
                            model, &webnn, tensors, &pins, backend,
                        )?
                    } else {
                        validate_cached_model_with_options_on_backend(
                            model, &webnn, &effective, &pins, backend,
                        )?
                    };
                    Ok(result)
                };
                match run() {
                    Ok(summary) => println!(
                        "PASS {} ({} inputs, {} outputs)",
                        model.display(),
                        summary.input_count,
                        summary.output_count
                    ),
                    Err(error) => {
                        eprintln!("FAIL {}: {error:#}", model.display());
                        failed.push(model.display().to_string());
                    }
                }
            }
            println!(
                "Local validation: {}/{} passed ({:.1}%)",
                total - failed.len(),
                total,
                (total - failed.len()) as f64 * 100.0 / total as f64
            );
            if !failed.is_empty() {
                anyhow::bail!("{} local model(s) failed", failed.len());
            }
        }
    }

    Ok(())
}
