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
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx2webnn::model_validation::manifest::Selection;
use onnx2webnn::model_validation::{run_manifest_validation, RunOptions, WeightMode};
use onnx2webnn::ConvertOptions;
use onnx2webnn::{cache_onnx_model, convert_onnx, validate_cached_model_with_overrides};

#[derive(Parser)]
#[command(name = "onnx2webnn")]
#[command(
    about = "Convert ONNX models to WebNN via MLGraphBuilder (ORT validation)",
    long_about = None
)]
struct Cli {
    /// Enable debug output
    #[arg(long, global = true)]
    debug: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Lower ONNX to MLGraphBuilder and validate with rustnn ORT (CPU build)
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

        /// Save the converted graph and self-contained ONNX model to the OS cache directory
        #[arg(long)]
        output: bool,

        /// Reload cached WebNN artifacts, dispatch deterministic inputs, and compare with native ORT
        #[arg(long, requires = "output")]
        validate: bool,

        /// Validate existing cache artifacts only; does not read or convert the source ONNX.
        #[arg(long, conflicts_with_all = ["validate", "output"])]
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
    if cli.debug {
        onnx2webnn::debug::enable();
    }

    match cli.cmd {
        Command::Convert {
            input,
            override_dims,
            override_dims_file,
            pin_inputs,
            allow_missing_external_data,
            optimize,
            experimental_dynamic_inputs,
            output,
            validate,
            validate_cached,
        } => {
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
                        "cached validation requires {} and {}; run with --output first",
                        onnx_cache.display(),
                        webnn_cache.display()
                    ));
                }
                let summary = validate_cached_model_with_overrides(
                    &onnx_cache,
                    &webnn_cache,
                    &validation_overrides,
                )
                .map_err(|e| anyhow::anyhow!("{e}"))?;
                println!(
                    "✓ cached deterministic validation passed ({} inputs, {} outputs)",
                    summary.input_count, summary.output_count
                );
                return Ok(());
            }

            let cache_paths = output.then(|| cache_paths(input_path)).transpose()?;
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

            let _validated = convert_onnx(input_path.to_str().unwrap(), options)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            // stdout (not stderr): PowerShell treats native stderr as NativeCommandError
            println!("✓ ORT graph build succeeded for {}", input);
            if let Some((onnx_cache, webnn_cache)) = &cache_paths {
                println!("✓ cached ONNX at {}", onnx_cache.display());
                println!("✓ cached WebNN at {}", webnn_cache.display());
                if validate {
                    let summary = validate_cached_model_with_overrides(
                        onnx_cache,
                        webnn_cache,
                        &validation_overrides,
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
            let selection = Selection::parse(&selection).map_err(anyhow::Error::msg)?;
            let mut options = RunOptions::new(selection, weights).map_err(anyhow::Error::msg)?;
            options.jobs = jobs;
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
    }

    Ok(())
}
