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

use clap::{Args, Parser, Subcommand};
use rustnn::mlcontext::Backend;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx2webnn::model_validation::manifest::Selection;
use onnx2webnn::model_validation::{run_manifest_validation, RunOptions, WeightMode};
use onnx2webnn::{
    convert_onnx_with_backend, export_onnx, validate_cached_model_with_inputs_on_backend,
    validate_cached_model_with_options_on_backend, ConvertOptions,
};

#[derive(Parser)]
#[command(name = "onnx2webnn")]
#[command(
    about = "Convert ONNX models to WebNN via MLGraphBuilder",
    long_about = None,
    after_help = "Set the log level with RUST_LOG (e.g. RUST_LOG=debug or RUST_LOG=onnx2webnn=trace)."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Args)]
struct CommonArgs {
    #[arg(long, global = true, help = backend_help(), value_parser = onnx2webnn::capabilities::parse_backend)]
    backend: Option<Backend>,
}

fn backend_help() -> String {
    let backends = onnx2webnn::capabilities::backend_summary();
    format!(
        "Execution backend for graph builds; omit for export only; required for validation. {backends}. Runtime libraries and hardware must also be available."
    )
}

#[derive(Subcommand)]
enum Command {
    /// Print the code-backed operator catalog as JSON
    OperatorCatalog {
        #[command(flatten)]
        common: CommonArgs,
    },
    /// Lower ONNX and export to .webnn. Optionally build a backend-specific graph.
    #[command(after_help = r#"Example:
  onnx2webnn convert --model model.onnx --output model.webnn --backend onnx

Dimension overrides and input pins specialize the exported graph.
Run validate separately to supply execution inputs without modifying the graph."#)]
    Convert {
        #[command(flatten)]
        common: CommonArgs,
        /// ONNX model path
        #[arg(long = "model", required = true)]
        model: PathBuf,
        #[command(flatten)]
        settings: ModelSettingsArgs,

        /// Output .webnn file or extensionless directory (created as needed)
        #[arg(long, short)]
        output: Option<PathBuf>,

        /// Zero-fill external tensors whose data file is missing (weight-stripped
        /// skeleton models; graph structure only)
        #[arg(long = "allow-missing-external-data")]
        allow_missing_external_data: bool,

        #[command(flatten)]
        folding: FoldingArgs,

        /// Preserve unresolved symbolic input dims as dynamic metadata (experimental;
        /// with --backend, requires a build with --features dynamic-inputs)
        #[arg(long)]
        experimental_dynamic_inputs: bool,
    },

    /// Numerically validate manifest models against reloaded WebNN artifacts
    ValidateManifest {
        #[command(flatten)]
        common: CommonArgs,
        #[command(flatten)]
        folding: FoldingArgs,
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

        /// Redownload manifest models and external weights from Hugging Face
        #[arg(long)]
        refresh_model_cache: bool,
    },

    /// Validate existing WebNN graphs against ONNX models without conversion
    #[command(after_help = r#"Example:
  onnx2webnn validate --onnx model.onnx --webnn model.webnn --input inputs.safetensors --backend onnx

Omit --webnn to select the cached graph; omit --input for deterministic feeds.
Dimension overrides and input files size execution feeds; they never rewrite the graph.
Pins must match the values used during conversion."#)]
    Validate {
        #[command(flatten)]
        common: CommonArgs,
        /// Reference ONNX model path
        #[arg(long, required = true)]
        onnx: PathBuf,
        #[command(flatten)]
        settings: ModelSettingsArgs,
        /// Complete SafeTensors execution inputs; omit for deterministic feeds
        #[arg(long = "input", short = 'i')]
        input: Option<PathBuf>,
        /// Existing .webnn file; omit for cache lookup
        #[arg(long)]
        webnn: Option<PathBuf>,
    },
}

#[derive(Args)]
struct FoldingArgs {
    /// Enable optional constant-folding passes (disabled by default); required simplifications always run
    #[arg(long)]
    constant_folding: bool,
}

#[derive(Args)]
struct ModelSettingsArgs {
    /// Symbolic dimension binding for a single model, NAME=VALUE (repeatable)
    #[arg(long = "override-dim")]
    override_dims: Vec<String>,
    /// Freeze a graph input for a single model, NAME=VALUE (repeatable)
    #[arg(long = "pin-input")]
    pin_inputs: Vec<String>,
}

struct ModelSettings {
    overrides: HashMap<String, u32>,
    pins: HashMap<String, i64>,
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

fn cached_webnn_path(input: &Path) -> anyhow::Result<PathBuf> {
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
    Ok(onnx2webnn::cache::webnn_cache_dir()
        .map_err(anyhow::Error::msg)?
        .join(format!("{key}.webnn")))
}

fn conversion_path(input: &Path, output: Option<&Path>) -> anyhow::Result<PathBuf> {
    let webnn = match output {
        None => cached_webnn_path(input)?,
        Some(path) => match path.extension() {
            Some(ext) if ext == "webnn" => {
                anyhow::ensure!(
                    !path.is_dir(),
                    "output .webnn path is a directory: {}",
                    path.display()
                );
                path.to_path_buf()
            }
            None => {
                anyhow::ensure!(
                    !path.is_file(),
                    "output directory path is a file: {}",
                    path.display()
                );
                let stem = input
                    .file_stem()
                    .ok_or_else(|| anyhow::anyhow!("input has no file name"))?;
                let mut filename = stem.to_os_string();
                filename.push(".webnn");
                path.join(filename)
            }
            Some(_) => anyhow::bail!(
                "output must be a .webnn file or an extensionless directory: {}",
                path.display()
            ),
        },
    };
    Ok(webnn)
}

fn model_settings(args: ModelSettingsArgs) -> anyhow::Result<ModelSettings> {
    Ok(ModelSettings {
        overrides: parse_overrides(args.override_dims)?,
        pins: parse_pins(args.pin_inputs)?,
    })
}

fn validate_graph(
    model: &Path,
    graph: &Path,
    overrides: &HashMap<String, u32>,
    pins: &HashMap<String, i64>,
    inputs: Option<Vec<rustnn::OnnxInput>>,
    backend: Backend,
) -> anyhow::Result<onnx2webnn::ValidationSummary> {
    let summary = if let Some(inputs) = inputs {
        validate_cached_model_with_inputs_on_backend(model, graph, inputs, pins, backend)
    } else {
        validate_cached_model_with_options_on_backend(model, graph, overrides, pins, backend)
    }?;
    Ok(summary)
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!(
                "Error: {}",
                rustnn::error::format_error_chain(error.as_ref())
            );
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let common = match &cli.cmd {
        Command::OperatorCatalog { common }
        | Command::Convert { common, .. }
        | Command::Validate { common, .. }
        | Command::ValidateManifest { common, .. } => common,
    };
    let backend = common.backend;
    if !matches!(cli.cmd, Command::OperatorCatalog { .. }) {
        std::env::set_var("O2W_OPERATOR_REPORT", "1");
    }
    pretty_env_logger::init();
    let validation_backend = if !matches!(cli.cmd, Command::OperatorCatalog { .. }) {
        let dynamic_inputs = matches!(
            cli.cmd,
            Command::Convert {
                experimental_dynamic_inputs: true,
                ..
            }
        );
        let validation = matches!(
            cli.cmd,
            Command::Validate { .. } | Command::ValidateManifest { .. }
        );
        onnx2webnn::capabilities::preflight(backend, dynamic_inputs, validation)?
    } else {
        None
    };
    match cli.cmd {
        Command::OperatorCatalog { .. } => println!(
            "{}",
            serde_json::to_string_pretty(onnx2webnn::onnx::ops::catalog::OPERATOR_CAPABILITIES)?
        ),
        Command::Convert {
            model,
            settings,
            output,
            allow_missing_external_data,
            folding: FoldingArgs { constant_folding },
            experimental_dynamic_inputs,
            ..
        } => {
            let ModelSettings { overrides, pins } = model_settings(settings)?;
            let graph = conversion_path(&model, output.as_deref())?;
            let options = ConvertOptions {
                free_dim_overrides: overrides,
                optimize: constant_folding,
                experimental_dynamic_inputs,
                pinned_inputs: pins,
                zero_fill_missing_external_data: allow_missing_external_data,
                output_path: Some(graph.clone()),
            };
            if let Some(backend) = backend {
                let built = convert_onnx_with_backend(&model, options, backend)?;
                println!(
                    "✓ {:?} graph build succeeded for {}",
                    built.context.rustnn_backend(),
                    model.display()
                );
            } else {
                export_onnx(&model, options)?;
                println!(
                    "✓ portable WebNN export succeeded for {} (no backend build)",
                    model.display()
                );
            }
            println!("✓ WebNN at {}", graph.display());
            println!("PASS {}", model.display());
        }
        Command::Validate {
            onnx,
            settings,
            input,
            webnn,
            ..
        } => {
            let backend = validation_backend.expect("validation backend established by preflight");
            let ModelSettings { overrides, pins } = model_settings(settings)?;
            let graph = match webnn {
                Some(graph) => graph,
                None => cached_webnn_path(&onnx)?,
            };
            anyhow::ensure!(
                graph.is_file(),
                "WebNN graph not found: {}; run convert first",
                graph.display()
            );
            let (effective, supplied) = if let Some(input) = input {
                let (dims, tensors) =
                    onnx2webnn::load_safetensors_inputs(&onnx, input, &overrides, &pins)?;
                (dims, Some(tensors))
            } else {
                (overrides, None)
            };
            let summary = validate_graph(&onnx, &graph, &effective, &pins, supplied, backend)?;
            println!(
                "PASS {} ({} inputs, {} outputs)",
                onnx.display(),
                summary.input_count,
                summary.output_count
            );
        }
        Command::ValidateManifest {
            folding: FoldingArgs { constant_folding },
            selection,
            weights,
            manifest,
            jobs,
            refresh_model_cache,
            ..
        } => {
            let backend = validation_backend.expect("validation backend established by preflight");
            let selection = Selection::parse(&selection).map_err(anyhow::Error::msg)?;
            let mut options = RunOptions::new(selection, weights).map_err(anyhow::Error::msg)?;
            options.jobs = jobs;
            options.refresh_model_cache = refresh_model_cache;
            options.backend = backend;
            options.optimize = constant_folding;
            if let Some(manifest) = manifest {
                options.manifest = manifest;
            }
            println!("Validating reloaded graphs with {backend:?} backend against native CPU ORT");
            let summary = run_manifest_validation(options).map_err(anyhow::Error::msg)?;
            println!("{summary}");
            anyhow::ensure!(
                !summary.has_failures(),
                "{} of {} model validations failed",
                summary.failed.len(),
                summary.selected
            );
        }
    }
    Ok(())
}
