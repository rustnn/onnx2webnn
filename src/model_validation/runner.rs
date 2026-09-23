/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::full_model::cache_full_model;
use super::manifest::{load_manifest_from, Entry, Selection};
use crate::{convert_onnx, validate_cached_model_with_options, ConvertOptions};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

type ModelCell = Arc<OnceLock<Result<PathBuf, String>>>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WeightMode {
    #[default]
    Real,
}

impl FromStr for WeightMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "real" => Ok(Self::Real),
            _ => Err(format!("invalid weight mode {value:?}; expected real")),
        }
    }
}

impl fmt::Display for WeightMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Real => "real",
        })
    }
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub selection: Selection,
    pub weights: WeightMode,
    pub manifest: PathBuf,
    pub jobs: usize,
    pub webnn_cache: PathBuf,
}

impl RunOptions {
    pub fn new(selection: Selection, weights: WeightMode) -> Result<Self, String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        Ok(Self {
            selection,
            weights,
            manifest: std::env::var_os("O2W_MANIFEST")
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| root.join("tests/models/manifest.json")),
            jobs: 1,
            webnn_cache: crate::cache::webnn_cache_dir()?,
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunSummary {
    pub selected: usize,
    pub passed: usize,
    pub succeeded: Vec<String>,
    pub failed: Vec<ModelFailure>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelFailure {
    pub label: String,
    pub error: String,
}

impl RunSummary {
    pub fn pass_percentage(&self) -> f64 {
        if self.selected == 0 {
            0.0
        } else {
            self.passed as f64 * 100.0 / self.selected as f64
        }
    }

    pub fn has_failures(&self) -> bool {
        !self.failed.is_empty()
    }
}

impl fmt::Display for RunSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Model validation summary")?;
        writeln!(f, "Succeeded ({}):", self.succeeded.len())?;
        if self.succeeded.is_empty() {
            writeln!(f, "  (none)")?;
        } else {
            for label in &self.succeeded {
                writeln!(f, "  PASS {label}")?;
            }
        }
        writeln!(f, "Failed ({}):", self.failed.len())?;
        if self.failed.is_empty() {
            writeln!(f, "  (none)")?;
        } else {
            for failure in &self.failed {
                writeln!(f, "  FAIL {}: {}", failure.label, failure.error)?;
            }
        }
        write!(
            f,
            "Overall: {}/{} passed ({:.1}%)",
            self.passed,
            self.selected,
            self.pass_percentage()
        )
    }
}

pub fn run_manifest_validation(options: RunOptions) -> Result<RunSummary, String> {
    if options.jobs == 0 {
        return Err("validation jobs must be at least 1".to_string());
    }
    let entries = load_manifest_from(&options.manifest)?;
    let selected: Vec<_> = entries
        .into_iter()
        .enumerate()
        .filter(|(index, entry)| options.selection.includes(*index, entry))
        .collect();
    if selected.is_empty() {
        return Err("validation selection matched no models".to_string());
    }
    let selected_count = selected.len();
    let (heavy, light): (Vec<_>, Vec<_>) = selected.into_iter().partition(|(_, entry)| entry.heavy);
    let sweep = Sweep {
        options,
        models: Mutex::new(HashMap::new()),
        results: Mutex::new(Vec::new()),
    };
    sweep.run(light, sweep.options.jobs);
    sweep.run(heavy, 1);
    let mut results = sweep.results.into_inner().unwrap();
    results.sort_unstable_by_key(|result| result.index);
    let mut succeeded = Vec::new();
    let mut failed = Vec::new();
    for result in results {
        if let Some(error) = result.error {
            failed.push(ModelFailure {
                label: result.label,
                error,
            });
        } else {
            succeeded.push(result.label);
        }
    }
    let passed = succeeded.len();
    Ok(RunSummary {
        selected: selected_count,
        passed,
        succeeded,
        failed,
    })
}

struct ModelResult {
    index: usize,
    label: String,
    error: Option<String>,
}

struct Sweep {
    options: RunOptions,
    models: Mutex<HashMap<String, ModelCell>>,
    results: Mutex<Vec<ModelResult>>,
}

impl Sweep {
    fn model(&self, entry: &Entry) -> Result<PathBuf, String> {
        let key = format!("{}:{}", self.options.weights, entry.source_key());
        let cell = self.models.lock().unwrap().entry(key).or_default().clone();
        cell.get_or_init(|| cache_full_model(entry)).clone()
    }

    fn validate(&self, index: usize, entry: &Entry) {
        let label = entry.label(index);
        match self.validate_inner(entry) {
            Ok((inputs, pins, outputs)) => {
                eprintln!(
                    "ok   {label}\n     {inputs} inputs + {pins} pinned, {outputs} outputs ({})",
                    self.options.weights
                );
                self.results.lock().unwrap().push(ModelResult {
                    index,
                    label,
                    error: None,
                });
            }
            Err(error) => {
                eprintln!("FAIL {label}\n     {error}");
                self.results.lock().unwrap().push(ModelResult {
                    index,
                    label,
                    error: Some(error),
                });
            }
        }
    }

    fn validate_inner(&self, entry: &Entry) -> Result<(usize, usize, usize), String> {
        let onnx_path = self
            .model(entry)
            .map_err(|e| format!("model preparation: {e}"))?;
        let webnn_path = self.options.webnn_cache.join(format!(
            "{}-{}.webnn",
            entry.cache_key(),
            self.options.weights
        ));
        std::fs::create_dir_all(webnn_path.parent().expect("cache parent"))
            .map_err(|e| format!("export: {e}"))?;
        convert_onnx(
            &onnx_path,
            ConvertOptions {
                free_dim_overrides: entry.override_dims.clone(),
                optimize: true,
                experimental_dynamic_inputs: false,
                pinned_inputs: entry.pin_inputs.clone(),
                zero_fill_missing_external_data: false,
                output_path: Some(webnn_path.clone()),
            },
        )
        .map_err(|e| format!("conversion/export: {e}"))?;
        if !webnn_path.exists() || !webnn_path.with_extension("safetensors").exists() {
            return Err("export did not produce .webnn and .safetensors".to_string());
        }
        let summary = validate_cached_model_with_options(
            &onnx_path,
            &webnn_path,
            &entry.override_dims,
            &entry.pin_inputs,
        )
        .map_err(|e| classify(&e.to_string()))?;
        Ok((
            summary.input_count,
            summary.pinned_input_count,
            summary.output_count,
        ))
    }

    fn run(&self, entries: Vec<(usize, Entry)>, workers: usize) {
        let queue = Mutex::new(entries);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let Some((index, entry)) = queue.lock().unwrap().pop() else {
                        break;
                    };
                    self.validate(index, &entry);
                });
            }
        });
    }
}

fn classify(error: &str) -> String {
    let stage = if error.contains("native ORT") {
        "native ORT"
    } else if error.contains("reload WebNN") || error.contains("graph build") {
        "reload"
    } else if error.contains("dispatch") || error.contains("write") || error.contains("read output")
    {
        "dispatch"
    } else if error.contains("mismatch") || error.contains("did not produce") {
        "comparison"
    } else {
        "validation"
    };
    format!("{stage}: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_weight_modes() {
        assert_eq!("real".parse(), Ok(WeightMode::Real));
        assert!("generated".parse::<WeightMode>().is_err());
        assert!("random".parse::<WeightMode>().is_err());
    }
    #[test]
    fn real_is_default() {
        assert_eq!(WeightMode::default(), WeightMode::Real);
    }

    #[test]
    fn summary_lists_all_results_and_percentage() {
        let summary = RunSummary {
            selected: 3,
            passed: 2,
            succeeded: vec!["#0 first".into(), "#2 third".into()],
            failed: vec![ModelFailure {
                label: "#1 second".into(),
                error: "comparison: mismatch".into(),
            }],
        };
        let report = summary.to_string();
        assert!(report.contains("Succeeded (2):\n  PASS #0 first\n  PASS #2 third"));
        assert!(report.contains("Failed (1):\n  FAIL #1 second: comparison: mismatch"));
        assert!(report.contains("Overall: 2/3 passed (66.7%)"));
        assert!(summary.has_failures());
    }
}
