#![cfg(feature = "env_logging")]

use onnx2webnn::capabilities::{backend_summary, parse_backend, preflight};
use rustnn::mlcontext::Backend;
use std::process::Command;

fn cli_error(backend: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--model", "missing.onnx", "--backend", backend])
        .output()
        .unwrap();
    assert!(!output.status.success());
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn enabled_backend_parsing_does_not_require_runtime_initialization() {
    assert_eq!(parse_backend("onnx").unwrap(), Backend::Onnx);
    for backend in Backend::compiled() {
        assert_eq!(parse_backend(backend.as_ref()).unwrap(), backend);
    }
}

#[test]
fn disabled_backend_is_rejected_before_model_processing() {
    for backend in Backend::compilable() {
        if Backend::compiled().contains(backend) {
            continue;
        }
        let error = cli_error(backend.as_ref());
        assert!(error.contains("compile-time disabled"), "{error}");
        assert!(error.contains("enabled backends: ["), "{error}");
        assert!(error.contains(&backend_summary()), "{error}");
        assert!(!error.contains("No such file"), "{error}");
        let error = preflight(Some(*backend), false, false)
            .unwrap_err()
            .to_string();
        assert!(error.contains("compile-time disabled"), "{error}");
    }
}

#[test]
fn unknown_backend_names_report_known_choices() {
    for name in ["typo", "ort"] {
        let error = cli_error(name);
        assert!(error.contains("not a known backend"), "{error}");
        assert!(error.contains("enabled backends: ["), "{error}");
        assert!(error.contains(&backend_summary()), "{error}");
    }
}

#[test]
fn help_reports_compile_time_support_without_runtime_initialization() {
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("enabled backends: ["), "{help}");
    assert!(help.contains(&backend_summary()), "{help}");
}

#[test]
fn help_includes_every_upstream_backend() {
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for backend in Backend::compilable() {
        assert!(
            help.contains(backend.as_ref()),
            "missing {backend:?}: {help}"
        );
    }
}

#[test]
fn runtime_initialization_error_retains_the_library_path() {
    let dir = tempfile::tempdir().unwrap();
    let missing_library = dir.path().join("missing-onnxruntime");
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--model", "missing.onnx", "--backend", "onnx"])
        .env("ORT_DYLIB_PATH", &missing_library)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("missing-onnxruntime"), "{stderr}");
    assert!(
        stderr.contains("Error: onnx context creation failed\n  No backend"),
        "{stderr}"
    );
    assert!(
        stderr.contains("\n    Failed to create context\n      onnx runtime initialization failed"),
        "{stderr}"
    );
    assert_eq!(stderr.matches("missing-onnxruntime").count(), 1, "{stderr}");
    assert!(!stderr.contains("RustNN does not expose"), "{stderr}");
}
