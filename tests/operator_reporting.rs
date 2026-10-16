//! CLI capability diagnostics for supported and rejected ONNX nodes.

use onnx2webnn::onnx::test_models::prelude::*;
use onnx2webnn::protos::onnx::TensorProto_DataType;
use prost::Message;
use std::fs;
use std::process::Command;

#[test]
fn unsupported_node_reports_identity_and_complete_assessment() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unsupported.onnx");
    let fixture = model(
        17,
        graph(
            "unsupported",
            vec![f32_input("x", &[1])],
            vec![f32_output("y", &[1])],
            vec![node("Acos", "bad_acos", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&path, fixture.encode_to_vec()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--input", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("unsupported: 1 nodes (100.0%)"), "{stdout}");
    assert!(stdout.contains("ai.onnx::Acos [bad_acos]"), "{stdout}");
    assert!(stdout.contains("rejected: ai.onnx::Acos"), "{stdout}");
}

#[test]
fn supported_direct_node_reports_full_coverage() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("identity.onnx");
    let fixture = model(
        17,
        graph(
            "identity",
            vec![f32_input("x", &[1])],
            vec![f32_output("y", &[1])],
            vec![node("Identity", "id", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&path, fixture.encode_to_vec()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--input", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("fully: 1 nodes (100.0%)"), "{stdout}");
    assert!(stdout.contains("0 unique ops (0.0%)") || stdout.contains("1 unique ops (100.0%)"));
}

#[test]
fn conditional_cast_path_reports_decomposition() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cast_bool.onnx");
    let fixture = model(
        17,
        graph(
            "cast_bool",
            vec![f32_input("x", &[1])],
            vec![bool_output("y", &[1])],
            vec![node(
                "Cast",
                "bool_cast",
                &["x"],
                &["y"],
                &[attr_int("to", TensorProto_DataType::Bool as i64)],
            )],
            vec![],
        ),
    );
    fs::write(&path, fixture.encode_to_vec()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--input", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("partially: 1 nodes (100.0%)"), "{stdout}");
}

#[test]
fn performance_limited_reports_total_and_partial_node_percentages() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("matmul_integer.onnx");
    let fixture = model(
        17,
        graph(
            "matmul_integer",
            vec![u8_input("a", &[1, 2])],
            vec![i32_output("y", &[1, 1])],
            vec![
                node("MatMulInteger", "matmul", &["a", "b"], &["product"], &[]),
                node("Identity", "identity", &["product"], &["y"], &[]),
            ],
            vec![u8_init("b", &[2, 1], &[3, 4])],
        ),
    );
    fs::write(&path, fixture.encode_to_vec()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--input", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("partially: 1 nodes (50.0%)"), "{stdout}");
    assert!(
        stdout.contains("\n\nOperator support (post-fold/specialization):"),
        "{stdout}"
    );
    assert!(
        stdout.contains(
            "performance-limited (within partially): 1 nodes (50.0% of total, 100.0% of partial)"
        ),
        "{stdout}"
    );
    assert!(stdout.contains("    - ai.onnx::MatMulInteger:"), "{stdout}");
    assert!(stdout.contains("\n\n✓"), "{stdout}");
}

#[test]
fn unsupported_form_reports_node_and_rejection_reason() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lp_pool.onnx");
    let fixture = model(
        17,
        graph(
            "lp_pool",
            vec![f32_input("x", &[1, 1, 4, 4])],
            vec![f32_output("y", &[1, 1, 2, 2])],
            vec![node(
                "LpPool",
                "bad_p",
                &["x"],
                &["y"],
                &[attr_int("p", 3), attr_ints("kernel_shape", &[2, 2])],
            )],
            vec![],
        ),
    );
    fs::write(&path, fixture.encode_to_vec()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_onnx2webnn"))
        .args(["convert", "--input", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("rejected: ai.onnx::LpPool [bad_p]"),
        "{stdout}"
    );
    assert!(stdout.contains("p=3"), "{stdout}");
}
