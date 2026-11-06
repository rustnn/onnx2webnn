//! API conversion regressions for supported and rejected ONNX nodes.

use onnx2webnn::onnx::test_models::prelude::*;
use onnx2webnn::protos::onnx::TensorProto_DataType;
use onnx2webnn::{convert_model_proto, ConvertOptions};

#[test]
fn unsupported_node_preserves_operator_and_node_identity() {
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
    let error = convert_model_proto(fixture, &ConvertOptions::default())
        .err()
        .expect("unsupported node must fail conversion");
    let ops = error
        .unsupported_ops()
        .expect("unsupported operator entries");
    assert_eq!(ops.len(), 1);
    assert_eq!(ops[0].op, "Acos");
    assert_eq!(ops[0].node, "bad_acos");
}

#[test]
fn supported_direct_node_converts() {
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
    convert_model_proto(fixture, &ConvertOptions::default()).expect("convert fixture");
}

#[test]
fn conditional_cast_converts() {
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
    convert_model_proto(fixture, &ConvertOptions::default()).expect("convert fixture");
}

#[test]
fn matmul_integer_and_identity_convert() {
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
    convert_model_proto(fixture, &ConvertOptions::default()).expect("convert fixture");
}

#[test]
fn unsupported_form_preserves_node_and_rejection_reason() {
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
    let error = convert_model_proto(fixture, &ConvertOptions::default())
        .err()
        .expect("unsupported pooling form must fail conversion");
    let message = error.to_string();
    assert!(message.contains("bad_p"), "{message}");
    assert!(message.contains("p=3"), "{message}");
}
