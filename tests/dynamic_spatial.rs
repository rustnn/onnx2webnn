//! Dynamic pixel-unshuffle shapes must survive export and cache reload.
use onnx2webnn::protos::onnx::{tensor_shape_proto::dimension::Value, type_proto};
use onnx2webnn::test_models::prelude::*;
use onnx2webnn::{export_model_proto, validate_cached_model_with_overrides, ConvertOptions};
use prost::Message;
use std::{collections::HashMap, fs};

#[test]
fn pixel_unshuffle_preserves_dynamic_metadata_and_matches_ort_when_specialized() {
    let mut input = f32_input("input", &[1, 3, 1, 1]);
    let Some(type_proto::Value::TensorType(tensor)) = input.r#type.as_mut().unwrap().value.as_mut()
    else {
        panic!("tensor input")
    };
    for (axis, name) in [(2, "h"), (3, "w")] {
        tensor.shape.as_mut().unwrap().dim[axis].value = Some(Value::DimParam(name.into()));
    }
    let mut output = f32_output("output", &[1, 2, 1, 1]);
    let Some(type_proto::Value::TensorType(tensor)) =
        output.r#type.as_mut().unwrap().value.as_mut()
    else {
        panic!("tensor output")
    };
    for (axis, name) in [(2, "h"), (3, "w")] {
        tensor.shape.as_mut().unwrap().dim[axis].value = Some(Value::DimParam(name.into()));
    }
    let fixture = model(
        17,
        graph(
            "pixel_unshuffle",
            vec![input],
            vec![output],
            vec![
                node(
                    "Unsqueeze",
                    "unsqueeze",
                    &["input", "axis"],
                    &["unsqueezed"],
                    &[],
                ),
                node(
                    "Reshape",
                    "split_h",
                    &["unsqueezed", "shape_h"],
                    &["split_h"],
                    &[],
                ),
                node(
                    "Reshape",
                    "split_w",
                    &["split_h", "shape_w"],
                    &["split_w"],
                    &[],
                ),
                node(
                    "Transpose",
                    "transpose",
                    &["split_w"],
                    &["transposed"],
                    &[attr_ints("perm", &[0, 1, 3, 5, 2, 4])],
                ),
                node(
                    "Reshape",
                    "merge",
                    &["transposed", "shape_merge"],
                    &["merged"],
                    &[],
                ),
                node(
                    "Squeeze",
                    "squeeze",
                    &["merged", "squeeze_axes"],
                    &["unshuffled"],
                    &[],
                ),
                node("Conv", "conv", &["unshuffled", "weight"], &["conv"], &[]),
                node(
                    "Constant",
                    "scales",
                    &[],
                    &["scales"],
                    &[attr_tensor(
                        "value",
                        f32_init("", &[4], &[1.0, 1.0, 2.0, 2.0]),
                    )],
                ),
                node(
                    "Resize",
                    "resize",
                    &["conv", "", "scales"],
                    &["output"],
                    &[attr_string("mode", "linear")],
                ),
            ],
            vec![
                i64_init("axis", &[1], &[3]),
                i64_init("shape_h", &[5], &[0, 0, -1, 2, 0]),
                i64_init("shape_w", &[6], &[0, 0, 0, 0, -1, 2]),
                i64_init("shape_merge", &[6], &[0, -1, 1, 1, 0, 0]),
                i64_init("squeeze_axes", &[2], &[2, 3]),
                f32_init(
                    "weight",
                    &[2, 12, 1, 1],
                    &(0..24).map(|i| i as f32 / 24.0).collect::<Vec<_>>(),
                ),
            ],
        ),
    );
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.onnx");
    let webnn = dir.path().join("graph.webnn");
    fs::write(&source, fixture.encode_to_vec()).unwrap();
    export_model_proto(
        fixture.clone(),
        &ConvertOptions {
            experimental_dynamic_inputs: true,
            output_path: Some(webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("export dynamic graph");
    let saved = rustnn::load_graph_from_path(&webnn).expect("reload dynamic metadata");
    let unshuffled = saved
        .operands
        .iter()
        .find(|operand| operand.name.as_deref() == Some("unshuffled"))
        .expect("unshuffled operand");
    assert_eq!(
        unshuffled.descriptor.shape,
        vec![
            rustnn::graph::Dimension::Static(1),
            rustnn::graph::Dimension::Static(12),
            rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "h / 2".into(),
                max_size: 32_767
            }),
            rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                name: "w / 2".into(),
                max_size: 32_767
            }),
        ]
    );
    for (h, w) in [(4, 6), (8, 4)] {
        let overrides = HashMap::from([("h".into(), h), ("w".into(), w)]);
        export_model_proto(
            fixture.clone(),
            &ConvertOptions {
                free_dim_overrides: overrides.clone(),
                output_path: Some(webnn.clone()),
                ..ConvertOptions::default()
            },
        )
        .expect("export concrete image size");
        validate_cached_model_with_overrides(&source, &webnn, &overrides)
            .expect("concrete export matches native ORT");
    }
}
