/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end cache export, reload, dispatch, and native ORT comparison.

use std::fs;

use onnx2webnn::onnx::test_models::prelude::*;
use onnx2webnn::{
    cache_onnx_model, convert_onnx, validate_cached_model, validate_cached_model_with_options,
    ConvertOptions,
};
use prost::Message;

#[test]
fn saved_add_graph_reloads_and_matches_native_ort() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("source.onnx");
    let cached_onnx = dir.path().join("cached.onnx");
    let cached_webnn = dir.path().join("cached.webnn");
    let model = model(
        17,
        graph(
            "add",
            vec![f32_input("x", &[1, 2])],
            vec![f32_output("y", &[1, 2])],
            vec![node("Add", "add", &["x", "w"], &["y"], &[])],
            vec![f32_init("w", &[1, 2], &[0.5, 1.0])],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write source model");
    cache_onnx_model(&source, &cached_onnx, false).expect("cache ONNX");
    convert_onnx(
        &cached_onnx,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert and export");
    assert!(cached_webnn.with_extension("safetensors").exists());
    assert_eq!(
        validate_cached_model(&cached_onnx, &cached_webnn)
            .expect("validate")
            .output_count,
        1
    );
}

#[test]
fn packed_uint4_matmul_round_trips_and_matches_native_ort() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("matmul-nbits.onnx");
    let cached_webnn = dir.path().join("matmul-nbits.webnn");
    let (k, n, block_size) = (32i64, 2i64, 32i64);
    let packed_weights = (0..n * block_size / 2)
        .map(|index| ((index * 13 + 7) % 256) as u8)
        .collect::<Vec<_>>();
    let mut matmul = node(
        "MatMulNBits",
        "matmul_nbits",
        &["A", "B", "scales"],
        &["Y"],
        &[
            attr_int("K", k),
            attr_int("N", n),
            attr_int("bits", 4),
            attr_int("block_size", block_size),
        ],
    );
    matmul.domain = "com.microsoft".to_string();
    let mut fixture = model(
        17,
        graph(
            "packed_uint4_matmul",
            vec![f32_input("A", &[1, k])],
            vec![f32_output("Y", &[1, n])],
            vec![matmul],
            vec![
                u8_init("B", &[n, 1, block_size / 2], &packed_weights),
                f32_init("scales", &[n], &[0.05, 0.08]),
            ],
        ),
    );
    fixture
        .opset_import
        .push(onnx2webnn::protos::onnx::OperatorSetIdProto {
            domain: "com.microsoft".to_string(),
            version: 1,
        });
    fs::write(&source, fixture.encode_to_vec()).expect("write packed fixture");

    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert and export packed fixture");

    assert!(cached_webnn.with_extension("safetensors").exists());

    let summary = validate_cached_model(&source, &cached_webnn).expect("validate packed fixture");
    assert_eq!(summary.input_count, 1);
    assert_eq!(summary.output_count, 1);
}

fn integer_dtype_round_trip(label: &str, elem_type: i32) -> Result<(), String> {
    let dir = tempfile::tempdir().map_err(|error| format!("temporary cache: {error}"))?;
    let source = dir.path().join(format!("{label}.onnx"));
    let cached_webnn = dir.path().join(format!("{label}.webnn"));
    let model = model(
        17,
        graph(
            label,
            vec![tensor_input("x", elem_type, &[4])],
            vec![tensor_output("y", elem_type, &[4])],
            vec![node("Identity", "identity", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec())
        .map_err(|error| format!("write {label} dtype model: {error}"))?;
    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .map_err(|error| format!("convert {label}: {error}"))?;
    let summary = validate_cached_model(&source, &cached_webnn)
        .map_err(|error| format!("validate {label}: {error}"))?;
    if summary.input_count != 1 || summary.output_count != 1 {
        return Err(format!(
            "validate {label}: expected one input and output, got {} and {}",
            summary.input_count, summary.output_count
        ));
    }
    Ok(())
}

fn expect_integer_dtype_round_trip(label: &str, elem_type: i32) {
    integer_dtype_round_trip(label, elem_type)
        .unwrap_or_else(|error| panic!("{label} should round-trip exactly: {error}"));
}

// WebNN permits all operand data types for Identity, but only requires float32,
// float16, and int32. CoreML therefore legitimately omits optional int8 graph
// boundary support. Keep executing it as a strict XFAIL instead of hiding it:
// a different error is a regression, and unexpected success is an XPASS.
// https://github.com/webmachinelearning/webnn/blob/main/index.bs#tensor-limits-elementwise-unary-identity
#[cfg(all(target_os = "macos", feature = "coreml"))]
fn expect_coreml_identity_boundary_xfail(label: &str, elem_type: i32) {
    let error = match integer_dtype_round_trip(label, elem_type) {
        Ok(()) => panic!("XPASS: CoreML now supports {label} Identity graph I/O"),
        Err(error) => error,
    };
    let normalized = error.to_ascii_lowercase();
    assert!(
        normalized.contains("coreml") && normalized.contains("identity"),
        "unexpected failure for CoreML {label} Identity graph I/O: {error}"
    );
    eprintln!("XFAIL: CoreML {label} Identity graph I/O: {error}");
}

#[cfg(not(all(target_os = "macos", feature = "coreml")))]
#[test]
fn accepted_int8_dtype_round_trips_exactly() {
    expect_integer_dtype_round_trip(
        "int8",
        onnx2webnn::protos::onnx::TensorProto_DataType::Int8 as i32,
    );
}

#[test]
fn accepted_uint32_dtype_round_trips_exactly() {
    expect_integer_dtype_round_trip(
        "uint32",
        onnx2webnn::protos::onnx::TensorProto_DataType::Uint32 as i32,
    );
}

#[test]
fn accepted_uint64_dtype_round_trips_exactly() {
    expect_integer_dtype_round_trip(
        "uint64",
        onnx2webnn::protos::onnx::TensorProto_DataType::Uint64 as i32,
    );
}

#[cfg(all(target_os = "macos", feature = "coreml"))]
#[test]
fn coreml_xfail_int8_identity_boundary() {
    expect_coreml_identity_boundary_xfail(
        "int8",
        onnx2webnn::protos::onnx::TensorProto_DataType::Int8 as i32,
    );
}

#[test]
fn pinned_input_is_used_by_native_ort_but_not_dispatched() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("pinned.onnx");
    let cached_webnn = dir.path().join("pinned.webnn");
    let model = model(
        17,
        graph(
            "pinned",
            vec![f32_input("x", &[2]), f32_input("scale", &[])],
            vec![f32_output("y", &[2])],
            vec![node("Mul", "mul", &["x", "scale"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write pinned model");
    let pins = std::collections::HashMap::from([("scale".to_string(), 2)]);
    convert_onnx(
        &source,
        ConvertOptions {
            pinned_inputs: pins.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert pinned model");
    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &std::collections::HashMap::new(),
        &pins,
    )
    .expect("validate pinned model");
    assert_eq!(summary.input_count, 1);
    assert_eq!(summary.pinned_input_count, 1);
    assert_eq!(summary.output_count, 1);
}

#[test]
fn pinned_input_name_reused_as_output_uses_the_converted_output_key() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("pinned-output-collision.onnx");
    let cached_webnn = dir.path().join("pinned-output-collision.webnn");
    let model = model(
        17,
        graph(
            "pinned-output-collision",
            vec![f32_input("x", &[2]), f32_input("scale", &[])],
            vec![f32_output("y", &[2]), f32_output("scale", &[])],
            vec![node("Identity", "identity", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write pinned-output collision model");
    let pins = std::collections::HashMap::from([("scale".to_string(), 2)]);
    convert_onnx(
        &source,
        ConvertOptions {
            pinned_inputs: pins.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert pinned-output collision model");

    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &std::collections::HashMap::new(),
        &pins,
    )
    .expect("validate output whose name matches a pinned input");
    assert_eq!(summary.input_count, 1);
    assert_eq!(summary.pinned_input_count, 1);
    assert_eq!(summary.output_count, 2);
}

#[test]
fn pinned_if_branches_validate_with_their_specialized_input_interfaces() {
    use onnx2webnn::protos::onnx::{AttributeProto, GraphProto};

    let branch = |name: &str, input: &str| GraphProto {
        name: name.to_string(),
        node: vec![node(
            "Identity",
            &format!("{name}_identity"),
            &[input],
            &["branch_output"],
            &[],
        )],
        output: vec![f32_output("branch_output", &[2])],
        ..Default::default()
    };
    let mut if_node = node("If", "gate", &["use_cache_branch"], &["y"], &[]);
    if_node.attribute = vec![
        AttributeProto {
            name: "then_branch".to_string(),
            r#type: 5,
            g: Some(branch("then", "then_input")),
            ..Default::default()
        },
        AttributeProto {
            name: "else_branch".to_string(),
            r#type: 5,
            g: Some(branch("else", "else_input")),
            ..Default::default()
        },
    ];
    let model = model(
        17,
        graph(
            "merged",
            vec![
                f32_input("then_input", &[2]),
                f32_input("else_input", &[2]),
                bool_input("use_cache_branch", &[1]),
            ],
            vec![f32_output("y", &[2])],
            vec![if_node],
            vec![],
        ),
    );

    for branch_value in [0, 1] {
        let dir = tempfile::tempdir().expect("temporary cache");
        let source = dir.path().join("merged.onnx");
        let cached_webnn = dir.path().join("merged.webnn");
        fs::write(&source, model.encode_to_vec()).expect("write merged model");
        let pins =
            std::collections::HashMap::from([("use_cache_branch".to_string(), branch_value)]);
        convert_onnx(
            &source,
            ConvertOptions {
                pinned_inputs: pins.clone(),
                output_path: Some(cached_webnn.clone()),
                ..ConvertOptions::default()
            },
        )
        .expect("convert selected branch");

        let summary = validate_cached_model_with_options(
            &source,
            &cached_webnn,
            &std::collections::HashMap::new(),
            &pins,
        )
        .expect("validate selected branch");
        assert_eq!(summary.input_count, 1);
        assert_eq!(summary.output_count, 1);
    }
}

#[test]
fn specialized_cache_accepts_pruned_inputs_and_empty_outputs() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("specialized.onnx");
    let cached_webnn = dir.path().join("specialized.webnn");
    let model = model(
        17,
        graph(
            "specialized",
            vec![
                f32_input("x", &[2]),
                f32_input("dead_branch_input", &[2]),
                bool_input("use_cache_branch", &[]),
            ],
            vec![f32_output("y", &[2]), f32_output("empty_cache", &[0])],
            vec![node("Identity", "identity", &["x"], &["y"], &[])],
            vec![f32_init("empty_cache", &[0], &[])],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write specialized model");
    let pins = std::collections::HashMap::from([("use_cache_branch".to_string(), 0)]);
    convert_onnx(
        &source,
        ConvertOptions {
            pinned_inputs: pins.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert specialized model");

    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &std::collections::HashMap::new(),
        &pins,
    )
    .expect("validate specialized interface");
    assert_eq!(summary.input_count, 1);
    assert_eq!(summary.output_count, 2);
}

#[test]
fn specialized_cache_rejects_an_omitted_nonempty_output() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("source.onnx");
    let cache_source = dir.path().join("cache-source.onnx");
    let cached_webnn = dir.path().join("cached.webnn");
    let source_model = model(
        17,
        graph(
            "source",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2]), f32_output("z", &[2])],
            vec![
                node("Identity", "y", &["x"], &["y"], &[]),
                node("Identity", "z", &["x"], &["z"], &[]),
            ],
            vec![],
        ),
    );
    let cache_model = model(
        17,
        graph(
            "cache",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Identity", "y", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, source_model.encode_to_vec()).expect("write source model");
    fs::write(&cache_source, cache_model.encode_to_vec()).expect("write cache source");
    convert_onnx(
        &cache_source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert cache source");

    let error = validate_cached_model(&source, &cached_webnn)
        .expect_err("non-empty source output must not be silently omitted");
    assert!(error
        .to_string()
        .contains("cached graph omitted non-empty ONNX output z"));
}

#[test]
fn specialized_cache_rejects_extra_inputs_and_outputs() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("source.onnx");
    let extra_input_source = dir.path().join("extra-input.onnx");
    let extra_output_source = dir.path().join("extra-output.onnx");
    let extra_input_webnn = dir.path().join("extra-input.webnn");
    let extra_output_webnn = dir.path().join("extra-output.webnn");
    let source_model = model(
        17,
        graph(
            "source",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Identity", "y", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    let extra_input_model = model(
        17,
        graph(
            "extra-input",
            vec![f32_input("x", &[2]), f32_input("extra", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Add", "y", &["x", "extra"], &["y"], &[])],
            vec![],
        ),
    );
    let extra_output_model = model(
        17,
        graph(
            "extra-output",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2]), f32_output("extra", &[2])],
            vec![
                node("Identity", "y", &["x"], &["y"], &[]),
                node("Identity", "extra", &["x"], &["extra"], &[]),
            ],
            vec![],
        ),
    );
    fs::write(&source, source_model.encode_to_vec()).expect("write source model");
    fs::write(&extra_input_source, extra_input_model.encode_to_vec())
        .expect("write extra-input model");
    fs::write(&extra_output_source, extra_output_model.encode_to_vec())
        .expect("write extra-output model");

    convert_onnx(
        &extra_input_source,
        ConvertOptions {
            output_path: Some(extra_input_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert extra-input model");
    let error = validate_cached_model(&source, &extra_input_webnn)
        .expect_err("cached-only input must be rejected");
    assert!(error
        .to_string()
        .contains("cached graph input extra has no matching unpinned ONNX input"));

    convert_onnx(
        &extra_output_source,
        ConvertOptions {
            output_path: Some(extra_output_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert extra-output model");
    let error = validate_cached_model(&source, &extra_output_webnn)
        .expect_err("cached-only output must be rejected");
    assert!(error
        .to_string()
        .contains("cached graph output extra has no matching ONNX output"));
}

#[test]
fn sanitized_source_input_collisions_are_rejected() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("source.onnx");
    let cache_source = dir.path().join("cache-source.onnx");
    let cached_webnn = dir.path().join("cached.webnn");
    let source_model = model(
        17,
        graph(
            "source",
            vec![f32_input("a/b", &[2]), f32_input("a_b", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Identity", "y", &["a/b"], &["y"], &[])],
            vec![],
        ),
    );
    let cache_model = model(
        17,
        graph(
            "cache",
            vec![f32_input("a_b", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Identity", "y", &["a_b"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, source_model.encode_to_vec()).expect("write source model");
    fs::write(&cache_source, cache_model.encode_to_vec()).expect("write cache source");
    convert_onnx(
        &cache_source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert cache source");

    let error = validate_cached_model(&source, &cached_webnn)
        .expect_err("ambiguous source input mapping must be rejected");
    assert!(error
        .to_string()
        .contains("both map to cached input key a_b"));
}

#[test]
fn external_data_model_round_trips_without_embedding_weights() {
    use onnx2webnn::protos::onnx::StringStringEntryProto;

    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("external.onnx");
    let weights = dir.path().join("weights.bin");
    let cached_webnn = dir.path().join("external.webnn");
    let mut model = model(
        17,
        graph(
            "external",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("Add", "add", &["x", "w"], &["y"], &[])],
            vec![f32_init("w", &[2], &[0.5, 1.0])],
        ),
    );
    let weight = &mut model.graph.as_mut().unwrap().initializer[0];
    weight.float_data.clear();
    weight.data_location = 1;
    weight.external_data = vec![
        StringStringEntryProto {
            key: "location".into(),
            value: "weights.bin".into(),
        },
        StringStringEntryProto {
            key: "offset".into(),
            value: "0".into(),
        },
        StringStringEntryProto {
            key: "length".into(),
            value: "8".into(),
        },
    ];
    fs::write(&source, model.encode_to_vec()).expect("write external-data model");
    fs::write(
        &weights,
        [0.5f32, 1.0f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .expect("write external weights");

    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert external-data model");
    validate_cached_model(&source, &cached_webnn)
        .expect("path-based native ORT resolves external weights");
}

fn validate_semantic_index_input(input_name: &str, length: i64) {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join(format!("{input_name}.onnx"));
    let cached_webnn = dir.path().join(format!("{input_name}.webnn"));
    let model = model(
        17,
        graph(
            input_name,
            vec![i64_input(input_name, &[length])],
            vec![f32_output("y", &[length, 1])],
            vec![node(
                "Gather",
                "gather",
                &["table", input_name],
                &["y"],
                &[],
            )],
            vec![f32_init("table", &[2, 1], &[0.25, 0.75])],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write semantic-input model");
    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert semantic-input model");
    validate_cached_model(&source, &cached_webnn).expect("semantic input stays in range");
}

#[test]
fn token_type_ids_stay_within_a_two_row_embedding() {
    validate_semantic_index_input("token_type_ids", 4);
}

#[test]
fn attention_mask_is_binary_and_valid_for_indexing() {
    validate_semantic_index_input("attention_mask", 65);
}

#[test]
fn zero_element_input_round_trips_without_becoming_a_scalar() {
    use onnx2webnn::protos::onnx::{
        tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeValue,
    };

    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("zero-element-input.onnx");
    let cached_webnn = dir.path().join("zero-element-input.webnn");
    let mut empty_cache = f32_input("past_key", &[1, 2, 1, 4]);
    let Some(TypeValue::TensorType(tensor)) =
        empty_cache.r#type.as_mut().and_then(|ty| ty.value.as_mut())
    else {
        panic!("tensor input");
    };
    tensor.shape.as_mut().unwrap().dim[2].value =
        Some(DimensionValue::DimParam("past_sequence_length".into()));
    let model = model(
        17,
        graph(
            "zero-element-input",
            vec![empty_cache, f32_input("x", &[1])],
            vec![f32_output("y", &[1])],
            vec![node("Identity", "identity", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&source, model.encode_to_vec()).expect("write zero-element-input model");
    let overrides = std::collections::HashMap::from([("past_sequence_length".to_string(), 0)]);
    convert_onnx(
        &source,
        ConvertOptions {
            free_dim_overrides: overrides.clone(),
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert zero-element-input model");
    let summary = validate_cached_model_with_options(
        &source,
        &cached_webnn,
        &overrides,
        &std::collections::HashMap::new(),
    )
    .expect("zero-element input survives native and cached execution");
    assert_eq!(summary.input_count, 2);
    assert_eq!(summary.output_count, 1);
}

#[test]
fn numeric_cast_to_bool_normalizes_nonzero_values() {
    use onnx2webnn::protos::onnx::TensorProto_DataType;

    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("cast-bool.onnx");
    let cached_webnn = dir.path().join("cast-bool.webnn");
    let fixture = model(
        17,
        graph(
            "cast-bool",
            vec![],
            vec![f32_output("y", &[5])],
            vec![
                node(
                    "Cast",
                    "to_bool",
                    &["x"],
                    &["as_bool"],
                    &[attr_int("to", TensorProto_DataType::Bool as i64)],
                ),
                node(
                    "Cast",
                    "to_float",
                    &["as_bool"],
                    &["y"],
                    &[attr_int("to", TensorProto_DataType::Float as i64)],
                ),
            ],
            vec![f32_init("x", &[5], &[0.0, 1.0, 2.0, -1.0, f32::NAN])],
        ),
    );
    fs::write(&source, fixture.encode_to_vec()).expect("write cast-bool fixture");
    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert cast-bool fixture");
    validate_cached_model(&source, &cached_webnn).expect("validate cast-bool fixture");
}

#[test]
fn positive_step_slice_round_trips_with_extent_semantics() {
    let dir = tempfile::tempdir().expect("temporary cache");
    let source = dir.path().join("strided-slice.onnx");
    let cached_webnn = dir.path().join("strided-slice.webnn");
    let fixture = model(
        17,
        graph(
            "strided-slice",
            vec![f32_input("x", &[10])],
            vec![f32_output("y", &[4])],
            vec![node(
                "Slice",
                "slice",
                &["x", "starts", "ends", "axes", "steps"],
                &["y"],
                &[],
            )],
            vec![
                i64_init("starts", &[1], &[1]),
                i64_init("ends", &[1], &[9]),
                i64_init("axes", &[1], &[0]),
                i64_init("steps", &[1], &[2]),
            ],
        ),
    );
    fs::write(&source, fixture.encode_to_vec()).expect("write strided-slice fixture");
    convert_onnx(
        &source,
        ConvertOptions {
            output_path: Some(cached_webnn.clone()),
            ..ConvertOptions::default()
        },
    )
    .expect("convert strided-slice fixture");
    let serialized = fs::read_to_string(&cached_webnn).expect("read serialized graph");
    assert!(serialized.contains("sizes=[8]"));
    assert!(serialized.contains("strides=[2]"));
    validate_cached_model(&source, &cached_webnn).expect("validate strided-slice fixture");
}
