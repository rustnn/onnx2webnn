//! SafeTensors input and local-model API regressions.

use onnx2webnn::onnx::test_models::prelude::*;
use onnx2webnn::protos::onnx::{tensor_shape_proto::dimension::Value, TensorProto_DataType};
use onnx2webnn::{load_safetensors_inputs, validate_cached_model_with_inputs_on_backend};
use prost::Message;
use rustnn::TensorData;
use safetensors::tensor::{Dtype, TensorView};
use std::collections::HashMap;
use std::fs;

fn identity_model(path: &std::path::Path, dtype: i32, shape: &[i64]) {
    let fixture = model(
        17,
        graph(
            "local_identity",
            vec![tensor_input("x", dtype, shape)],
            vec![tensor_output("y", dtype, shape)],
            vec![node("Identity", "id", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(path, fixture.encode_to_vec()).unwrap();
}

fn inputs_file(path: &std::path::Path, names: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
    let views: Vec<_> = names
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                *name,
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    fs::write(path, safetensors::tensor::serialize(views, None).unwrap()).unwrap();
}

#[test]
fn complete_inputs_validate_names_dtypes_shapes_and_symbols() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("symbolic.onnx");
    let mut fixture = model(
        17,
        graph(
            "symbolic",
            vec![f32_input("x", &[1, 2])],
            vec![f32_output("y", &[1, 2])],
            vec![node("Identity", "id", &["x"], &["y"], &[])],
            vec![],
        ),
    );
    let input_shape = fixture.graph.as_mut().unwrap().input[0]
        .r#type
        .as_mut()
        .unwrap();
    if let Some(onnx2webnn::protos::onnx::type_proto::Value::TensorType(tensor)) =
        &mut input_shape.value
    {
        tensor.shape.as_mut().unwrap().dim[1].value = Some(Value::DimParam("sequence".into()));
    }
    fs::write(&model_path, fixture.encode_to_vec()).unwrap();
    let feeds = dir.path().join("feeds.safetensors");
    let data: Vec<u8> = [1.0f32, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    inputs_file(&feeds, &[("x", Dtype::F32, vec![1, 2], data)]);
    let (dims, loaded) =
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(dims["sequence"], 2);
    assert_eq!(loaded[0].shape, vec![1, 2]);
    assert!(load_safetensors_inputs(
        &model_path,
        &feeds,
        &HashMap::from([("sequence".into(), 3)]),
        &HashMap::new()
    )
    .err()
    .unwrap()
    .to_string()
    .contains("conflicts"));
    inputs_file(&feeds, &[("wrong", Dtype::F32, vec![1, 2], vec![0; 8])]);
    assert!(
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new())
            .err()
            .unwrap()
            .to_string()
            .contains("missing")
    );
    inputs_file(&feeds, &[("x", Dtype::I32, vec![1, 2], vec![0; 8])]);
    assert!(
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new())
            .err()
            .unwrap()
            .to_string()
            .contains("dtype")
    );
    inputs_file(&feeds, &[("x", Dtype::F32, vec![2, 1], vec![0; 8])]);
    assert!(
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new())
            .err()
            .unwrap()
            .to_string()
            .contains("dimension")
    );
    inputs_file(
        &feeds,
        &[
            ("x", Dtype::F32, vec![1, 2], vec![0; 8]),
            ("extra", Dtype::U8, vec![1], vec![0]),
        ],
    );
    assert!(
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new())
            .err()
            .unwrap()
            .to_string()
            .contains("extra")
    );
    fs::write(&feeds, b"not a SafeTensors file").unwrap();
    assert!(
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new())
            .err()
            .unwrap()
            .to_string()
            .contains("invalid SafeTensors")
    );
}

#[test]
fn explicit_feeds_can_supply_a_quantization_scale() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("feedable-scale.onnx");
    let fixture = model(
        13,
        graph(
            "feedable_scale",
            vec![f32_input("x", &[3]), f32_input("scale", &[])],
            vec![u8_output("y", &[3])],
            vec![node("QuantizeLinear", "q", &["x", "scale"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&model_path, fixture.encode_to_vec()).unwrap();
    let feeds = dir.path().join("feeds.safetensors");
    let values: Vec<u8> = [0.0f32, 0.25, 0.5]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    inputs_file(
        &feeds,
        &[
            ("x", Dtype::F32, vec![3], values),
            ("scale", Dtype::F32, vec![], 0.25f32.to_le_bytes().to_vec()),
        ],
    );
    let (_, loaded) =
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new()).unwrap();
    assert_eq!(loaded.len(), 2);
    let scale = loaded.iter().find(|input| input.name == "scale").unwrap();
    assert!(matches!(&scale.data, TensorData::Float32(values) if values == &[0.25]));
}

#[test]
fn scalar_and_zero_element_inputs_load() {
    let dir = tempfile::tempdir().unwrap();
    for (shape, storage_shape, bytes) in [
        (vec![], vec![], 1.0f32.to_le_bytes().to_vec()),
        (vec![0], vec![0], vec![]),
    ] {
        let model_path = dir.path().join("shape.onnx");
        let feeds = dir.path().join("feeds.safetensors");
        identity_model(&model_path, TensorProto_DataType::Float as i32, &shape);
        inputs_file(&feeds, &[("x", Dtype::F32, storage_shape, bytes)]);
        let (_, tensors) =
            load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(
            tensors[0].shape,
            shape.iter().map(|v| *v as usize).collect::<Vec<_>>()
        );
    }
}

#[test]
fn accepted_input_dtypes_and_pins_are_checked() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (
            TensorProto_DataType::Float as i32,
            Dtype::F32,
            1.5f32.to_le_bytes().to_vec(),
        ),
        (
            TensorProto_DataType::Float16 as i32,
            Dtype::F16,
            vec![0x00, 0x3c],
        ),
        (TensorProto_DataType::Int8 as i32, Dtype::I8, vec![0xfe]),
        (TensorProto_DataType::Uint8 as i32, Dtype::U8, vec![250]),
        (
            TensorProto_DataType::Int32 as i32,
            Dtype::I32,
            (-9i32).to_le_bytes().to_vec(),
        ),
        (
            TensorProto_DataType::Uint32 as i32,
            Dtype::U32,
            9u32.to_le_bytes().to_vec(),
        ),
        (
            TensorProto_DataType::Int64 as i32,
            Dtype::I64,
            (-9i64).to_le_bytes().to_vec(),
        ),
        (
            TensorProto_DataType::Uint64 as i32,
            Dtype::U64,
            9u64.to_le_bytes().to_vec(),
        ),
        (TensorProto_DataType::Bool as i32, Dtype::BOOL, vec![1]),
    ];
    for (index, (onnx_dtype, storage, bytes)) in cases.into_iter().enumerate() {
        let model_path = dir.path().join(format!("dtype{index}.onnx"));
        let feeds = dir.path().join(format!("dtype{index}.safetensors"));
        identity_model(&model_path, onnx_dtype, &[1]);
        inputs_file(&feeds, &[("x", storage, vec![1], bytes)]);
        let (_, loaded) =
            load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new()).unwrap();
        assert_eq!(loaded[0].shape, vec![1]);
        let correct = match onnx_dtype {
            x if x == TensorProto_DataType::Float as i32 => {
                matches!(&loaded[0].data, TensorData::Float32(_))
            }
            x if x == TensorProto_DataType::Float16 as i32 => {
                matches!(&loaded[0].data, TensorData::Float16(_))
            }
            x if x == TensorProto_DataType::Int8 as i32 => {
                matches!(&loaded[0].data, TensorData::Int8(_))
            }
            x if x == TensorProto_DataType::Uint8 as i32
                || x == TensorProto_DataType::Bool as i32 =>
            {
                matches!(&loaded[0].data, TensorData::Uint8(_))
            }
            x if x == TensorProto_DataType::Int32 as i32 => {
                matches!(&loaded[0].data, TensorData::Int32(_))
            }
            x if x == TensorProto_DataType::Uint32 as i32 => {
                matches!(&loaded[0].data, TensorData::Uint32(_))
            }
            x if x == TensorProto_DataType::Int64 as i32 => {
                matches!(&loaded[0].data, TensorData::Int64(_))
            }
            x if x == TensorProto_DataType::Uint64 as i32 => {
                matches!(&loaded[0].data, TensorData::Uint64(_))
            }
            _ => false,
        };
        assert!(correct, "dtype {onnx_dtype}");
    }
    let model_path = dir.path().join("pinned.onnx");
    let fixture = model(
        17,
        graph(
            "pinned",
            vec![i64_input("use_cache_branch", &[])],
            vec![i64_output("y", &[])],
            vec![node("Identity", "id", &["use_cache_branch"], &["y"], &[])],
            vec![],
        ),
    );
    fs::write(&model_path, fixture.encode_to_vec()).unwrap();
    let feeds = dir.path().join("empty.safetensors");
    fs::write(
        &feeds,
        safetensors::tensor::serialize(Vec::<(&str, TensorView<'_>)>::new(), None).unwrap(),
    )
    .unwrap();
    let (_, loaded) = load_safetensors_inputs(
        &model_path,
        &feeds,
        &HashMap::new(),
        &HashMap::from([("use_cache_branch".into(), 1)]),
    )
    .unwrap();
    assert!(matches!(&loaded[0].data, TensorData::Int64(values) if values == &[1]));
}

#[test]
fn local_models_use_supplied_inputs() {
    let dir = tempfile::tempdir().unwrap();
    for index in 0..2 {
        let model_path = dir.path().join(format!("model{index}.onnx"));
        let feeds = dir.path().join(format!("feeds{index}.safetensors"));
        let mut fixture = model(
            17,
            graph(
                "configured",
                vec![f32_input("x", &[1]), i64_input("flag", &[1])],
                vec![f32_output("y", &[1])],
                vec![
                    node("Cast", "cast", &["flag"], &["f"], &[attr_int("to", 1)]),
                    node("Add", "add", &["x", "f"], &["y"], &[]),
                ],
                vec![],
            ),
        );
        let graph = fixture.graph.as_mut().unwrap();
        for value in graph.input.iter_mut().chain(graph.output.iter_mut()) {
            if value.name == "flag" {
                continue;
            }
            if let Some(onnx2webnn::protos::onnx::type_proto::Value::TensorType(tensor)) =
                &mut value.r#type.as_mut().unwrap().value
            {
                tensor.shape.as_mut().unwrap().dim[0].value = Some(Value::DimParam("n".into()));
            }
        }
        fs::write(&model_path, fixture.encode_to_vec()).unwrap();
        let length = index + 1;
        let webnn = dir.path().join(format!("model{index}.webnn"));
        onnx2webnn::export_onnx(
            &model_path,
            onnx2webnn::ConvertOptions {
                output_path: Some(webnn.clone()),
                free_dim_overrides: HashMap::from([("n".into(), length as u32)]),
                pinned_inputs: HashMap::from([("flag".into(), i64::from(index == 1))]),
                ..Default::default()
            },
        )
        .unwrap();
        inputs_file(
            &feeds,
            &[(
                "x",
                Dtype::F32,
                vec![length],
                (0..length)
                    .flat_map(|_| (index as f32 + 0.25).to_le_bytes())
                    .collect(),
            )],
        );
        let pins = HashMap::from([("flag".into(), i64::from(index == 1))]);
        let (dims, inputs) = load_safetensors_inputs(
            &model_path,
            &feeds,
            &HashMap::from([("n".into(), length as u32)]),
            &pins,
        )
        .unwrap();
        assert_eq!(dims["n"], length as u32);
        let summary = validate_cached_model_with_inputs_on_backend(
            &model_path,
            &webnn,
            inputs,
            &pins,
            rustnn::mlcontext::Backend::Onnx,
        )
        .expect("validate supplied inputs");
        assert_eq!(summary.input_count, 1);
        assert_eq!(summary.pinned_input_count, 1);
        assert_eq!(summary.output_count, 1);
    }
}

#[test]
fn numerical_disagreement_fails_validation() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = dir.path().join("cumprod.onnx");
    let feeds = dir.path().join("negative.safetensors");
    let fixture = model(
        26,
        graph(
            "cumprod",
            vec![f32_input("x", &[2])],
            vec![f32_output("y", &[2])],
            vec![node("CumProd", "cumprod", &["x", "axis"], &["y"], &[])],
            vec![i64_init("axis", &[], &[0])],
        ),
    );
    fs::write(&model_path, fixture.encode_to_vec()).unwrap();
    let bytes: Vec<u8> = [-1.0f32, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    inputs_file(&feeds, &[("x", Dtype::F32, vec![2], bytes)]);
    let webnn = dir.path().join("cumprod.webnn");
    onnx2webnn::export_onnx(
        &model_path,
        onnx2webnn::ConvertOptions {
            output_path: Some(webnn.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let (_, inputs) =
        load_safetensors_inputs(&model_path, &feeds, &HashMap::new(), &HashMap::new()).unwrap();
    let error = validate_cached_model_with_inputs_on_backend(
        &model_path,
        &webnn,
        inputs,
        &HashMap::new(),
        rustnn::mlcontext::Backend::Onnx,
    )
    .expect_err("numerical disagreement must fail validation");
    assert!(error.to_string().contains("mismatch"), "{error}");
}
