/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Run ONNX Runtime reference inference and compare rustnn dispatch results.

use std::collections::{HashMap, HashSet};

use onnx2webnn::onnx::builder::OnnxBuilder;
use onnx2webnn::protos::onnx::{
    type_proto, ModelProto, TensorProto_DataType, ValueInfoProto,
};
use onnx2webnn::{convert_model_proto, ConvertOptions, OnnxError, ValidatedGraph};
use prost::Message;
use rustnn::graph::OperandDescriptor;
use rustnn::mlcontext::{MLContext, MLTensor, MLTensorDescriptor};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{OnnxInput, TensorData, run_onnx_with_inputs};

/// Expected outcome for operator-level conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectConvertOp {
    Success,
    UnsupportedOp,
}

/// Convert (when supported), execute via rustnn dispatch, and compare against ORT.
///
/// Fixtures are built at the opset declared in `model.opset_import`. The converter
/// itself accepts any `ai.onnx` opset in the supported range (9–26).
pub fn assert_op_matches_ort(
    model: ModelProto,
    expect: ExpectConvertOp,
    test_opset: i64,
) {
    let declared_opset = model
        .opset_import
        .iter()
        .find(|opset| opset.domain.is_empty())
        .map(|opset| opset.version)
        .unwrap_or_default();
    assert_eq!(
        declared_opset, test_opset,
        "fixture opset and test opset should match"
    );
    let result = convert_model_proto(model.clone(), &ConvertOptions::default());
    match expect {
        ExpectConvertOp::UnsupportedOp => match result {
            Err(err) if err.is_unsupported_op() => {}
            Err(err) => panic!("expected UnsupportedOp, got {err}"),
            Ok(_) => panic!("expected UnsupportedOp, got Ok"),
        },
        ExpectConvertOp::Success => {
            let mut validated = result.unwrap_or_else(|err| panic!("expected conversion success, got {err}"));
            let inputs = build_ort_inputs(&model).expect("failed to build ORT inputs");
            let model_bytes = model.encode_to_vec();
            let reference = run_onnx_with_inputs(&model_bytes, None, clone_ort_inputs(&inputs))
                .unwrap_or_else(|err| panic!("ORT reference run failed: {err}"));
            let actual = dispatch_and_collect(&mut validated, &model, &inputs)
                .unwrap_or_else(|err| panic!("rustnn dispatch failed: {err}"));
            compare_outputs(&model, &reference, &actual);
        }
    }
}

fn graph_proto(model: &ModelProto) -> &onnx2webnn::protos::onnx::GraphProto {
    model.graph.as_ref().expect("model graph")
}

fn initializer_names(model: &ModelProto) -> HashSet<String> {
    graph_proto(model)
        .initializer
        .iter()
        .map(|init| init.name.clone())
        .collect()
}

fn feedable_inputs(model: &ModelProto) -> Vec<&ValueInfoProto> {
    let inits = initializer_names(model);
    graph_proto(model)
        .input
        .iter()
        .filter(|vi| !inits.contains(&vi.name))
        .collect()
}

fn tensor_dims(vi: &ValueInfoProto) -> Option<(i32, Vec<usize>)> {
    let ty = vi.r#type.as_ref()?;
    let tensor = match ty.value.as_ref()? {
        type_proto::Value::TensorType(t) => t,
        _ => return None,
    };
    let elem_type = tensor.elem_type;
    let shape = tensor.shape.as_ref()?;
    let dims = shape
        .dim
        .iter()
        .map(|d| {
            use onnx2webnn::protos::onnx::tensor_shape_proto::dimension::Value as Dim;
            match d.value.as_ref() {
                Some(Dim::DimValue(v)) => (*v).max(1) as usize,
                _ => 1,
            }
        })
        .collect();
    Some((elem_type, dims))
}

fn deterministic_float_data(name: &str, len: usize) -> Vec<f32> {
    let seed = name.bytes().map(u64::from).sum::<u64>() % 7;
    (0..len)
        .map(|i| 0.125 * (i as f32 + 1.0) + (seed as f32) * 0.01)
        .collect()
}

fn deterministic_int_data(name: &str, len: usize) -> Vec<i64> {
    let seed = (name.bytes().map(u64::from).sum::<u64>() % 5) as i64;
    (0..len)
        .map(|i| (i as i64 + seed) % 7)
        .collect()
}

fn deterministic_bool_data(len: usize) -> Vec<u8> {
    (0..len).map(|i| u8::from(i % 2 == 0)).collect()
}

fn build_ort_inputs(model: &ModelProto) -> Result<Vec<OnnxInput>, String> {
    let mut inputs = Vec::new();
    for vi in feedable_inputs(model) {
        let (elem_type, shape) = tensor_dims(vi)
            .ok_or_else(|| format!("unsupported input kind for {}", vi.name))?;
        let count = shape.iter().product::<usize>().max(1);
        let data = match elem_type {
            x if x == TensorProto_DataType::Float as i32 => {
                TensorData::Float32(deterministic_float_data(&vi.name, count))
            }
            x if x == TensorProto_DataType::Int32 as i32 => {
                let vals = deterministic_int_data(&vi.name, count);
                TensorData::Int32(vals.into_iter().map(|v| v as i32).collect())
            }
            x if x == TensorProto_DataType::Int64 as i32 => {
                TensorData::Int64(deterministic_int_data(&vi.name, count))
            }
            x if x == TensorProto_DataType::Bool as i32 => {
                TensorData::Uint8(deterministic_bool_data(count))
            }
            x if x == TensorProto_DataType::Uint8 as i32 => {
                TensorData::Uint8((0..count).map(|i| (i % 255) as u8).collect())
            }
            other => return Err(format!("unsupported ORT input dtype {other} for {}", vi.name)),
        };
        inputs.push(OnnxInput {
            name: vi.name.clone(),
            shape,
            data,
        });
    }
    Ok(inputs)
}

fn clone_tensor_data(data: &TensorData) -> TensorData {
    match data {
        TensorData::Float32(v) => TensorData::Float32(v.clone()),
        TensorData::Float16(v) => TensorData::Float16(v.clone()),
        TensorData::Int8(v) => TensorData::Int8(v.clone()),
        TensorData::Uint8(v) => TensorData::Uint8(v.clone()),
        TensorData::Int32(v) => TensorData::Int32(v.clone()),
        TensorData::Uint32(v) => TensorData::Uint32(v.clone()),
        TensorData::Int64(v) => TensorData::Int64(v.clone()),
        TensorData::Uint64(v) => TensorData::Uint64(v.clone()),
    }
}

fn clone_ort_inputs(inputs: &[OnnxInput]) -> Vec<OnnxInput> {
    inputs
        .iter()
        .map(|input| OnnxInput {
            name: input.name.clone(),
            shape: input.shape.clone(),
            data: clone_tensor_data(&input.data),
        })
        .collect()
}

fn operand_descriptor_to_tensor_desc(desc: &OperandDescriptor) -> MLTensorDescriptor {
    let shape = desc
        .static_or_max_shape()
        .into_iter()
        .map(|d| d as u64)
        .collect();
    let data_type = MLOperandDataType::try_from(desc.data_type)
        .expect("operand descriptor should map to WebNN type");
    let mut tensor_desc = MLTensorDescriptor::new(data_type, shape);
    tensor_desc.set_readable(true);
    tensor_desc.set_writable(true);
    tensor_desc
}

fn write_ort_input(context: &mut MLContext, tensor: &MLTensor, input: &OnnxInput) {
    match &input.data {
        TensorData::Float32(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Int32(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Int64(data) => context.write_tensor(tensor, data).unwrap(),
        TensorData::Uint8(data) => context.write_tensor(tensor, data).unwrap(),
        _ => panic!("unsupported tensor write for ORT input"),
    }
}

fn onnx_input_webnn_names(model: &ModelProto) -> HashSet<String> {
    feedable_inputs(model)
        .iter()
        .map(|vi| OnnxBuilder::webnn_id(&vi.name))
        .collect()
}

fn dispatch_and_collect(
    validated: &mut ValidatedGraph,
    model: &ModelProto,
    ort_inputs: &[OnnxInput],
) -> Result<HashMap<String, Vec<f64>>, String> {
    let context = &mut validated.context;
    let graph = &mut validated.graph;
    let input_webnn_names = onnx_input_webnn_names(model);

    let mut owned_inputs: Vec<MLTensor> = Vec::new();
    let mut input_names: Vec<String> = Vec::new();
    for ort_input in ort_inputs {
        let webnn_name = OnnxBuilder::webnn_id(&ort_input.name);
        let desc = graph
            .input_descriptors
            .get(&webnn_name)
            .ok_or_else(|| format!("missing graph input descriptor for {webnn_name}"))?;
        let tensor_desc = operand_descriptor_to_tensor_desc(desc);
        let tensor = context
            .create_tensor(&tensor_desc)
            .map_err(|e| e.to_string())?;
        write_ort_input(context, &tensor, ort_input);
        owned_inputs.push(tensor);
        input_names.push(webnn_name);
    }
    let mut input_bindings: HashMap<&str, &MLTensor> = HashMap::new();
    for (name, tensor) in input_names.iter().zip(owned_inputs.iter()) {
        input_bindings.insert(name.as_str(), tensor);
    }

    let mut owned_outputs: Vec<MLTensor> = Vec::new();
    let mut output_names: Vec<String> = Vec::new();
    let mut output_keys: HashMap<String, String> = HashMap::new();
    for out in &graph_proto(model).output {
        let webnn_key = OnnxBuilder::output_key_for(&out.name, &input_webnn_names);
        let desc = graph
            .output_descriptors
            .get(&webnn_key)
            .ok_or_else(|| format!("missing graph output descriptor for {webnn_key}"))?;
        let tensor_desc = operand_descriptor_to_tensor_desc(desc);
        let tensor = context
            .create_tensor(&tensor_desc)
            .map_err(|e| e.to_string())?;
        output_names.push(webnn_key.clone());
        owned_outputs.push(tensor);
        output_keys.insert(out.name.clone(), webnn_key);
    }
    let mut output_bindings: HashMap<&str, &MLTensor> = HashMap::new();
    for (name, tensor) in output_names.iter().zip(owned_outputs.iter()) {
        output_bindings.insert(name.as_str(), tensor);
    }

    context
        .dispatch(graph, &input_bindings, &output_bindings)
        .map_err(|e| e.to_string())?;

    let mut results = HashMap::new();
    for (onnx_name, webnn_key) in output_keys {
        let desc = graph
            .output_descriptors
            .get(&webnn_key)
            .expect("validated above");
        let tensor = output_bindings
            .get(webnn_key.as_str())
            .expect("output binding");
        let values = read_tensor_as_f64(context, tensor, desc)?;
        results.insert(onnx_name, values);
    }
    Ok(results)
}

fn read_tensor_as_f64(
    context: &mut MLContext,
    tensor: &MLTensor,
    desc: &OperandDescriptor,
) -> Result<Vec<f64>, String> {
    let count = desc.element_count().unwrap_or(1).max(1);
    match desc.data_type {
        rustnn::DataType::Float32 => {
            let mut buf = vec![0.0f32; count];
            context.read_tensor(tensor, &mut buf).map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(f64::from).collect())
        }
        rustnn::DataType::Int32 => {
            let mut buf = vec![0i32; count];
            context.read_tensor(tensor, &mut buf).map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(|v| v as f64).collect())
        }
        rustnn::DataType::Int64 => {
            let mut buf = vec![0i64; count];
            context.read_tensor(tensor, &mut buf).map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(|v| v as f64).collect())
        }
        rustnn::DataType::Uint8 => {
            let mut buf = vec![0u8; count];
            context.read_tensor(tensor, &mut buf).map_err(|e| e.to_string())?;
            Ok(buf.into_iter().map(f64::from).collect())
        }
        other => Err(format!("unsupported output read dtype {other:?}")),
    }
}

fn compare_outputs(
    model: &ModelProto,
    reference: &[rustnn::OnnxOutputWithData],
    actual: &HashMap<String, Vec<f64>>,
) {
    let outputs = &graph_proto(model).output;
    assert_eq!(
        reference.len(),
        outputs.len(),
        "reference output count mismatch"
    );
    for (idx, out) in outputs.iter().enumerate() {
        let ort = &reference[idx];
        let got = actual
            .get(&out.name)
            .unwrap_or_else(|| panic!("missing rustnn output for {}", out.name));
        if let Some(expected_bool) = ort.bool_data.as_ref() {
            assert_same_bool_values(&out.name, expected_bool, got);
            continue;
        }
        let expected = ort
            .float32_data
            .as_ref()
            .map(|data| data.iter().map(|&v| f64::from(v)).collect::<Vec<_>>())
            .or_else(|| ort.int64_data.as_ref().map(|data| data.iter().map(|&v| v as f64).collect()))
            .unwrap_or_else(|| ort.data.clone());
        assert_same_values(&out.name, &expected, got);
    }
}

fn assert_same_bool_values(name: &str, expected: &[bool], actual: &[f64]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "output length mismatch for {name}"
    );
    for (i, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
        let actual_bool = if (*actual - 0.0).abs() <= f64::EPSILON {
            false
        } else if (*actual - 1.0).abs() <= f64::EPSILON {
            true
        } else {
            panic!("output {name}[{i}] expected bool-compatible 0/1, got {actual}");
        };
        assert_eq!(
            *expected, actual_bool,
            "output {name}[{i}] mismatch: ORT={expected}, rustnn={actual_bool}"
        );
    }
}

fn assert_same_values(name: &str, expected: &[f64], actual: &[f64]) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "output length mismatch for {name}"
    );
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        let rounded_e = (e * 1_000_000.0).round() / 1_000_000.0;
        let rounded_a = (a * 1_000_000.0).round() / 1_000_000.0;
        assert!(
            (rounded_e - rounded_a).abs() <= 1e-5,
            "output {name}[{i}] mismatch: ORT={rounded_e}, rustnn={rounded_a}"
        );
    }
}
