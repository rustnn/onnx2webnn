//! Cache reload and deterministic native-ORT validation for exported WebNN graphs.
use crate::onnx::builder::OnnxBuilder;
use crate::onnx::convert::{OnnxError, ValidatedGraph};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    ModelProto, TensorProto_DataType, ValueInfoProto,
};
use half::f16;
use prost::Message;
use rustnn::graph::OperandDescriptor;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLPowerPreference, MLTensor, MLTensorDescriptor,
};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::{load_graph_from_path, run_onnx_path_with_inputs, OnnxInput, TensorData};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValidationSummary {
    pub input_count: usize,
    pub pinned_input_count: usize,
    pub output_count: usize,
}

/// Reload saved WebNN artifacts, dispatch deterministic fixed-shape inputs,
/// and compare results with native ONNX Runtime on the cached ONNX model.
pub fn validate_cached_model(
    onnx_path: impl AsRef<Path>,
    webnn_path: impl AsRef<Path>,
) -> Result<ValidationSummary, OnnxError> {
    validate_cached_model_with_options(onnx_path, webnn_path, &HashMap::new(), &HashMap::new())
}

/// Validate cached artifacts using explicit bindings for symbolic ONNX input dimensions.
pub fn validate_cached_model_with_overrides(
    onnx_path: impl AsRef<Path>,
    webnn_path: impl AsRef<Path>,
    free_dim_overrides: &HashMap<String, u32>,
) -> Result<ValidationSummary, OnnxError> {
    validate_cached_model_with_options(onnx_path, webnn_path, free_dim_overrides, &HashMap::new())
}

/// Validate cached artifacts with symbolic dimension bindings and converter-pinned inputs.
pub fn validate_cached_model_with_options(
    onnx_path: impl AsRef<Path>,
    webnn_path: impl AsRef<Path>,
    free_dim_overrides: &HashMap<String, u32>,
    pinned_inputs: &HashMap<String, i64>,
) -> Result<ValidationSummary, OnnxError> {
    let onnx_bytes = fs::read(onnx_path.as_ref())?;
    let model = ModelProto::decode(onnx_bytes.as_slice())
        .map_err(|e| OnnxError::ProtobufError(e.to_string()))?;
    let inputs = deterministic_inputs(&model, free_dim_overrides, pinned_inputs)?;
    let reference = run_onnx_path_with_inputs(onnx_path.as_ref(), clone_inputs(&inputs))
        .map_err(|e| OnnxError::Validation(format!("native ORT run failed: {e}")))?;
    let graph_info = load_graph_from_path(webnn_path.as_ref())
        .map_err(|e| OnnxError::Validation(format!("failed to reload WebNN cache: {e}")))?;
    let mut context = MLContext::create(&MLContextOptions::new(MLPowerPreference::Default, false))
        .map_err(|e| OnnxError::Validation(format!("MLContext::create failed: {e}")))?;
    let graph = context
        .rustnn_build_graph(graph_info)
        .map_err(|e| OnnxError::Validation(format!("cached graph build failed: {e}")))?;
    let mut validated = ValidatedGraph { context, graph };
    let input_count = validated.graph.input_descriptors.len();
    let actual = dispatch_and_collect(&mut validated, &model, &inputs, pinned_inputs)?;
    compare_outputs(&model, &reference, &actual)?;
    Ok(ValidationSummary {
        input_count,
        pinned_input_count: pinned_inputs.len(),
        output_count: reference.len(),
    })
}

fn graph(model: &ModelProto) -> Result<&crate::protos::onnx::GraphProto, OnnxError> {
    model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::ProtobufError("Missing graph in model".to_string()))
}

fn tensor_dims(
    vi: &ValueInfoProto,
    free_dim_overrides: &HashMap<String, u32>,
) -> Result<(i32, Vec<usize>), OnnxError> {
    let ty = vi
        .r#type
        .as_ref()
        .and_then(|ty| ty.value.as_ref())
        .ok_or_else(|| OnnxError::Validation(format!("missing tensor type for {}", vi.name)))?;
    let tensor = match ty {
        TypeProtoValue::TensorType(tensor) => tensor,
        _ => {
            return Err(OnnxError::Validation(format!(
                "non-tensor input {}",
                vi.name
            )))
        }
    };
    let shape = tensor
        .shape
        .as_ref()
        .ok_or_else(|| OnnxError::Validation(format!("missing shape for {}", vi.name)))?;
    let dims = shape
        .dim
        .iter()
        .map(|dim| match dim.value.as_ref() {
            Some(DimensionValue::DimValue(value)) if *value >= 0 => Ok(*value as usize),
            Some(DimensionValue::DimParam(name)) => free_dim_overrides
                .get(name)
                .map(|value| *value as usize)
                .ok_or_else(|| {
                    OnnxError::Validation(format!(
                        "dynamic input dimension {name} in {}; provide an override for validation",
                        vi.name
                    ))
                }),
            _ => Err(OnnxError::Validation(format!(
                "dynamic input dimension in {}; use fixed shapes for validation",
                vi.name
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tensor.elem_type, dims))
}

fn feedable_inputs(model: &ModelProto) -> Result<Vec<&ValueInfoProto>, OnnxError> {
    let graph = graph(model)?;
    let initializer_names: HashSet<&str> =
        graph.initializer.iter().map(|t| t.name.as_str()).collect();
    Ok(graph
        .input
        .iter()
        .filter(|input| !initializer_names.contains(input.name.as_str()))
        .collect())
}

fn checked_element_count(name: &str, shape: &[usize]) -> Result<usize, OnnxError> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, &dimension| {
        count.checked_mul(dimension).ok_or_else(|| {
            OnnxError::Validation(format!(
                "input {name} shape {shape:?} overflows element count"
            ))
        })
    })
}

fn semantic_input_value(name: &str) -> Option<i64> {
    match name {
        "token_type_ids" => Some(0),
        "attention_mask" => Some(1),
        _ => None,
    }
}

fn repeated_data(
    elem_type: i32,
    count: usize,
    value: i64,
    name: &str,
) -> Result<TensorData, OnnxError> {
    macro_rules! checked {
        ($variant:ident, $type:ty) => {
            TensorData::$variant(vec![
                <$type>::try_from(value).map_err(|_| {
                    OnnxError::Validation(format!("value {value} is out of range for {name}"))
                })?;
                count
            ])
        };
    }
    Ok(match elem_type {
        x if x == TensorProto_DataType::Float as i32 => {
            TensorData::Float32(vec![value as f32; count])
        }
        x if x == TensorProto_DataType::Float16 as i32 => {
            TensorData::Float16(vec![f16::from_f32(value as f32).to_bits(); count])
        }
        x if x == TensorProto_DataType::Int8 as i32 => checked!(Int8, i8),
        x if x == TensorProto_DataType::Uint8 as i32 => checked!(Uint8, u8),
        x if x == TensorProto_DataType::Int32 as i32 => checked!(Int32, i32),
        x if x == TensorProto_DataType::Uint32 as i32 => checked!(Uint32, u32),
        x if x == TensorProto_DataType::Int64 as i32 => TensorData::Int64(vec![value; count]),
        x if x == TensorProto_DataType::Uint64 as i32 => checked!(Uint64, u64),
        x if x == TensorProto_DataType::Bool as i32 && matches!(value, 0 | 1) => {
            TensorData::Uint8(vec![value as u8; count])
        }
        x if x == TensorProto_DataType::Bool as i32 => {
            return Err(OnnxError::Validation(format!(
                "bool input {name} must be 0 or 1, got {value}"
            )))
        }
        other => {
            return Err(OnnxError::Validation(format!(
                "unsupported deterministic input dtype {other} for {name}"
            )))
        }
    })
}

fn deterministic_data(elem_type: i32, count: usize, name: &str) -> Result<TensorData, OnnxError> {
    Ok(match elem_type {
        x if x == TensorProto_DataType::Float as i32 => {
            TensorData::Float32((0..count).map(|i| ((i % 17) as f32 - 8.0) / 16.0).collect())
        }
        x if x == TensorProto_DataType::Float16 as i32 => TensorData::Float16(
            (0..count)
                .map(|i| f16::from_f32(((i % 17) as f32 - 8.0) / 16.0).to_bits())
                .collect(),
        ),
        x if x == TensorProto_DataType::Int8 as i32 => {
            TensorData::Int8((0..count).map(|i| (i % 7) as i8).collect())
        }
        x if x == TensorProto_DataType::Uint8 as i32 => {
            TensorData::Uint8((0..count).map(|i| (i % 7) as u8).collect())
        }
        x if x == TensorProto_DataType::Int32 as i32 => {
            TensorData::Int32((0..count).map(|i| (i % 7) as i32).collect())
        }
        x if x == TensorProto_DataType::Uint32 as i32 => {
            TensorData::Uint32((0..count).map(|i| (i % 7) as u32).collect())
        }
        x if x == TensorProto_DataType::Int64 as i32 => {
            TensorData::Int64((0..count).map(|i| (i % 7) as i64).collect())
        }
        x if x == TensorProto_DataType::Uint64 as i32 => {
            TensorData::Uint64((0..count).map(|i| (i % 7) as u64).collect())
        }
        x if x == TensorProto_DataType::Bool as i32 => {
            TensorData::Uint8((0..count).map(|i| u8::from(i % 2 == 0)).collect())
        }
        other => {
            return Err(OnnxError::Validation(format!(
                "unsupported deterministic input dtype {other} for {name}"
            )))
        }
    })
}

fn input_data(
    elem_type: i32,
    count: usize,
    name: &str,
    pinned_value: Option<i64>,
) -> Result<TensorData, OnnxError> {
    if let Some(value) = pinned_value.or_else(|| semantic_input_value(name)) {
        repeated_data(elem_type, count, value, name)
    } else {
        deterministic_data(elem_type, count, name)
    }
}

fn deterministic_inputs(
    model: &ModelProto,
    free_dim_overrides: &HashMap<String, u32>,
    pinned_inputs: &HashMap<String, i64>,
) -> Result<Vec<OnnxInput>, OnnxError> {
    let feedable = feedable_inputs(model)?;
    let feedable_names: HashSet<&str> = feedable.iter().map(|input| input.name.as_str()).collect();
    for name in pinned_inputs.keys() {
        if !feedable_names.contains(name.as_str()) {
            return Err(OnnxError::Validation(format!(
                "pinned input {name} is not a feedable graph input"
            )));
        }
    }
    feedable
        .into_iter()
        .map(|input| {
            let (elem_type, shape) = tensor_dims(input, free_dim_overrides)?;
            let count = checked_element_count(&input.name, &shape)?;
            let data = input_data(
                elem_type,
                count,
                &input.name,
                pinned_inputs.get(&input.name).copied(),
            )?;
            Ok(OnnxInput {
                name: input.name.clone(),
                shape,
                data,
            })
        })
        .collect()
}

fn clone_inputs(inputs: &[OnnxInput]) -> Vec<OnnxInput> {
    inputs
        .iter()
        .map(|input| OnnxInput {
            name: input.name.clone(),
            shape: input.shape.clone(),
            data: match &input.data {
                TensorData::Float32(v) => TensorData::Float32(v.clone()),
                TensorData::Float16(v) => TensorData::Float16(v.clone()),
                TensorData::Int8(v) => TensorData::Int8(v.clone()),
                TensorData::Uint8(v) => TensorData::Uint8(v.clone()),
                TensorData::Int32(v) => TensorData::Int32(v.clone()),
                TensorData::Uint32(v) => TensorData::Uint32(v.clone()),
                TensorData::Int64(v) => TensorData::Int64(v.clone()),
                TensorData::Uint64(v) => TensorData::Uint64(v.clone()),
            },
        })
        .collect()
}

fn tensor_descriptor(desc: &OperandDescriptor) -> MLTensorDescriptor {
    let data_type = MLOperandDataType::try_from(desc.data_type).expect("WebNN operand type");
    let mut tensor = MLTensorDescriptor::new(
        data_type,
        desc.static_or_max_shape()
            .into_iter()
            .map(u64::from)
            .collect(),
    );
    tensor.set_readable(true);
    tensor.set_writable(true);
    tensor
}

fn write_input(
    context: &mut MLContext,
    tensor: &MLTensor,
    input: &OnnxInput,
) -> Result<(), OnnxError> {
    if checked_element_count(&input.name, &input.shape)? == 0 {
        return Ok(());
    }
    let result = match &input.data {
        TensorData::Float32(data) => context.write_tensor(tensor, data),
        TensorData::Float16(data) => context.write_tensor(tensor, data),
        TensorData::Int8(data) => context.write_tensor(tensor, data),
        TensorData::Uint32(data) => context.write_tensor(tensor, data),
        TensorData::Uint64(data) => context.write_tensor(tensor, data),
        TensorData::Int32(data) => context.write_tensor(tensor, data),
        TensorData::Int64(data) => context.write_tensor(tensor, data),
        TensorData::Uint8(data) => context.write_tensor(tensor, data),
    };
    result.map_err(|e| OnnxError::Validation(format!("failed to write {}: {e}", input.name)))
}

enum CollectedOutput {
    Numeric(Vec<f64>),
    Int64(Vec<i64>),
    Uint64(Vec<u64>),
}

impl CollectedOutput {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(data) => data.len(),
            Self::Int64(data) => data.len(),
            Self::Uint64(data) => data.len(),
        }
    }
}

fn read_output(
    context: &mut MLContext,
    tensor: &MLTensor,
    desc: &OperandDescriptor,
) -> Result<CollectedOutput, OnnxError> {
    let count = desc
        .element_count()
        .ok_or_else(|| OnnxError::Validation("output shape overflows element count".to_string()))?;
    macro_rules! read_numeric {
        ($type:ty) => {{
            let mut data = vec![<$type>::default(); count];
            if count != 0 {
                context
                    .read_tensor(tensor, &mut data)
                    .map_err(|e| OnnxError::Validation(format!("failed to read output: {e}")))?;
            }
            CollectedOutput::Numeric(data.into_iter().map(|v| v as f64).collect())
        }};
    }
    Ok(match desc.data_type {
        rustnn::DataType::Float32 => read_numeric!(f32),
        rustnn::DataType::Float16 => {
            let mut data = vec![0u16; count];
            if count != 0 {
                context
                    .read_tensor(tensor, &mut data)
                    .map_err(|e| OnnxError::Validation(format!("failed to read output: {e}")))?;
            }
            CollectedOutput::Numeric(
                data.into_iter()
                    .map(|v| f64::from(f16::from_bits(v).to_f32()))
                    .collect(),
            )
        }
        rustnn::DataType::Int8 => read_numeric!(i8),
        rustnn::DataType::Int32 => read_numeric!(i32),
        rustnn::DataType::Int64 => {
            let mut data = vec![0i64; count];
            if count != 0 {
                context
                    .read_tensor(tensor, &mut data)
                    .map_err(|e| OnnxError::Validation(format!("failed to read output: {e}")))?;
            }
            CollectedOutput::Int64(data)
        }
        rustnn::DataType::Uint8 => read_numeric!(u8),
        rustnn::DataType::Uint32 => read_numeric!(u32),
        rustnn::DataType::Uint64 => {
            let mut data = vec![0u64; count];
            if count != 0 {
                context
                    .read_tensor(tensor, &mut data)
                    .map_err(|e| OnnxError::Validation(format!("failed to read output: {e}")))?;
            }
            CollectedOutput::Uint64(data)
        }
        other => {
            return Err(OnnxError::Validation(format!(
                "unsupported output dtype {other:?}"
            )))
        }
    })
}

fn dispatch_and_collect(
    validated: &mut ValidatedGraph,
    model: &ModelProto,
    inputs: &[OnnxInput],
    pinned_inputs: &HashMap<String, i64>,
) -> Result<HashMap<String, CollectedOutput>, OnnxError> {
    let graph_proto = graph(model)?;
    let mut source_inputs = HashMap::new();
    for input in inputs
        .iter()
        .filter(|input| !pinned_inputs.contains_key(&input.name))
    {
        let key = OnnxBuilder::webnn_id(&input.name);
        if let Some(previous) = source_inputs.insert(key.clone(), input) {
            return Err(OnnxError::Validation(format!(
                "ONNX inputs {} and {} both map to cached input key {key}",
                previous.name, input.name
            )));
        }
    }

    // Pinning a merged model's branch selector specializes the converted graph. Inputs used only
    // by the other branch are deliberately absent from the cached interface, while native ORT
    // still needs the complete source-model feed above. Drive WebNN from its actual interface and
    // require every retained cached input to map back to the source model.
    let mut input_storage = Vec::new();
    let mut input_keys = Vec::new();
    for (key, desc) in &validated.graph.input_descriptors {
        let input = source_inputs.get(key).ok_or_else(|| {
            OnnxError::Validation(format!(
                "cached graph input {key} has no matching unpinned ONNX input"
            ))
        })?;
        let tensor = validated
            .context
            .create_tensor(&tensor_descriptor(desc))
            .map_err(|e| OnnxError::Validation(format!("failed to create {key}: {e}")))?;
        write_input(&mut validated.context, &tensor, input)?;
        input_keys.push(key.clone());
        input_storage.push(tensor);
    }
    let input_bindings: BTreeMap<&str, &MLTensor> = input_keys
        .iter()
        .zip(input_storage.iter())
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    let mut output_storage = Vec::new();
    let mut output_keys = Vec::new();
    let mut output_map = HashMap::new();
    let cached_input_names: HashSet<String> =
        validated.graph.input_descriptors.keys().cloned().collect();
    let mut source_outputs = HashMap::new();
    for output in &graph_proto.output {
        let key = OnnxBuilder::output_key_for(&output.name, &cached_input_names);
        if let Some(previous) = source_outputs.insert(key.clone(), output.name.as_str()) {
            return Err(OnnxError::Validation(format!(
                "ONNX outputs {previous} and {} both map to cached output key {key}",
                output.name
            )));
        }
    }
    for (key, desc) in &validated.graph.output_descriptors {
        let onnx_name = source_outputs.get(key).ok_or_else(|| {
            OnnxError::Validation(format!(
                "cached graph output {key} has no matching ONNX output"
            ))
        })?;
        let tensor = validated
            .context
            .create_tensor(&tensor_descriptor(desc))
            .map_err(|e| OnnxError::Validation(format!("failed to create {key}: {e}")))?;
        output_keys.push(key.clone());
        output_storage.push(tensor);
        output_map.insert((*onnx_name).to_string(), key.clone());
    }
    let output_bindings: BTreeMap<&str, &MLTensor> = output_keys
        .iter()
        .zip(output_storage.iter())
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    validated
        .context
        .dispatch(&mut validated.graph, &input_bindings, &output_bindings)
        .map_err(|e| OnnxError::Validation(format!("cached graph dispatch failed: {e}")))?;
    output_map
        .into_iter()
        .map(|(onnx_name, key)| {
            let desc = validated
                .graph
                .output_descriptors
                .get(&key)
                .expect("validated above");
            let tensor = output_bindings.get(key.as_str()).expect("bound above");
            read_output(&mut validated.context, tensor, desc).map(|values| (onnx_name, values))
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatComparisonProfile {
    Standard,
    MatMulNBits4,
    MatMulNBits8,
    UnsupportedMatMulNBitsAccuracy(i64),
}

fn float_comparison_profile(model: &ModelProto) -> Result<FloatComparisonProfile, OnnxError> {
    let mut profile = FloatComparisonProfile::Standard;
    for node in graph(model)?
        .node
        .iter()
        .filter(|node| node.op_type == "MatMulNBits")
    {
        let bits = node
            .attribute
            .iter()
            .find(|attr| attr.name == "bits")
            .map(|attr| attr.i)
            .unwrap_or(4);
        let accuracy_level = node
            .attribute
            .iter()
            .find(|attr| attr.name == "accuracy_level")
            .map(|attr| attr.i)
            .unwrap_or(0);
        if accuracy_level != 0 {
            return Ok(FloatComparisonProfile::UnsupportedMatMulNBitsAccuracy(
                accuracy_level,
            ));
        }
        profile = match (profile, bits) {
            (_, 8) => FloatComparisonProfile::MatMulNBits8,
            (FloatComparisonProfile::Standard, 4) => FloatComparisonProfile::MatMulNBits4,
            (current, 4) => current,
            (current, _) => current,
        };
    }
    Ok(profile)
}

fn float_tolerance(elem_type: Option<i32>, expected: f64, profile: FloatComparisonProfile) -> f64 {
    match elem_type {
        Some(x) if x == TensorProto_DataType::Float16 as i32 => 1e-3 + expected.abs() * 1e-2,
        Some(x)
            if x == TensorProto_DataType::Float as i32
                && profile == FloatComparisonProfile::MatMulNBits8 =>
        {
            2e-3 + expected.abs() * 2e-3
        }
        Some(x)
            if x == TensorProto_DataType::Float as i32
                && profile == FloatComparisonProfile::MatMulNBits4 =>
        {
            1e-3 + expected.abs() * 1e-3
        }
        Some(x) if x == TensorProto_DataType::Float as i32 => 1e-5 + expected.abs() * 1e-4,
        _ => 0.0,
    }
}

fn compare_outputs(
    model: &ModelProto,
    reference: &[rustnn::OnnxOutputWithData],
    actual: &HashMap<String, CollectedOutput>,
) -> Result<(), OnnxError> {
    let outputs = &graph(model)?.output;
    let comparison_profile = float_comparison_profile(model)?;
    if let FloatComparisonProfile::UnsupportedMatMulNBitsAccuracy(level) = comparison_profile {
        return Err(OnnxError::Validation(format!(
            "MatMulNBits accuracy_level={level} is unsupported for numerical comparison"
        )));
    }
    if outputs.len() != reference.len() {
        return Err(OnnxError::Validation(
            "native ORT output count mismatch".to_string(),
        ));
    }
    let mut reference_by_name = HashMap::new();
    for expected in reference {
        if reference_by_name
            .insert(expected.name.as_str(), expected)
            .is_some()
        {
            return Err(OnnxError::Validation(format!(
                "native ORT returned duplicate output {}",
                expected.name
            )));
        }
    }
    for output in outputs {
        let expected = reference_by_name.get(output.name.as_str()).ok_or_else(|| {
            OnnxError::Validation(format!(
                "native ORT did not return declared output {}",
                output.name
            ))
        })?;
        let Some(got) = actual.get(&output.name) else {
            if expected.data.is_empty() {
                continue;
            }
            return Err(OnnxError::Validation(format!(
                "cached graph omitted non-empty ONNX output {} ({} elements)",
                output.name,
                expected.data.len()
            )));
        };
        let elem_type = output
            .r#type
            .as_ref()
            .and_then(|ty| ty.value.as_ref())
            .and_then(|value| match value {
                TypeProtoValue::TensorType(tensor) => Some(tensor.elem_type),
                _ => None,
            });
        if expected.data.len() != got.len() {
            return Err(OnnxError::Validation(format!(
                "{} length mismatch: ORT={}, WebNN={}",
                output.name,
                expected.data.len(),
                got.len()
            )));
        }
        match got {
            CollectedOutput::Int64(actual) => {
                let expected = expected.int64_data.as_ref().ok_or_else(|| {
                    OnnxError::Validation(format!(
                        "native ORT did not return typed int64 data for {}",
                        output.name
                    ))
                })?;
                if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
                    return Err(OnnxError::Validation(format!(
                        "{}[{index}] mismatch: ORT={}, WebNN={}",
                        output.name, expected[index], actual[index]
                    )));
                }
            }
            CollectedOutput::Uint64(actual) => {
                let expected = expected.uint64_data.as_ref().ok_or_else(|| {
                    OnnxError::Validation(format!(
                        "native ORT did not return typed uint64 data for {}",
                        output.name
                    ))
                })?;
                if let Some(index) = expected.iter().zip(actual).position(|(a, b)| a != b) {
                    return Err(OnnxError::Validation(format!(
                        "{}[{index}] mismatch: ORT={}, WebNN={}",
                        output.name, expected[index], actual[index]
                    )));
                }
            }
            CollectedOutput::Numeric(actual) => {
                let mut first_mismatch = None;
                let mut failure_count = 0usize;
                let mut max_abs_error = 0.0f64;
                let mut sum_abs_error = 0.0f64;
                let mut sum_squared_error = 0.0f64;
                let mut max_normalized_error = 0.0f64;
                for (index, (expected, actual)) in expected.data.iter().zip(actual).enumerate() {
                    if expected.is_nan() && actual.is_nan() {
                        continue;
                    }
                    let tolerance = float_tolerance(elem_type, *expected, comparison_profile);
                    let abs_error = if expected == actual {
                        0.0
                    } else if !expected.is_finite() || !actual.is_finite() {
                        f64::INFINITY
                    } else {
                        (expected - actual).abs()
                    };
                    max_abs_error = max_abs_error.max(abs_error);
                    sum_abs_error += abs_error;
                    sum_squared_error += abs_error * abs_error;
                    let normalized_error = if tolerance == 0.0 {
                        if abs_error == 0.0 {
                            0.0
                        } else {
                            f64::INFINITY
                        }
                    } else {
                        abs_error / tolerance
                    };
                    max_normalized_error = max_normalized_error.max(normalized_error);
                    if abs_error > tolerance {
                        failure_count += 1;
                        first_mismatch.get_or_insert((index, *expected, *actual, tolerance));
                    }
                }
                if let Some((index, expected, observed, tolerance)) = first_mismatch {
                    let count = actual.len().max(1) as f64;
                    return Err(OnnxError::Validation(format!(
                        "{}[{index}] mismatch: ORT={expected}, WebNN={observed}, tolerance={tolerance}; failures={failure_count}/{}, max_abs_error={max_abs_error}, mean_abs_error={}, rmse={}, max_normalized_error={max_normalized_error}",
                        output.name,
                        actual.len(),
                        sum_abs_error / count,
                        (sum_squared_error / count).sqrt(),
                    )));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comparison_model(op_type: &str) -> ModelProto {
        ModelProto {
            graph: Some(crate::protos::onnx::GraphProto {
                node: vec![crate::protos::onnx::NodeProto {
                    op_type: op_type.to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn matmul_nbits_model(bits: i64, accuracy_level: i64) -> ModelProto {
        let mut model = comparison_model("MatMulNBits");
        let node = &mut model.graph.as_mut().unwrap().node[0];
        node.attribute.push(crate::protos::onnx::AttributeProto {
            name: "bits".to_string(),
            i: bits,
            ..Default::default()
        });
        if accuracy_level != 0 {
            node.attribute.push(crate::protos::onnx::AttributeProto {
                name: "accuracy_level".to_string(),
                i: accuracy_level,
                ..Default::default()
            });
        }
        model
    }

    #[test]
    fn matmul_nbits_models_use_attribute_derived_tolerances() {
        let q4 = matmul_nbits_model(4, 0);
        let q8 = matmul_nbits_model(8, 0);
        let unsupported = matmul_nbits_model(4, 4);
        assert_eq!(
            float_comparison_profile(&q4).unwrap(),
            FloatComparisonProfile::MatMulNBits4
        );
        assert_eq!(
            float_comparison_profile(&q8).unwrap(),
            FloatComparisonProfile::MatMulNBits8
        );
        assert_eq!(
            float_comparison_profile(&unsupported).unwrap(),
            FloatComparisonProfile::UnsupportedMatMulNBitsAccuracy(4)
        );

        let expected = 0.009632587432861328f64;
        let ordinary = float_tolerance(
            Some(TensorProto_DataType::Float as i32),
            expected,
            FloatComparisonProfile::Standard,
        );
        let q4_tolerance = float_tolerance(
            Some(TensorProto_DataType::Float as i32),
            expected,
            FloatComparisonProfile::MatMulNBits4,
        );
        let q8_tolerance = float_tolerance(
            Some(TensorProto_DataType::Float as i32),
            expected,
            FloatComparisonProfile::MatMulNBits8,
        );
        assert!(ordinary < q4_tolerance);
        assert!(q4_tolerance < q8_tolerance);

        let qwen_expected = -0.12115895748138428f64;
        let qwen_observed = -0.11994504928588867f64;
        let qwen_difference = (qwen_expected - qwen_observed).abs();
        assert!(
            qwen_difference
                > float_tolerance(
                    Some(TensorProto_DataType::Float as i32),
                    qwen_expected,
                    FloatComparisonProfile::MatMulNBits4,
                )
        );
        assert!(
            qwen_difference
                <= float_tolerance(
                    Some(TensorProto_DataType::Float as i32),
                    qwen_expected,
                    FloatComparisonProfile::MatMulNBits8,
                )
        );
        assert_eq!(
            float_comparison_profile(&comparison_model("MatMul")).unwrap(),
            FloatComparisonProfile::Standard
        );
    }

    #[test]
    fn element_count_distinguishes_scalars_zero_dimensions_and_overflow() {
        assert_eq!(checked_element_count("scalar", &[]).unwrap(), 1);
        assert_eq!(checked_element_count("empty", &[1, 2, 0, 64]).unwrap(), 0);
        assert_eq!(
            checked_element_count("empty", &[usize::MAX, 2, 0]).unwrap(),
            0
        );
        let TensorData::Float32(empty) =
            input_data(TensorProto_DataType::Float as i32, 0, "past_key", None).unwrap()
        else {
            panic!("expected float32 empty input");
        };
        assert!(empty.is_empty());
        assert!(checked_element_count("huge", &[usize::MAX, 2]).is_err());
    }

    #[test]
    fn semantic_inputs_use_valid_deterministic_values() {
        let TensorData::Int64(token_types) = input_data(
            TensorProto_DataType::Int64 as i32,
            4,
            "token_type_ids",
            None,
        )
        .unwrap() else {
            panic!("expected int64 token types");
        };
        assert_eq!(token_types, vec![0; 4]);

        let TensorData::Int64(mask) = input_data(
            TensorProto_DataType::Int64 as i32,
            65,
            "attention_mask",
            None,
        )
        .unwrap() else {
            panic!("expected int64 attention mask");
        };
        assert_eq!(mask, vec![1; 65]);
        assert_eq!(mask.iter().sum::<i64>() - 1, 64);
    }

    #[test]
    fn generic_pattern_and_pinned_precedence_are_preserved() {
        let TensorData::Int64(generic) =
            input_data(TensorProto_DataType::Int64 as i32, 9, "input_ids", None).unwrap()
        else {
            panic!("expected int64 generic input");
        };
        assert_eq!(generic, vec![0, 1, 2, 3, 4, 5, 6, 0, 1]);

        let TensorData::Int64(pinned) = input_data(
            TensorProto_DataType::Int64 as i32,
            3,
            "attention_mask",
            Some(0),
        )
        .unwrap() else {
            panic!("expected int64 pinned input");
        };
        assert_eq!(pinned, vec![0; 3]);
    }
}
