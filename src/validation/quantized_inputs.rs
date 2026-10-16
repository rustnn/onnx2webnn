//! Scale-aware fallback feeds only where the ONNX input has an explicit quantization boundary.

use crate::onnx::builder::tensor_proto_to_bytes;
use crate::onnx::convert::OnnxError;
use crate::protos::onnx::{
    tensor_proto::DataLocation, ModelProto, NodeProto, TensorProto, TensorProto_DataType,
};
use half::f16;
use rustnn::TensorData;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};

fn invalid(name: &str, reason: impl std::fmt::Display) -> OnnxError {
    OnnxError::Validation(format!(
        "cannot synthesize quantized input {name}: {reason}; supply explicit inputs"
    ))
}

/// Return a feed only for a single direct Q/DQ consumer with constant scalar parameters.
pub(super) fn quantized_input_data(
    model: &ModelProto,
    model_path: &Path,
    name: &str,
    elem_type: i32,
    count: usize,
) -> Result<Option<TensorData>, OnnxError> {
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| invalid(name, "missing graph"))?;
    let mut boundary: Option<&NodeProto> = None;
    let mut other_use = false;
    for node in &graph.node {
        let quantizer = (node.domain.is_empty() || node.domain == "ai.onnx")
            && matches!(node.op_type.as_str(), "QuantizeLinear" | "DequantizeLinear");
        if quantizer && node.input.iter().skip(1).any(|input| input == name) {
            return Err(invalid(name, "scale/zero-point values are model-specific"));
        }
        if node.input.first().is_some_and(|input| input == name) && quantizer {
            if boundary.replace(node).is_some() {
                return Err(invalid(name, "multiple quantization boundaries"));
            }
        } else if node.input.iter().any(|input| input == name) {
            other_use = true;
        }
    }
    let Some(node) = boundary else {
        return Ok(None);
    };
    if other_use {
        return Err(invalid(name, "both quantized and other consumers"));
    }
    let scale_name = node
        .input
        .get(1)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| invalid(name, "missing scale"))?;
    let scale = scalar_initializer(graph.initializer.as_slice(), model_path, scale_name, name)?;
    let scale = match scale.data_type {
        x if x == TensorProto_DataType::Float as i32 => f32::from_le_bytes(
            scale
                .bytes
                .as_slice()
                .try_into()
                .map_err(|_| invalid(name, "invalid float32 scale"))?,
        ),
        x if x == TensorProto_DataType::Float16 as i32 => f16::from_bits(u16::from_le_bytes(
            scale
                .bytes
                .as_slice()
                .try_into()
                .map_err(|_| invalid(name, "invalid float16 scale"))?,
        ))
        .to_f32(),
        _ => return Err(invalid(name, "scale must be float32 or float16")),
    };
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid(name, "scale must be finite and positive"));
    }

    let zero_point = node
        .input
        .get(2)
        .filter(|value| !value.is_empty())
        .map(|value| scalar_initializer(graph.initializer.as_slice(), model_path, value, name))
        .transpose()?;
    let quant_dtype = if node.op_type == "DequantizeLinear" {
        elem_type
    } else {
        if !matches!(elem_type, x if x == TensorProto_DataType::Float as i32 || x == TensorProto_DataType::Float16 as i32)
        {
            return Err(invalid(name, "QuantizeLinear expects a float input"));
        }
        if node
            .attribute
            .iter()
            .any(|attr| attr.name == "output_dtype" && attr.i != 0)
        {
            return Err(invalid(
                name,
                "explicit output_dtype requires supplied inputs",
            ));
        }
        zero_point
            .as_ref()
            .map_or(TensorProto_DataType::Uint8 as i32, |value| value.data_type)
    };
    let (min, max) = match quant_dtype {
        x if x == TensorProto_DataType::Uint8 as i32 => (0i16, 255i16),
        x if x == TensorProto_DataType::Int8 as i32 => (-128i16, 127i16),
        _ => return Err(invalid(name, "only int8/uint8 boundaries are supported")),
    };
    let zero = match zero_point {
        Some(value) if value.data_type == quant_dtype && value.bytes.len() == 1 => {
            if quant_dtype == TensorProto_DataType::Int8 as i32 {
                i16::from(value.bytes[0] as i8)
            } else {
                i16::from(value.bytes[0])
            }
        }
        Some(_) => {
            return Err(invalid(
                name,
                "zero point dtype differs from quantized dtype",
            ))
        }
        None => 0,
    };
    let low = (-8i16).max(min - zero);
    let high = 8i16.min(max - zero);
    if node.op_type == "DequantizeLinear" {
        let values = (0..count).map(|i| zero + bin_offset(i, count, low, high));
        return Ok(Some(if quant_dtype == TensorProto_DataType::Int8 as i32 {
            TensorData::Int8(values.map(|value| value as i8).collect())
        } else {
            TensorData::Uint8(values.map(|value| value as u8).collect())
        }));
    }
    let values = (0..count).map(|i| bin_offset(i, count, low, high) as f32 * scale);
    let data = if elem_type == TensorProto_DataType::Float16 as i32 {
        let values: Vec<_> = values.map(f16::from_f32).collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(name, "scale produces non-finite float16 values"));
        }
        if count > 1
            && values
                .iter()
                .all(|value| value.to_bits() == values[0].to_bits())
        {
            return Err(invalid(
                name,
                "float16 rounding collapses every quantization bin",
            ));
        }
        TensorData::Float16(values.into_iter().map(f16::to_bits).collect())
    } else {
        let values: Vec<_> = values.collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(invalid(name, "scale produces non-finite float32 values"));
        }
        if count > 1
            && values
                .iter()
                .all(|value| value.to_bits() == values[0].to_bits())
        {
            return Err(invalid(
                name,
                "float32 rounding collapses every quantization bin",
            ));
        }
        TensorData::Float32(values)
    };
    Ok(Some(data))
}

fn bin_offset(index: usize, count: usize, low: i16, high: i16) -> i16 {
    if count == 1 {
        return 0i16.clamp(low, high);
    }
    let span = usize::try_from(high - low).expect("8-bit quantization range");
    let step = if count <= span + 1 {
        (index * span + (count - 1) / 2) / (count - 1)
    } else {
        index % (span + 1)
    };
    low + step as i16
}

struct ScalarInitializer {
    data_type: i32,
    bytes: Vec<u8>,
}

fn scalar_initializer(
    tensors: &[TensorProto],
    model_path: &Path,
    tensor_name: &str,
    input_name: &str,
) -> Result<ScalarInitializer, OnnxError> {
    let mut matches = tensors.iter().filter(|tensor| tensor.name == tensor_name);
    let tensor = matches.next().ok_or_else(|| {
        invalid(
            input_name,
            format!("{tensor_name} is not a fixed initializer"),
        )
    })?;
    if matches.next().is_some() || !tensor.dims.iter().all(|&dim| dim == 1) {
        return Err(invalid(
            input_name,
            format!("{tensor_name} is not a unique scalar"),
        ));
    }
    let width = match tensor.data_type {
        x if x == TensorProto_DataType::Float as i32 => 4,
        x if x == TensorProto_DataType::Float16 as i32 => 2,
        x if x == TensorProto_DataType::Int8 as i32 || x == TensorProto_DataType::Uint8 as i32 => 1,
        _ => {
            return Err(invalid(
                input_name,
                format!("unsupported {tensor_name} dtype"),
            ))
        }
    };
    let bytes = if tensor.data_location == DataLocation::External as i32 {
        read_external_scalar(tensor, model_path, width, input_name)?
    } else {
        tensor_proto_to_bytes(tensor).map_err(|error| invalid(input_name, error))?
    };
    if bytes.len() != width {
        return Err(invalid(
            input_name,
            format!("{tensor_name} has {} bytes, expected {width}", bytes.len()),
        ));
    }
    Ok(ScalarInitializer {
        data_type: tensor.data_type,
        bytes,
    })
}

fn read_external_scalar(
    tensor: &TensorProto,
    model_path: &Path,
    width: usize,
    input_name: &str,
) -> Result<Vec<u8>, OnnxError> {
    let field = |key| {
        tensor
            .external_data
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.value.as_str())
    };
    let location =
        field("location").ok_or_else(|| invalid(input_name, "external scalar has no location"))?;
    let relative = Path::new(location);
    if !relative
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
    {
        return Err(invalid(
            input_name,
            "external scalar path must stay inside the model directory",
        ));
    }
    let parse = |key| -> Result<Option<usize>, OnnxError> {
        field(key)
            .map(|value| {
                value
                    .parse::<usize>()
                    .map_err(|_| invalid(input_name, format!("invalid {key}")))
            })
            .transpose()
    };
    let offset = parse("offset")?.unwrap_or(0);
    if parse("length")?.is_some_and(|length| length != width) {
        return Err(invalid(
            input_name,
            "external scalar length differs from its dtype",
        ));
    }
    let path = model_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(relative);
    let mut file = File::open(&path).map_err(|error| invalid(input_name, error))?;
    let end = offset
        .checked_add(width)
        .ok_or_else(|| invalid(input_name, "external scalar offset overflows"))?;
    if file
        .metadata()
        .map_err(|error| invalid(input_name, error))?
        .len()
        < end as u64
    {
        return Err(invalid(
            input_name,
            "external scalar range exceeds its sidecar",
        ));
    }
    let mut bytes = vec![0; width];
    file.seek(SeekFrom::Start(offset as u64))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(|error| invalid(input_name, error))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{bin_offset, quantized_input_data};
    use crate::onnx::test_models::prelude::*;
    use crate::protos::onnx::{
        tensor_proto::DataLocation, StringStringEntryProto, TensorProto, TensorProto_DataType,
    };
    use rustnn::TensorData;
    use std::fs;
    use std::path::Path;

    #[test]
    fn quantize_input_spans_bins_around_its_zero_point() {
        assert_eq!(
            (0..3).map(|i| bin_offset(i, 3, -8, 8)).collect::<Vec<_>>(),
            vec![-8, 0, 8]
        );
        let fixture = model(
            13,
            graph(
                "quantized_input",
                vec![f32_input("x", &[17])],
                vec![u8_output("y", &[17])],
                vec![node(
                    "QuantizeLinear",
                    "q",
                    &["x", "scale", "zp"],
                    &["y"],
                    &[],
                )],
                vec![f32_init("scale", &[], &[0.25]), u8_init("zp", &[], &[128])],
            ),
        );
        let Some(TensorData::Float32(values)) = quantized_input_data(
            &fixture,
            Path::new("unused.onnx"),
            "x",
            TensorProto_DataType::Float as i32,
            17,
        )
        .unwrap() else {
            panic!("expected scale-aware float input");
        };
        assert_eq!(values[0], -2.0);
        assert_eq!(values[8], 0.0);
        assert_eq!(values[16], 2.0);
        assert_eq!(values.iter().filter(|&&value| value == 0.0).count(), 1);
    }

    #[test]
    fn invalid_scale_is_not_used_as_a_fallback_range() {
        let fixture = model(
            13,
            graph(
                "invalid_scale",
                vec![f32_input("x", &[3])],
                vec![u8_output("y", &[3])],
                vec![node("QuantizeLinear", "q", &["x", "scale"], &["y"], &[])],
                vec![f32_init("scale", &[], &[0.0])],
            ),
        );
        assert!(quantized_input_data(
            &fixture,
            Path::new("unused.onnx"),
            "x",
            TensorProto_DataType::Float as i32,
            3
        )
        .err()
        .expect("zero scale is invalid")
        .to_string()
        .contains("finite and positive"));
    }

    #[test]
    fn dequantize_input_tracks_the_quantized_zero_point() {
        let fixture = model(
            13,
            graph(
                "dequantized_input",
                vec![u8_input("x", &[17])],
                vec![f32_output("y", &[17])],
                vec![node(
                    "DequantizeLinear",
                    "dq",
                    &["x", "scale", "zp"],
                    &["y"],
                    &[],
                )],
                vec![f32_init("scale", &[], &[0.25]), u8_init("zp", &[], &[128])],
            ),
        );
        let Some(TensorData::Uint8(values)) = quantized_input_data(
            &fixture,
            Path::new("unused.onnx"),
            "x",
            TensorProto_DataType::Uint8 as i32,
            17,
        )
        .unwrap() else {
            panic!("expected zero-point-aware integer input");
        };
        assert_eq!(values, (120..=136).collect::<Vec<_>>());
    }

    #[test]
    fn float16_input_uses_signed_int8_quantization_bins() {
        let fixture = model(
            13,
            graph(
                "signed_quantized_input",
                vec![f16_input("x", &[3])],
                vec![i8_output("y", &[3])],
                vec![node(
                    "QuantizeLinear",
                    "q",
                    &["x", "scale", "zp"],
                    &["y"],
                    &[],
                )],
                vec![
                    f16_init("scale", &[], &[half::f16::from_f32(0.25).to_bits()]),
                    i8_init("zp", &[], &[-3]),
                ],
            ),
        );
        let Some(TensorData::Float16(values)) = quantized_input_data(
            &fixture,
            Path::new("unused.onnx"),
            "x",
            TensorProto_DataType::Float16 as i32,
            3,
        )
        .unwrap() else {
            panic!("expected Float16 scale-aware input");
        };
        let decoded: Vec<_> = values
            .into_iter()
            .map(|bits| half::f16::from_bits(bits).to_f32())
            .collect();
        assert_eq!(decoded, vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn ambiguous_or_feedable_quantization_parameters_require_explicit_inputs() {
        let mut fixture = model(
            13,
            graph(
                "quantized_input",
                vec![f32_input("x", &[4]), f32_input("scale", &[])],
                vec![u8_output("y", &[4])],
                vec![node("QuantizeLinear", "q", &["x", "scale"], &["y"], &[])],
                vec![],
            ),
        );
        for name in ["x", "scale"] {
            assert!(quantized_input_data(
                &fixture,
                Path::new("unused.onnx"),
                name,
                TensorProto_DataType::Float as i32,
                4,
            )
            .err()
            .expect("quantization parameter needs supplied inputs")
            .to_string()
            .contains("supply explicit inputs"));
        }
        fixture
            .graph
            .as_mut()
            .unwrap()
            .initializer
            .push(f32_init("scale", &[2], &[0.25, 0.5]));
        assert!(quantized_input_data(
            &fixture,
            Path::new("unused.onnx"),
            "x",
            TensorProto_DataType::Float as i32,
            4,
        )
        .err()
        .expect("per-axis scale needs supplied inputs")
        .to_string()
        .contains("not a unique scalar"));
    }

    #[test]
    fn external_scalar_scale_reads_only_its_declared_range() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar = dir.path().join("weights.bin");
        fs::write(&sidecar, [9, 9, 0, 0, 0, 63, 9]).unwrap();
        let external = TensorProto {
            name: "scale".into(),
            data_type: TensorProto_DataType::Float as i32,
            data_location: DataLocation::External as i32,
            external_data: [
                ("location", "weights.bin"),
                ("offset", "2"),
                ("length", "4"),
            ]
            .into_iter()
            .map(|(key, value)| StringStringEntryProto {
                key: key.into(),
                value: value.into(),
            })
            .collect(),
            ..Default::default()
        };
        let fixture = model(
            13,
            graph(
                "external_scale",
                vec![f32_input("x", &[9])],
                vec![u8_output("y", &[9])],
                vec![node("QuantizeLinear", "q", &["x", "scale"], &["y"], &[])],
                vec![external],
            ),
        );
        let Some(TensorData::Float32(values)) = quantized_input_data(
            &fixture,
            &dir.path().join("model.onnx"),
            "x",
            TensorProto_DataType::Float as i32,
            9,
        )
        .unwrap() else {
            panic!("expected scale-aware input");
        };
        assert_eq!(values, (0..9).map(|i| i as f32 * 0.5).collect::<Vec<_>>());
    }
}
