/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Build ONNX [`ModelProto`] graphs programmatically for converter tests.

use crate::protos::onnx::{
    attribute_proto, type_proto, AttributeProto, GraphProto, ModelProto, NodeProto,
    OperatorSetIdProto, TensorProto, TensorProto_DataType, TensorShapeProto, ValueInfoProto,
};

fn dim_value(value: i64) -> crate::protos::onnx::tensor_shape_proto::Dimension {
    crate::protos::onnx::tensor_shape_proto::Dimension {
        value: Some(
            crate::protos::onnx::tensor_shape_proto::dimension::Value::DimValue(value),
        ),
        denotation: String::new(),
    }
}

fn tensor_shape(dims: &[i64]) -> TensorShapeProto {
    TensorShapeProto {
        dim: dims.iter().copied().map(dim_value).collect(),
    }
}

fn tensor_type_proto(elem_type: i32, shape: &[i64]) -> crate::protos::onnx::TypeProto {
    crate::protos::onnx::TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type,
            shape: Some(tensor_shape(shape)),
        })),
        denotation: String::new(),
    }
}

fn sequence_of_tensor(elem_type: i32, shape: &[i64]) -> crate::protos::onnx::TypeProto {
    crate::protos::onnx::TypeProto {
        value: Some(type_proto::Value::SequenceType(Box::new(
            type_proto::Sequence {
                elem_type: Some(Box::new(tensor_type_proto(elem_type, shape))),
            },
        ))),
        denotation: String::new(),
    }
}

fn optional_of_tensor(elem_type: i32, shape: &[i64]) -> crate::protos::onnx::TypeProto {
    crate::protos::onnx::TypeProto {
        value: Some(type_proto::Value::OptionalType(Box::new(
            type_proto::Optional {
                elem_type: Some(Box::new(tensor_type_proto(elem_type, shape))),
            },
        ))),
        denotation: String::new(),
    }
}

fn value_info(name: &str, ty: crate::protos::onnx::TypeProto) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(ty),
        ..Default::default()
    }
}

pub fn tensor_input(name: &str, elem_type: i32, shape: &[i64]) -> ValueInfoProto {
    value_info(name, tensor_type_proto(elem_type, shape))
}

pub fn tensor_output(name: &str, elem_type: i32, shape: &[i64]) -> ValueInfoProto {
    value_info(name, tensor_type_proto(elem_type, shape))
}

pub fn f32_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::Float as i32, shape)
}

pub fn f32_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::Float as i32, shape)
}

pub fn i32_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::Int32 as i32, shape)
}

pub fn i32_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::Int32 as i32, shape)
}

pub fn i64_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::Int64 as i32, shape)
}

pub fn i64_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::Int64 as i32, shape)
}

pub fn u8_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::Uint8 as i32, shape)
}

pub fn u8_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::Uint8 as i32, shape)
}

pub fn i8_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::Int8 as i32, shape)
}

pub fn i8_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::Int8 as i32, shape)
}

pub fn bool_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::Bool as i32, shape)
}

pub fn bool_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::Bool as i32, shape)
}

pub fn string_input(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_input(name, TensorProto_DataType::String as i32, shape)
}

pub fn string_output(name: &str, shape: &[i64]) -> ValueInfoProto {
    tensor_output(name, TensorProto_DataType::String as i32, shape)
}

pub fn sequence_f32_output(name: &str, elem_shape: &[i64]) -> ValueInfoProto {
    value_info(
        name,
        sequence_of_tensor(TensorProto_DataType::Float as i32, elem_shape),
    )
}

pub fn optional_f32_output(name: &str, elem_shape: &[i64]) -> ValueInfoProto {
    value_info(
        name,
        optional_of_tensor(TensorProto_DataType::Float as i32, elem_shape),
    )
}

pub fn f32_init(name: &str, shape: &[i64], data: &[f32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::Float as i32,
        float_data: data.to_vec(),
        ..Default::default()
    }
}

pub fn i32_init(name: &str, shape: &[i64], data: &[i32]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::Int32 as i32,
        int32_data: data.to_vec(),
        ..Default::default()
    }
}

pub fn i64_init(name: &str, shape: &[i64], data: &[i64]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::Int64 as i32,
        int64_data: data.to_vec(),
        ..Default::default()
    }
}

pub fn u8_init(name: &str, shape: &[i64], data: &[u8]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::Uint8 as i32,
        int32_data: data.iter().map(|&b| i32::from(b)).collect(),
        raw_data: data.to_vec(),
        ..Default::default()
    }
}

pub fn i8_init(name: &str, shape: &[i64], data: &[i8]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::Int8 as i32,
        int32_data: data.iter().map(|&b| i32::from(b)).collect(),
        raw_data: data.iter().map(|&b| b as u8).collect(),
        ..Default::default()
    }
}

pub fn bool_init(name: &str, shape: &[i64], data: &[bool]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::Bool as i32,
        int32_data: data.iter().map(|&b| i32::from(b)).collect(),
        ..Default::default()
    }
}

pub fn string_init(name: &str, shape: &[i64], values: &[&str]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        dims: shape.to_vec(),
        data_type: TensorProto_DataType::String as i32,
        string_data: values.iter().map(|s| s.as_bytes().to_vec()).collect(),
        ..Default::default()
    }
}

pub fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

pub fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Float as i32,
        f: value,
        ..Default::default()
    }
}

pub fn attr_string(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

pub fn attr_ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Ints as i32,
        ints: values.to_vec(),
        ..Default::default()
    }
}

pub fn attr_floats(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Floats as i32,
        floats: values.to_vec(),
        ..Default::default()
    }
}

pub fn attr_tensor(name: &str, tensor: TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Tensor as i32,
        t: Some(tensor),
        ..Default::default()
    }
}

pub fn attr_graph(name: &str, graph: GraphProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Graph as i32,
        g: Some(graph),
        ..Default::default()
    }
}

pub fn node(
    op_type: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: &[AttributeProto],
) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        name: name.to_string(),
        input: inputs.iter().map(|s| (*s).to_string()).collect(),
        output: outputs.iter().map(|s| (*s).to_string()).collect(),
        attribute: attributes.to_vec(),
        ..Default::default()
    }
}

pub fn graph(
    name: &str,
    inputs: Vec<ValueInfoProto>,
    outputs: Vec<ValueInfoProto>,
    nodes: Vec<NodeProto>,
    initializers: Vec<TensorProto>,
) -> GraphProto {
    GraphProto {
        name: name.to_string(),
        input: inputs,
        output: outputs,
        node: nodes,
        initializer: initializers,
        ..Default::default()
    }
}

pub fn model(opset: i64, graph: GraphProto) -> ModelProto {
    ModelProto {
        ir_version: 8,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        graph: Some(graph),
        ..Default::default()
    }
}

/// Map ONNX tensor element type code to the `f32_input`-style helper suffix.
pub fn tensor_vi_helper(elem_type: i32, is_output: bool) -> &'static str {
    use TensorProto_DataType::*;
    let base = match elem_type {
        x if x == Float as i32 => "f32",
        x if x == Int32 as i32 => "i32",
        x if x == Int64 as i32 => "i64",
        x if x == Uint8 as i32 => "u8",
        x if x == Bool as i32 => "bool",
        x if x == String as i32 => "string",
        _ => "tensor",
    };
    if base == "tensor" {
        if is_output {
            "tensor_output"
        } else {
            "tensor_input"
        }
    } else if is_output {
        match base {
            "f32" => "f32_output",
            "i32" => "i32_output",
            "i64" => "i64_output",
            "u8" => "u8_output",
            "bool" => "bool_output",
            "string" => "string_output",
            _ => "tensor_output",
        }
    } else {
        match base {
            "f32" => "f32_input",
            "i32" => "i32_input",
            "i64" => "i64_input",
            "u8" => "u8_input",
            "bool" => "bool_input",
            "string" => "string_input",
            _ => "tensor_input",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::convert::{convert_model, ConvertOptions};

    #[test]
    fn hand_written_abs_fixture_builds() {
        let model = model(
            18,
            graph(
                "test_Abs_graph",
                vec![f32_input("X", &[1, 2])],
                vec![f32_output("Y", &[1, 2])],
                vec![node("Abs", "test_Abs", &["X"], &["Y"], &[])],
                vec![],
            ),
        );
        convert_model(model, &ConvertOptions::default()).expect("Abs should convert");
    }
}
