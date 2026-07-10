/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Type conversion and constant operators: Cast, Constant, QuantizeLinear, DequantizeLinear

use crate::onnx::builder::{
    map_onnx_tensor_type, map_op_error, tensor_proto_to_bytes, OnnxBuilder,
};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;

pub struct ConversionHandler;

impl OpHandler for ConversionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Cast" | "Constant" | "QuantizeLinear" | "DequantizeLinear"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.clone()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Cast" => self.convert_cast(node, &node_name, b),
            "Constant" => self.convert_constant(node, &node_name, b),
            "QuantizeLinear" => self.convert_quantize_linear(node, &node_name, context, b),
            "DequantizeLinear" => self.convert_dequantize_linear(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl ConversionHandler {
    fn convert_cast(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Cast expects 1 input, got {}",
                inputs.len()
            )));
        }

        let mut to_type: Option<i64> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "to" && attr.i != 0 {
                to_type = Some(attr.i);
            }
        }
        if to_type.is_none() {
            return Err(OnnxError::MissingAttribute {
                attr: "to".to_string(),
                op: "Cast".to_string(),
            });
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let target_type = map_onnx_tensor_type(to_type.unwrap() as i32)?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .cast_with_options(input, target_type, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_constant(
        &self,
        node: &NodeProto,
        _node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let onnx_out = node
            .output
            .first()
            .ok_or_else(|| OnnxError::InvalidShape("Constant expects 1 output".to_string()))?;

        let tensor = node
            .attribute
            .iter()
            .find_map(|attr| (attr.name.as_str() == "value").then(|| attr.t.as_ref()))
            .flatten()
            .ok_or_else(|| OnnxError::MissingAttribute {
                attr: "value".to_string(),
                op: "Constant".to_string(),
            })?;

        let data_type = crate::onnx::convert::map_onnx_data_type(tensor.data_type)?;
        let shape: Vec<u32> = tensor.dims.iter().map(|&d| d.max(0) as u32).collect();
        let bytes = tensor_proto_to_bytes(tensor)?;
        b.register_constant_from_bytes(onnx_out, data_type, &shape, &bytes)?;
        Ok(ConversionResult::default())
    }

    fn convert_quantize_linear(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "QuantizeLinear expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        if parse_quantize_axis(node) != 1 {
            return Err(OnnxError::unsupported_op("QuantizeLinear", node_name));
        }

        let x_name = &inputs[0];
        let scale_name = &inputs[1];
        validate_quantize_scale_shape(x_name, scale_name, context, node_name)?;

        let output_name = output_label(node, node_name);
        let x = b.resolve_operand(x_name)?;
        let scale = b.resolve_operand(scale_name)?;
        let out = emit_quantize_linear(b, x, scale, inputs.get(2), &output_name)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_dequantize_linear(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "DequantizeLinear expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        if parse_quantize_axis(node) != 1 {
            return Err(OnnxError::unsupported_op("DequantizeLinear", node_name));
        }

        let x_name = &inputs[0];
        let scale_name = &inputs[1];
        validate_quantize_scale_shape(x_name, scale_name, context, node_name)?;

        let output_name = output_label(node, node_name);
        let x = b.resolve_operand(x_name)?;
        let scale = b.resolve_operand(scale_name)?;
        let out = emit_dequantize_linear(b, x, scale, inputs.get(2), &output_name)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }
}

fn parse_quantize_axis(node: &NodeProto) -> i64 {
    let mut axis = 1i64;
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "axis" {
            axis = attr.i;
        }
    }
    axis
}

fn validate_quantize_scale_shape(
    x_name: &str,
    scale_name: &str,
    context: &ConversionContext,
    node_name: &str,
) -> Result<(), OnnxError> {
    let scale_shape = resolve_tensor_shape(scale_name, context);
    if is_scalar_shape(scale_shape.as_deref()) {
        return Ok(());
    }

    let x_shape = context.resolve_shape(x_name);
    if let (Some(x_shape), Some(scale_shape)) = (x_shape, scale_shape.as_deref()) {
        if x_shape == scale_shape {
            return Ok(());
        }
    }

    Err(OnnxError::unsupported_op(
        "QuantizeLinear/DequantizeLinear per-channel",
        node_name,
    ))
}

fn resolve_tensor_shape(name: &str, context: &ConversionContext) -> Option<Vec<i64>> {
    if let Some(shape) = context.resolve_shape(name) {
        return Some(shape.clone());
    }
    context
        .initializers
        .get(name)
        .or_else(|| {
            context
                .initializers
                .get(&crate::onnx::convert::sanitize_identifier(name))
        })
        .map(|tensor| tensor.dims.clone())
}

fn is_scalar_shape(shape: Option<&[i64]>) -> bool {
    match shape {
        None | Some([]) => true,
        Some(dims) => dims.iter().all(|&d| d == 1),
    }
}

fn emit_quantize_linear(
    b: &mut OnnxBuilder<'_, '_, '_>,
    x: MLOperand,
    scale: MLOperand,
    zero_point_name: Option<&String>,
    output_name: &str,
) -> Result<MLOperand, OnnxError> {
    let opts = OnnxBuilder::labeled_options(output_name);
    if let Some(zp_name) = zero_point_name.filter(|s| !s.is_empty()) {
        let zero_point = b.resolve_operand(zp_name)?;
        return b
            .builder
            .quantize_linear_with_zeropoint(x, scale, zero_point)
            .map_err(map_op_error);
    }
    b.builder
        .quantize_linear_with_options(x, scale, None, opts)
        .map_err(map_op_error)
}

fn emit_dequantize_linear(
    b: &mut OnnxBuilder<'_, '_, '_>,
    x: MLOperand,
    scale: MLOperand,
    zero_point_name: Option<&String>,
    output_name: &str,
) -> Result<MLOperand, OnnxError> {
    let opts = OnnxBuilder::labeled_options(output_name);
    if let Some(zp_name) = zero_point_name.filter(|s| !s.is_empty()) {
        let zero_point = b.resolve_operand(zp_name)?;
        return b
            .builder
            .dequantize_linear_with_zeropoint(x, scale, zero_point)
            .map_err(map_op_error);
    }
    b.builder
        .dequantize_linear_with_options(x, scale, None, opts)
        .map_err(map_op_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        });
    }

    #[test]
    fn test_conversion_handler_supports() {
        let handler = ConversionHandler;
        assert!(handler.supports("Cast"));
    }

    #[test]
    fn test_convert_cast() {
        let handler = ConversionHandler;
        let mut node = create_test_node("Cast", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "to", 7);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_constant_registers_operand() {
        let handler = ConversionHandler;
        let mut node = create_test_node("Constant", vec![], vec!["c0"]);
        let tensor = crate::protos::onnx::TensorProto {
            data_type: crate::protos::onnx::TensorProto_DataType::Float as i32,
            dims: vec![1],
            raw_data: vec![0, 0, 128, 63],
            ..Default::default()
        };
        node.attribute.push(AttributeProto {
            name: "value".to_string(),
            t: Some(tensor),
            ..Default::default()
        });
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
