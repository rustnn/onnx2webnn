/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Recurrent operators: GRU, LSTM

use crate::onnx::builder::{map_op_error, operand_index, OnnxBuilder};
use crate::onnx::builder_helpers::{
    map_op_result, output_label, record_node_output, slice_with_params,
};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{
    MLDimension, MLGruOptions, MLLstmOptions, MLSqueezeOptions, MLUnsqueezeOptions,
};

pub struct RnnHandler;

impl OpHandler for RnnHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "GRU" | "LSTM")
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
            "GRU" => self.convert_gru(node, &node_name, context, b),
            "LSTM" => self.convert_lstm(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl RnnHandler {
    fn convert_gru(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "GRU expects at least 3 inputs (X, W, R), got {}",
                inputs.len()
            )));
        }

        validate_rnn_attrs(node, node_name, "GRU")?;
        reject_optional_rnn_inputs(inputs, 4, node_name, "GRU")?;

        let hidden_size = require_hidden_size(node, "GRU")?;
        let gate_bias_len = 3u32 * hidden_size;

        let x = b.resolve_operand(&inputs[0])?;
        let w = b.resolve_operand(&inputs[1])?;
        let r = b.resolve_operand(&inputs[2])?;
        let steps = resolve_steps(context, &inputs[0]);

        let (bias, recurrent_bias) = split_combined_bias(
            b,
            node_name,
            inputs.get(3).map(String::as_str),
            gate_bias_len,
        )?;

        let outputs = node.output.as_slice();
        let wants_sequence = outputs.first().is_some_and(|name| !name.is_empty());
        let wants_hidden = outputs.get(1).is_some_and(|name| !name.is_empty());

        let mut linear_before_reset = 0i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "linear_before_reset" {
                linear_before_reset = attr.i;
            }
        }

        let label = output_label(node, node_name);
        let options = MLGruOptions {
            label: label.clone(),
            bias,
            recurrent_bias,
            return_sequence: wants_sequence,
            direction: "forward".to_string(),
            reset_after: linear_before_reset != 0,
            ..Default::default()
        };

        let gru_outputs = b
            .builder
            .gru_with_options(x, w, r, steps, hidden_size, options)
            .map_err(map_op_error)?;

        let mut result = ConversionResult::default();
        let input_dtype = context
            .value_types
            .get(&inputs[0])
            .cloned()
            .unwrap_or(rustnn::DataType::Float32);

        if wants_sequence {
            let seq = gru_outputs.get(1).copied().ok_or_else(|| {
                OnnxError::InvalidShape("GRU missing sequence output".to_string())
            })?;
            let mapped = map_onnx_sequence_output(b, node_name, seq, context, &outputs[0])?;
            record_node_output(b, &outputs[0], &format!("{label}_y"), mapped);
            result.output_types.insert(outputs[0].clone(), input_dtype);
        }

        if wants_hidden {
            let hidden = gru_outputs[0];
            let out_name = outputs.get(1).expect("checked above");
            record_node_output(b, out_name, &format!("{label}_y_h"), hidden);
            result.output_types.insert(out_name.clone(), input_dtype);
        }

        Ok(result)
    }

    fn convert_lstm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 3 {
            return Err(OnnxError::InvalidShape(format!(
                "LSTM expects at least 3 inputs (X, W, R), got {}",
                inputs.len()
            )));
        }

        validate_rnn_attrs(node, node_name, "LSTM")?;
        reject_optional_rnn_inputs(inputs, 4, node_name, "LSTM")?;

        let hidden_size = require_hidden_size(node, "LSTM")?;
        let gate_bias_len = 4u32 * hidden_size;

        let x = b.resolve_operand(&inputs[0])?;
        let w = b.resolve_operand(&inputs[1])?;
        let r = b.resolve_operand(&inputs[2])?;
        let steps = resolve_steps(context, &inputs[0]);

        let (bias, recurrent_bias) = split_combined_bias(
            b,
            node_name,
            inputs.get(3).map(String::as_str),
            gate_bias_len,
        )?;

        let outputs = node.output.as_slice();
        let wants_sequence = outputs.first().is_some_and(|name| !name.is_empty());
        let wants_hidden = outputs.get(1).is_some_and(|name| !name.is_empty());
        let wants_cell = outputs.get(2).is_some_and(|name| !name.is_empty());

        let label = output_label(node, node_name);
        let options = MLLstmOptions {
            label: label.clone(),
            bias,
            recurrent_bias,
            return_sequence: wants_sequence,
            direction: "forward".to_string(),
            ..Default::default()
        };

        let lstm_outputs = b
            .builder
            .lstm_with_options(x, w, r, steps, hidden_size, options)
            .map_err(map_op_error)?;

        let mut result = ConversionResult::default();
        let input_dtype = context
            .value_types
            .get(&inputs[0])
            .cloned()
            .unwrap_or(rustnn::DataType::Float32);

        if wants_sequence {
            let seq = lstm_outputs.get(2).copied().ok_or_else(|| {
                OnnxError::InvalidShape("LSTM missing sequence output".to_string())
            })?;
            let mapped = map_onnx_sequence_output(b, node_name, seq, context, &outputs[0])?;
            record_node_output(b, &outputs[0], &format!("{label}_y"), mapped);
            result.output_types.insert(outputs[0].clone(), input_dtype);
        }

        if wants_hidden {
            let out_name = outputs.get(1).expect("checked above");
            record_node_output(b, out_name, &format!("{label}_y_h"), lstm_outputs[0]);
            result.output_types.insert(out_name.clone(), input_dtype);
        }

        if wants_cell {
            let out_name = outputs.get(2).expect("checked above");
            record_node_output(b, out_name, &format!("{label}_y_c"), lstm_outputs[1]);
            result.output_types.insert(out_name.clone(), input_dtype);
        }

        Ok(result)
    }
}

fn require_hidden_size(node: &NodeProto, op: &str) -> Result<u32, OnnxError> {
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "hidden_size" && attr.i > 0 {
            return u32::try_from(attr.i).map_err(|_| {
                OnnxError::InvalidShape(format!("{op} hidden_size {} is out of range", attr.i))
            });
        }
    }
    Err(OnnxError::MissingAttribute {
        attr: "hidden_size".to_string(),
        op: op.to_string(),
    })
}

fn validate_rnn_attrs(node: &NodeProto, node_name: &str, op: &str) -> Result<(), OnnxError> {
    for attr in node.attribute.as_slice() {
        match attr.name.as_str() {
            "direction" => {
                let direction = String::from_utf8_lossy(&attr.s);
                if !direction.is_empty() && direction != "forward" && direction != "FORWARD" {
                    return Err(OnnxError::unsupported_op(
                        format!("{op}(direction={direction})"),
                        node_name.to_string(),
                    ));
                }
            }
            "layout" => {
                let layout = String::from_utf8_lossy(&attr.s);
                if !layout.is_empty() && layout != "zrh" && layout != "iofg" {
                    return Err(OnnxError::unsupported_op(
                        format!("{op}(layout={layout})"),
                        node_name.to_string(),
                    ));
                }
            }
            "activations" if !attr.strings.is_empty() => {
                return Err(OnnxError::unsupported_op(
                    format!("{op}(custom activations)"),
                    node_name.to_string(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn reject_optional_rnn_inputs(
    inputs: &[String],
    first_optional_index: usize,
    node_name: &str,
    op: &str,
) -> Result<(), OnnxError> {
    for (idx, name) in inputs.iter().enumerate().skip(first_optional_index) {
        if !name.is_empty() {
            return Err(OnnxError::unsupported_op(
                format!("{op} optional input {idx}"),
                node_name.to_string(),
            ));
        }
    }
    Ok(())
}

fn resolve_steps(context: &ConversionContext, x_name: &str) -> u32 {
    match context.input_rank(x_name) {
        Some(3) => context
            .resolve_shape(x_name)
            .and_then(|shape| shape.first().copied())
            .filter(|&dim| dim > 0)
            .and_then(|dim| u32::try_from(dim).ok())
            .unwrap_or(1),
        _ => 1,
    }
}

/// Split ONNX combined bias `[1, 2*gate_bias_len]` into WebNN `bias` and `recurrent_bias` `[1, gate_bias_len]`.
fn split_combined_bias(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node_name: &str,
    bias_name: Option<&str>,
    gate_bias_len: u32,
) -> Result<(Option<u32>, Option<u32>), OnnxError> {
    let Some(name) = bias_name.filter(|n| !n.is_empty()) else {
        return Ok((None, None));
    };

    let combined = b.resolve_operand(name)?;
    let half = gate_bias_len;
    let bias = slice_with_params(
        b,
        combined,
        &format!("{node_name}_bias"),
        &[0, 0],
        &[MLDimension::Static(1), MLDimension::Static(half)],
    )?;
    let recurrent_bias = slice_with_params(
        b,
        combined,
        &format!("{node_name}_recurrent_bias"),
        &[0, half],
        &[MLDimension::Static(1), MLDimension::Static(half)],
    )?;
    Ok((
        Some(operand_index(bias)),
        Some(operand_index(recurrent_bias)),
    ))
}

/// Map WebNN sequence `[steps, num_directions, batch, hidden]` to ONNX `Y` layout.
fn map_onnx_sequence_output(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node_name: &str,
    seq: MLOperand,
    context: &ConversionContext,
    onnx_output: &str,
) -> Result<MLOperand, OnnxError> {
    let expected_rank = context.resolve_shape(onnx_output).map(|shape| shape.len());

    match expected_rank {
        Some(3) => {
            let opts = MLSqueezeOptions {
                label: format!("{node_name}_squeeze_dir"),
                axes: vec![1],
            };
            map_op_result(b.builder.squeeze_with_options(seq, opts))
        }
        Some(4) => Ok(seq),
        _ => {
            // Unidirectional forward: squeeze `num_directions` when rank is unspecified.
            let opts = MLSqueezeOptions {
                label: format!("{node_name}_squeeze_dir"),
                axes: vec![1],
            };
            let squeezed = map_op_result(b.builder.squeeze_with_options(seq, opts))?;
            if let Some(shape) = context.resolve_shape(onnx_output) {
                if shape.len() == 4 {
                    let unsqueeze_opts = MLUnsqueezeOptions {
                        label: format!("{node_name}_unsqueeze_dir"),
                        axes: vec![1],
                    };
                    return map_op_result(
                        b.builder.unsqueeze_with_options(squeezed, unsqueeze_opts),
                    );
                }
            }
            Ok(squeezed)
        }
    }
}
