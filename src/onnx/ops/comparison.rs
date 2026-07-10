/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Comparison and logical operators. ONNX bool tensors lower as WebNN `uint8` (0/1).

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;
use rustnn::DataType;

pub struct ComparisonHandler;

fn is_unary_logical(op_type: &str) -> bool {
    op_type == "Not"
}

fn is_binary_logical(op_type: &str) -> bool {
    matches!(op_type, "And" | "Or" | "Xor")
}

impl OpHandler for ComparisonHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Greater"
                | "Less"
                | "Equal"
                | "GreaterOrEqual"
                | "LessOrEqual"
                | "Not"
                | "And"
                | "Or"
                | "Xor"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        let output_name = output_name_for(node, &node_name);
        let opts = OnnxBuilder::labeled_options(&output_name);

        let out = if is_unary_logical(op_type) {
            let inputs = node.input.as_slice();
            if inputs.len() != 1 {
                return Err(OnnxError::InvalidShape(format!(
                    "{op_type} expects 1 input, got {}",
                    inputs.len()
                )));
            }
            let input0 = b.resolve_operand(&inputs[0])?;
            emit_unary_logical(op_type, b, input0, opts, &node_name)?
        } else {
            let inputs = node.input.as_slice();
            if inputs.len() != 2 {
                return Err(OnnxError::InvalidShape(format!(
                    "{op_type} expects 2 inputs, got {}",
                    inputs.len()
                )));
            }
            let input0 = b.resolve_operand(&inputs[0])?;
            let input1 = b.resolve_operand(&inputs[1])?;
            if is_binary_logical(op_type) {
                emit_binary_logical(op_type, b, input0, input1, opts, &node_name)?
            } else {
                emit_comparison(op_type, b, input0, input1, opts, &node_name)?
            }
        };

        record_output(b, node, &output_name, out);

        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            result
                .output_types
                .insert(output.to_string(), DataType::Uint8);
        }

        Ok(result)
    }
}

fn output_name_for(node: &NodeProto, node_name: &str) -> String {
    if node.output.as_slice().is_empty() {
        format!("{}_output", node_name)
    } else {
        sanitize_identifier(&node.output.as_slice()[0].to_string())
    }
}

fn record_output(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node: &NodeProto,
    output_name: &str,
    out: MLOperand,
) {
    if let Some(output) = node.output.as_slice().first() {
        b.record_operand(&[output.as_str(), output_name], out);
    } else {
        b.record_operand(&[output_name], out);
    }
}

fn emit_comparison(
    op_type: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    a: MLOperand,
    b_in: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match op_type {
        "Greater" => b
            .builder
            .greater_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Less" => b
            .builder
            .lesser_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Equal" => b
            .builder
            .equal_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "GreaterOrEqual" => b
            .builder
            .greater_or_equal_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "LessOrEqual" => b
            .builder
            .lesser_or_equal_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::unsupported_op(
                op_type.to_string(),
                node_name.to_string(),
            ))
        }
    })
}

fn emit_unary_logical(
    op_type: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match op_type {
        "Not" => b
            .builder
            .logical_not_with_options(input, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::unsupported_op(
                op_type.to_string(),
                node_name.to_string(),
            ))
        }
    })
}

fn emit_binary_logical(
    op_type: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    a: MLOperand,
    b_in: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match op_type {
        "And" => b
            .builder
            .logical_and_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Or" => b
            .builder
            .logical_or_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Xor" => b
            .builder
            .logical_xor_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::unsupported_op(
                op_type.to_string(),
                node_name.to_string(),
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;
    use std::collections::HashMap;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_comparison_handler_supports() {
        let handler = ComparisonHandler;
        assert!(handler.supports("Greater"));
        assert!(handler.supports("Less"));
        assert!(handler.supports("Equal"));
        assert!(handler.supports("GreaterOrEqual"));
        assert!(handler.supports("LessOrEqual"));
        assert!(handler.supports("Not"));
        assert!(handler.supports("And"));
        assert!(handler.supports("Or"));
        assert!(handler.supports("Xor"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_greater() {
        let handler = ComparisonHandler;
        let node = create_test_node("Greater", vec!["a", "b"], vec!["c"]);
        let result = crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
        assert_eq!(result.output_types.get("c"), Some(&DataType::Uint8));
    }

    #[test]
    fn test_convert_equal() {
        let handler = ComparisonHandler;
        let node = create_test_node("Equal", vec!["x", "y"], vec!["z"]);
        let result = crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
        assert_eq!(result.output_types.get("z"), Some(&DataType::Uint8));
    }

    #[test]
    fn test_convert_less() {
        let handler = ComparisonHandler;
        let node = create_test_node("Less", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_greater_or_equal() {
        let handler = ComparisonHandler;
        let node = create_test_node("GreaterOrEqual", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_less_or_equal() {
        let handler = ComparisonHandler;
        let node = create_test_node("LessOrEqual", vec!["a", "b"], vec!["c"]);
        let result = crate::onnx::ops::convert_with_test_builder(&handler, &node);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convert_not() {
        let handler = ComparisonHandler;
        let node = create_test_node("Not", vec!["x"], vec!["y"]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("x".to_string(), vec![2, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("x".to_string(), DataType::Uint8);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(result.output_types.get("y"), Some(&DataType::Uint8));
    }

    #[test]
    fn test_convert_and() {
        let handler = ComparisonHandler;
        let node = create_test_node("And", vec!["a", "b"], vec!["c"]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("a".to_string(), vec![2, 2]);
        value_shapes.insert("b".to_string(), vec![2, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("a".to_string(), DataType::Uint8);
        value_types.insert("b".to_string(), DataType::Uint8);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(result.output_types.get("c"), Some(&DataType::Uint8));
    }

    #[test]
    fn test_convert_or() {
        let handler = ComparisonHandler;
        let node = create_test_node("Or", vec!["a", "b"], vec!["c"]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("a".to_string(), vec![2, 2]);
        value_shapes.insert("b".to_string(), vec![2, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("a".to_string(), DataType::Uint8);
        value_types.insert("b".to_string(), DataType::Uint8);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }

    #[test]
    fn test_convert_xor() {
        let handler = ComparisonHandler;
        let node = create_test_node("Xor", vec!["a", "b"], vec!["c"]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("a".to_string(), vec![2, 2]);
        value_shapes.insert("b".to_string(), vec![2, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("a".to_string(), DataType::Uint8);
        value_types.insert("b".to_string(), DataType::Uint8);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }
}
