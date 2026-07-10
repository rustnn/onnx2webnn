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

// Utility operators: Shape, Gather, GatherND, GatherElements, ReverseSequence, Slice

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    ast_dims_to_mldim, expand_with_shape, i64_starts_as_u32, output_label, record_node_output,
    slice_sizes_from_i64, slice_with_params,
};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{
    normalize_axis_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use half::f16;
use rustnn::operator_options::{
    MLDimension, MLDynamicDimension, MLGatherOptions, MLReverseOptions, MLTriangularOptions,
};
use rustnn::DataType;

pub struct UtilityHandler;

impl OpHandler for UtilityHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Shape"
                | "Gather"
                | "GatherND"
                | "GatherElements"
                | "ReverseSequence"
                | "Slice"
                | "ConstantOfShape"
                | "Range"
                | "Trilu"
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
            node.name.as_str().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Shape" => self.convert_shape(node, &node_name, b),
            "Gather" => self.convert_gather(node, &node_name, context, b),
            "GatherND" => self.convert_gather_nd(node, &node_name, context, b),
            "GatherElements" => self.convert_gather_elements(node, &node_name, context, b),
            "ReverseSequence" => self.convert_reverse_sequence(node, &node_name, context, b),
            "Slice" => self.convert_slice(node, &node_name, context, b),
            "ConstantOfShape" => self.convert_constant_of_shape(node, &node_name, context, b),
            "Range" => self.convert_range(node, &node_name, context, b),
            "Trilu" => self.convert_trilu(node, &node_name, context, b),
            _ => Err(OnnxError::unsupported_op(op_type.to_string(), node_name)),
        }
    }
}

impl UtilityHandler {
    /// Convert ONNX Shape to WebNN shape operation
    /// Returns a 1D tensor containing the dimensions of the input
    fn convert_shape(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Shape expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .shape_with_options(input, opts)
            .map_err(map_op_error)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        Ok(ConversionResult::default())
    }

    fn read_scalar_f64(&self, name: &str, context: &ConversionContext) -> Option<f64> {
        use crate::protos::onnx::TensorProto_DataType;
        if let Some(t) = context.initializers.get(name) {
            if t.data_type == TensorProto_DataType::Float as i32 {
                if !t.float_data.is_empty() {
                    return Some(f64::from(t.float_data[0]));
                }
                let raw = t.raw_data.as_slice();
                if raw.len() >= 4 {
                    let bits = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                    return Some(f64::from(f32::from_bits(bits)));
                }
            }
            if t.data_type == TensorProto_DataType::Float16 as i32 {
                if !t.int32_data.is_empty() {
                    return Some(f64::from(f16::from_bits(t.int32_data[0] as u16).to_f32()));
                }
                let raw = t.raw_data.as_slice();
                if raw.len() >= 2 {
                    let bits = u16::from_le_bytes([raw[0], raw[1]]);
                    return Some(f64::from(f16::from_bits(bits).to_f32()));
                }
            }
            if t.data_type == TensorProto_DataType::Double as i32 {
                if !t.double_data.is_empty() {
                    return Some(t.double_data[0]);
                }
                let raw = t.raw_data.as_slice();
                if raw.len() >= 8 {
                    let bits = u64::from_le_bytes([
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                    ]);
                    return Some(f64::from_bits(bits));
                }
            }
        }
        // Integer initializers / folded consts represent exact values.
        self.read_scalar_i64(name, context).map(|v| v as f64)
    }

    /// Element type of an ONNX `Range` (all three inputs share type `T`).
    ///
    /// Returns the ONNX `Range` element type. `float64` is materialized as `float32`; `float16`
    /// stays float16.
    fn range_element_type(&self, inputs: &[String], context: &ConversionContext) -> DataType {
        use crate::protos::onnx::TensorProto_DataType;
        for name in inputs.iter().take(3) {
            if let Some(t) = context.initializers.get(name.as_str()) {
                let dt = t.data_type;
                if dt == TensorProto_DataType::Float16 as i32 {
                    return DataType::Float16;
                }
                if dt == TensorProto_DataType::Float as i32
                    || dt == TensorProto_DataType::Double as i32
                {
                    return DataType::Float32;
                }
            }
            if let Some(dt) = context.value_types.get(name.as_str()) {
                if matches!(dt, DataType::Float16) {
                    return DataType::Float16;
                }
                if matches!(dt, DataType::Float32) {
                    return DataType::Float32;
                }
            }
        }
        DataType::Int64
    }

    fn read_scalar_i64(&self, name: &str, context: &ConversionContext) -> Option<i64> {
        if let Some(vals) = context.const_values.get(name) {
            return vals.first().copied();
        }
        if let Some(t) = context.initializers.get(name) {
            let raw = t.raw_data.as_slice();
            if !raw.is_empty() {
                if t.data_type == crate::protos::onnx::TensorProto_DataType::Int32 as i32 {
                    return Some(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64);
                }
                if raw.len() >= 8 {
                    return Some(i64::from_le_bytes([
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                    ]));
                }
            } else if !t.int64_data.as_slice().is_empty() {
                return t.int64_data.as_slice().first().copied();
            } else if !t.int32_data.as_slice().is_empty() {
                return t.int32_data.as_slice().first().map(|v| *v as i64);
            }
        }
        None
    }

    fn scalar_as_i64(&self, name: &str, context: &ConversionContext) -> Option<i64> {
        self.read_scalar_i64(name, context)
            .or_else(|| self.read_scalar_f64(name, context).map(|v| v as i64))
    }

    fn convert_range(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "Range expects 3 inputs (start, limit, delta), got {}",
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        // Float ranges materialize a constant. The dynamic (symbolic-dim) path below is
        // integer-only — it exists to express runtime shape dimensions, which are always integral.
        let range_dtype = self.range_element_type(inputs, context);
        if matches!(range_dtype, DataType::Float32 | DataType::Float16) {
            return self.convert_range_static_float(
                node,
                node_name,
                &output_name,
                range_dtype,
                context,
                b,
            );
        }

        let start = self.scalar_as_i64(&inputs[0], context);
        let limit = self.scalar_as_i64(&inputs[1], context);
        let delta = self.scalar_as_i64(&inputs[2], context);

        let start_dim = crate::onnx::shape_inference::dynamic_scalar_dimension_for_value(
            &inputs[0],
            context.value_shape_dims,
        );
        if let (Some(start), Some(delta), Some(limit_dim)) = (
            start,
            delta,
            crate::onnx::shape_inference::dynamic_scalar_dimension_for_value(
                &inputs[1],
                context.value_shape_dims,
            ),
        ) {
            let range_dim = crate::onnx::shape_inference::dynamic_range_length_dimension(
                start,
                delta,
                start_dim.as_ref(),
                &limit_dim,
            )
            .ok_or_else(|| {
                OnnxError::InvalidShape(format!(
                    "Range {} requires dynamic range length to be representable as <dim> +/- const with delta=1",
                    node_name,
                ))
            })?;

            let max_len = usize::try_from(range_dim.max_size).map_err(|_| {
                OnnxError::InvalidShape(format!(
                    "Range {} max size {} does not fit in usize",
                    node_name, range_dim.max_size
                ))
            })?;

            let use_runtime_start = start_dim.is_some();
            let mut values = Vec::with_capacity(max_len.max(1));
            let mut current = if use_runtime_start { 0 } else { start };
            for _ in 0..max_len {
                values.push(current);
                current += delta;
            }
            if values.is_empty() {
                values.push(if use_runtime_start { 0 } else { start });
            }

            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| v.to_le_bytes().to_vec())
                .collect();

            let range_const_name = format!("{}_range_const", output_name);
            b.register_constant_from_bytes(
                &range_const_name,
                DataType::Int64,
                &[values.len() as u32],
                &bytes,
            )?;

            let range_const = b.resolve_operand(&range_const_name)?;
            let slice_sizes = vec![MLDimension::Dynamic(MLDynamicDimension {
                name: range_dim.name.clone(),
                max_size: range_dim.max_size,
            })];
            let sliced = slice_with_params(b, range_const, &output_name, &[0], &slice_sizes)?;
            let out = if use_runtime_start {
                let start_op = b.resolve_operand(&inputs[0])?;
                let opts = OnnxBuilder::labeled_options(&output_name);
                b.builder
                    .add_with_options(sliced, start_op, opts)
                    .map_err(map_op_error)?
            } else {
                sliced
            };
            if let Some(out_name) = node.output.as_slice().first() {
                record_node_output(b, out_name, &output_name, out);
            }
            let mut result = ConversionResult::default();
            if let Some(out) = node.output.as_slice().first() {
                result.output_types.insert(out.to_string(), DataType::Int64);
            }
            return Ok(result);
        }

        let start = start.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {} requires a constant scalar start input",
                node_name
            ))
        })?;
        let limit = limit.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {} requires a constant scalar or supported dynamic limit input",
                node_name
            ))
        })?;
        let delta = delta.ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {} requires a constant scalar delta input",
                node_name
            ))
        })?;

        if delta == 0 {
            return Err(OnnxError::InvalidShape(
                "Range delta cannot be zero".to_string(),
            ));
        }

        let mut values = Vec::new();
        let mut v = start;
        if delta > 0 {
            while v < limit {
                values.push(v);
                v += delta;
            }
        } else {
            while v > limit {
                values.push(v);
                v += delta;
            }
        }

        if values.is_empty() {
            values.push(0);
        }

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect();

        b.register_constant_from_bytes(
            &output_name,
            DataType::Int64,
            &[values.len() as u32],
            &bytes,
        )?;
        if let Some(out) = node.output.as_slice().first() {
            record_node_output(b, out, &output_name, b.resolve_operand(&output_name)?);
        }
        let mut result = ConversionResult::default();
        if let Some(out) = node.output.as_slice().first() {
            result.output_types.insert(out.to_string(), DataType::Int64);
        }

        Ok(result)
    }

    /// Fully-static float `Range`: emit the sequence as a `float32` constant.
    ///
    /// ONNX defines the length as `max(ceil((limit - start) / delta), 0)` and element `i` as
    /// `start + i * delta`. Indexing by `i` (rather than accumulating) avoids float drift.
    fn convert_range_static_float(
        &self,
        node: &NodeProto,
        node_name: &str,
        output_name: &str,
        dtype: DataType,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        let start = self.read_scalar_f64(&inputs[0], context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {node_name} requires a constant scalar start input"
            ))
        })?;
        let limit = self.read_scalar_f64(&inputs[1], context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {node_name} requires a constant scalar limit input"
            ))
        })?;
        let delta = self.read_scalar_f64(&inputs[2], context).ok_or_else(|| {
            OnnxError::InvalidShape(format!(
                "Range {node_name} requires a constant scalar delta input"
            ))
        })?;

        if delta == 0.0 {
            return Err(OnnxError::InvalidShape(
                "Range delta cannot be zero".to_string(),
            ));
        }

        let count = ((limit - start) / delta).ceil();
        let count = if count.is_finite() && count > 0.0 {
            count as usize
        } else {
            0
        };

        let mut values: Vec<f32> = (0..count)
            .map(|i| (start + (i as f64) * delta) as f32)
            .collect();
        // WebNN constants cannot be zero-length; match the integer path which emits one element
        // for an empty range.
        if values.is_empty() {
            values.push(0.0);
        }

        let bytes: Vec<u8> = match dtype {
            DataType::Float16 => values
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
            _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        };
        b.register_constant_from_bytes(output_name, dtype, &[values.len() as u32], &bytes)?;
        if let Some(out) = node.output.as_slice().first() {
            record_node_output(b, out, output_name, b.resolve_operand(output_name)?);
        }

        let mut result = ConversionResult::default();
        if let Some(out) = node.output.as_slice().first() {
            result.output_types.insert(out.to_string(), dtype);
        }
        Ok(result)
    }

    fn convert_trilu(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Trilu expects at least 1 input (data)".to_string(),
            ));
        }

        if inputs.len() > 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Trilu expects at most 2 inputs (data, k), got {}",
                inputs.len()
            )));
        }

        let mut upper = true;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "upper" {
                upper = attr.i != 0;
            }
        }

        let mut k: i64 = 0;
        if inputs.len() == 2 {
            let k_input = inputs[1].as_str();
            if let Some(offset) = self.read_scalar_i64(k_input, context) {
                k = offset;
            } else {
                return Err(OnnxError::InvalidShape(
                    "Trilu k input must be a constant scalar for WebNN".to_string(),
                ));
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let opts = MLTriangularOptions {
            label: output_name.clone(),
            upper: Some(upper),
            diagonal: k as i32,
        };
        let out = b
            .builder
            .triangular_with_options(input, opts)
            .map_err(map_op_error)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(output.to_string(), *dtype);
            }
        }

        Ok(result)
    }

    /// Convert ConstantOfShape into an inline constant when the output shape is statically known.
    fn convert_constant_of_shape(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let output_dim_shape = node
            .output
            .as_slice()
            .first()
            .and_then(|out| {
                let out_s = out.to_string();
                context
                    .value_shape_dims
                    .get(&out_s)
                    .or_else(|| context.value_shape_dims.get(&sanitize_identifier(&out_s)))
                    .or_else(|| context.value_shape_dims.get(out_s.trim_start_matches('/')))
            })
            .cloned();

        // Determine the target shape: prefer inferred output shape, otherwise try the shape input const.
        let mut shape: Option<Vec<i64>> = None;
        if let Some(out) = node.output.as_slice().first() {
            if let Some(s) = context.value_shapes.get(out) {
                shape = Some(s.clone());
            } else {
                let sanitized = sanitize_identifier(out);
                if let Some(s) = context.value_shapes.get(&sanitized) {
                    shape = Some(s.clone());
                }
            }
        }
        if shape.is_none() {
            if let Some(shape_input) = node.input.as_slice().first() {
                if let Some(vals) = context.const_values.get(shape_input) {
                    shape = Some(vals.clone());
                } else if let Some(len_shape) = context.value_shapes.get(shape_input) {
                    // If we only know the length of the shape tensor, default the dims to 1s.
                    if len_shape.len() == 1 && len_shape[0] > 0 {
                        shape = Some(vec![1; len_shape[0] as usize]);
                    }
                }
            }
        }

        // Determine fill value and data type (default int64 zero)
        let mut fill_value_i64: i64 = 0;
        let mut dtype = DataType::Int64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "value" {
                if let Some(t) = attr.t.as_ref() {
                    match t.data_type {
                        // FLOAT
                        x if x == crate::protos::onnx::TensorProto_DataType::Float as i32 => {
                            dtype = DataType::Float32;
                            if !t.float_data.as_slice().is_empty() {
                                fill_value_i64 = t.float_data.as_slice()[0].to_bits() as i64;
                            } else if !t.raw_data.as_slice().is_empty()
                                && t.raw_data.as_slice().len() >= 4
                            {
                                let raw = &t.raw_data.as_slice()[..4];
                                let bits = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                                fill_value_i64 = bits as i64;
                            } else {
                                fill_value_i64 = 0f32.to_bits() as i64;
                            }
                        }
                        x if x == crate::protos::onnx::TensorProto_DataType::Float16 as i32 => {
                            dtype = DataType::Float16;
                            if !t.int32_data.as_slice().is_empty() {
                                fill_value_i64 = t.int32_data.as_slice()[0] as i64;
                            } else if !t.raw_data.as_slice().is_empty()
                                && t.raw_data.as_slice().len() >= 2
                            {
                                let raw = &t.raw_data.as_slice()[..2];
                                fill_value_i64 = u16::from_le_bytes([raw[0], raw[1]]) as i64;
                            } else {
                                fill_value_i64 = f16::from_f32(0.0).to_bits() as i64;
                            }
                        }
                        // INT64
                        x if x == crate::protos::onnx::TensorProto_DataType::Int64 as i32 => {
                            dtype = DataType::Int64;
                            if !t.int64_data.as_slice().is_empty() {
                                fill_value_i64 = t.int64_data.as_slice()[0];
                            } else if !t.raw_data.as_slice().is_empty()
                                && t.raw_data.as_slice().len() >= 8
                            {
                                let raw = &t.raw_data.as_slice()[..8];
                                fill_value_i64 = i64::from_le_bytes([
                                    raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                                ]);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        if let Some(dims) = output_dim_shape.as_ref().filter(|dims| {
            dims.iter()
                .any(|d| matches!(d, rustnn::graph::Dimension::Dynamic(_)))
        }) {
            let scalar_bytes = match dtype {
                DataType::Float32 => {
                    let f = f32::from_bits(fill_value_i64 as u32);
                    f.to_le_bytes().to_vec()
                }
                DataType::Float16 => (fill_value_i64 as u16).to_le_bytes().to_vec(),
                _ => fill_value_i64.to_le_bytes().to_vec(),
            };
            let scalar_name = format!("{}_fill", output_name);
            b.register_constant_from_bytes(&scalar_name, dtype, &[1], &scalar_bytes)?;

            let scalar = b.resolve_operand(&scalar_name)?;
            let out = expand_with_shape(b, scalar, &output_name, ast_dims_to_mldim(dims))?;
            if let Some(out_name) = node.output.as_slice().first() {
                record_node_output(b, out_name, &output_name, out);
            }
            let mut result = ConversionResult::default();
            if let Some(out) = node.output.as_slice().first() {
                result.output_types.insert(out.to_string(), dtype);
            }
            return Ok(result);
        }

        let shape = shape.unwrap_or_else(|| vec![1]);

        let mut numel: usize = 1;
        for d in &shape {
            if *d <= 0 {
                return Err(OnnxError::InvalidShape(format!(
                    "ConstantOfShape '{}' has non-positive dimension {:?}",
                    node_name, shape
                )));
            }
            numel = numel.saturating_mul(*d as usize);
        }

        let bytes = match dtype {
            DataType::Float32 => {
                let f = f32::from_bits(fill_value_i64 as u32);
                let val = f.to_le_bytes();
                val.repeat(numel)
            }
            DataType::Float16 => {
                let val = (fill_value_i64 as u16).to_le_bytes();
                val.repeat(numel)
            }
            _ => {
                let val = fill_value_i64.to_le_bytes();
                val.repeat(numel)
            }
        };

        b.register_constant_from_bytes(
            &output_name,
            dtype,
            &shape.iter().map(|d| *d as u32).collect::<Vec<_>>(),
            &bytes,
        )?;
        if let Some(out) = node.output.as_slice().first() {
            record_node_output(b, out, &output_name, b.resolve_operand(&output_name)?);
        }
        let mut result = ConversionResult::default();
        if let Some(out) = node.output.as_slice().first() {
            result.output_types.insert(out.to_string(), dtype);
        }

        Ok(result)
    }

    /// Convert ONNX Gather to WebNN gather
    /// Gathers elements along a specified axis using indices
    fn convert_gather(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gather expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (default: 0)
        let mut axis = 0i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = output_label(node, node_name);
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank)
        } else {
            axis
        };
        let data = b.resolve_operand(&inputs[0])?;
        let indices = b.resolve_operand(&inputs[1])?;
        let opts = MLGatherOptions {
            label: output_name.clone(),
            axis: axis as u32,
        };
        let out = b
            .builder
            .gather_with_options(data, indices, opts)
            .map_err(map_op_error)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(output.to_string(), *dtype);
            }
        }

        Ok(result)
    }

    /// Convert ONNX GatherND to WebNN gatherND.
    ///
    /// `batch_dims` is ignored for now (WebNN gatherND has no batch-dimension option).
    fn convert_gather_nd(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "GatherND expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let data = b.resolve_operand(&inputs[0])?;
        let indices = b.resolve_operand(&inputs[1])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .gather_nd_with_options(data, indices, opts)
            .map_err(map_op_error)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(output.to_string(), *dtype);
            }
        }

        Ok(result)
    }

    /// Convert ONNX GatherElements to WebNN gatherElements.
    fn convert_gather_elements(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "GatherElements expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        let mut axis = 0i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = output_label(node, node_name);
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank)
        } else {
            axis
        };
        let data = b.resolve_operand(&inputs[0])?;
        let indices = b.resolve_operand(&inputs[1])?;
        let opts = MLGatherOptions {
            label: output_name.clone(),
            axis: axis as u32,
        };
        let out = b
            .builder
            .gather_elements_with_options(data, indices, opts)
            .map_err(map_op_error)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(output.to_string(), *dtype);
            }
        }

        Ok(result)
    }

    /// Convert ONNX ReverseSequence to WebNN reverse along `time_axis`.
    ///
    /// `sequence_lens` is accepted as the second input but not used yet — WebNN `reverse`
    /// reverses the full axis and has no per-sequence length support.
    fn convert_reverse_sequence(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "ReverseSequence expects 2 inputs (data, sequence_lens), got {}",
                inputs.len()
            )));
        }

        let mut _batch_axis = 1i64;
        let mut time_axis = 0i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "batch_axis" => _batch_axis = attr.i,
                "time_axis" => time_axis = attr.i,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let time_axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(time_axis, rank)
        } else {
            time_axis
        };
        let input = b.resolve_operand(&inputs[0])?;
        let opts = MLReverseOptions {
            label: output_name.clone(),
            axes: Some(vec![time_axis as u32]),
        };
        let out = b
            .builder
            .reverse_with_options(input, opts)
            .map_err(map_op_error)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(output.to_string(), *dtype);
            }
        }

        Ok(result)
    }

    /// Convert ONNX Slice to WebNN slice
    /// Extracts a slice from the input tensor
    fn convert_slice(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Slice expects at least 1 input".to_string(),
            ));
        }

        let output_name = output_label(node, node_name);
        let mut slice_params: Option<(Vec<u32>, Vec<MLDimension>)> = None;

        let read_ints = |name: &str, context: &ConversionContext| -> Option<Vec<i64>> {
            if let Some(vals) = context.const_values.get(name) {
                return Some(vals.clone());
            }
            if let Some(t) = context.initializers.get(name) {
                let raw = t.raw_data.as_slice();
                if !raw.is_empty() {
                    if t.data_type == crate::protos::onnx::TensorProto_DataType::Int32 as i32 {
                        return Some(
                            raw.chunks_exact(4)
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                .collect(),
                        );
                    }
                    return Some(
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect(),
                    );
                } else if !t.int64_data.as_slice().is_empty() {
                    return Some(t.int64_data.as_slice().to_vec());
                } else if !t.int32_data.as_slice().is_empty() {
                    return Some(t.int32_data.as_slice().iter().map(|&v| v as i64).collect());
                }
            }
            None
        };

        // In opset >= 10, starts/ends/axes/steps are inputs
        // WebNN requires static values, so we enforce const-ness here.
        if inputs.len() >= 3 {
            let starts_name = inputs[1].as_str();
            let ends_name = inputs[2].as_str();
            let mut starts = read_ints(starts_name, context);
            let mut ends = read_ints(ends_name, context);

            if starts.is_none() || ends.is_none() {
                // As a last resort, try to pull starts/ends from sibling consts
                // produced by earlier shape inference passes.
                if let Some(s) = context.const_values.get(starts_name) {
                    starts = Some(s.clone());
                }
                if let Some(e) = context.const_values.get(ends_name) {
                    ends = Some(e.clone());
                }

                let fallback_len = if let Some(axes_name) = inputs.get(3).map(|s| s.as_str()) {
                    read_ints(axes_name, context)
                        .map(|v| v.len())
                        .unwrap_or_else(|| {
                            starts
                                .as_ref()
                                .map(|v| v.len())
                                .or_else(|| {
                                    context
                                        .value_shapes
                                        .get(inputs[0].as_str())
                                        .map(|s| s.len())
                                })
                                .unwrap_or(1)
                        })
                } else {
                    starts
                        .as_ref()
                        .map(|v| v.len())
                        .or_else(|| {
                            context
                                .value_shapes
                                .get(inputs[0].as_str())
                                .map(|s| s.len())
                        })
                        .unwrap_or(1)
                };

                starts.get_or_insert(vec![0; fallback_len]);
                // Keep Slice dynamic when ONNX ends input is non-const.
                ends.get_or_insert(vec![i64::MAX; fallback_len]);

                crate::debug_println!(
                    "[slice] using fallback starts/ends for {}, starts={:?} ends={:?}",
                    node_name,
                    starts,
                    ends
                );
            }

            let starts = starts.ok_or_else(|| {
                OnnxError::InvalidShape("Slice starts must be constant for WebNN".to_string())
            })?;
            let ends = ends.ok_or_else(|| {
                OnnxError::InvalidShape("Slice ends must be constant for WebNN".to_string())
            })?;

            // Normalize lengths: starts/ends must match axes length if provided,
            // otherwise match each other.
            let mut axes_opt: Option<Vec<i64>> = None;
            if inputs.len() >= 4 {
                let axes_name = inputs[3].as_str();
                if let Some(axes) = read_ints(axes_name, context) {
                    axes_opt = Some(axes);
                }
            }

            let desired_len = axes_opt
                .as_ref()
                .map(|a| a.len())
                .unwrap_or_else(|| starts.len().max(ends.len()));
            let mut starts_norm = starts;
            let mut ends_norm = ends;
            if starts_norm.len() > desired_len {
                starts_norm.truncate(desired_len);
            } else {
                starts_norm.resize(desired_len, 0);
            }
            if ends_norm.len() > desired_len {
                ends_norm.truncate(desired_len);
            } else {
                // If we know data shape, use its dims; otherwise use max i64.
                let fill = context
                    .value_shapes
                    .get(inputs[0].as_str())
                    .and_then(|s| s.first())
                    .copied()
                    .unwrap_or(i64::MAX);
                ends_norm.resize(desired_len, fill);
            }

            if let Some(input_shape) = context.resolve_shape(inputs[0].as_str()) {
                let rank = input_shape.len();
                let mut axes = if let Some(a) = axes_opt {
                    if a.is_empty() {
                        (0..desired_len as i64).collect::<Vec<_>>()
                    } else {
                        a
                    }
                } else {
                    (0..desired_len as i64).collect::<Vec<_>>()
                };
                if axes.len() != desired_len {
                    axes.resize(desired_len, 0);
                }
                let axes: Vec<i64> = axes
                    .iter()
                    .map(|&a| normalize_axis_best_effort(a, rank))
                    .collect();

                let mut steps = if inputs.len() >= 5 {
                    let steps_name = inputs[4].as_str();
                    read_ints(steps_name, context).unwrap_or_default()
                } else {
                    Vec::new()
                };
                if steps.len() > desired_len {
                    steps.truncate(desired_len);
                } else {
                    steps.resize(desired_len, 1);
                }

                let mut dense_starts = vec![0i64; rank];
                let mut dense_sizes: Vec<i64> = input_shape.clone();
                let mut dense_strides = vec![1i64; rank];

                // Check if ends input has dynamic dimension metadata
                let ends_dims = context.value_shape_dims.get(ends_name).or_else(|| {
                    context
                        .value_shape_dims
                        .get(&sanitize_identifier(ends_name))
                });

                // Track which dense axes have dynamic sizes
                let mut dynamic_size_info: Vec<Option<rustnn::graph::DynamicDimension>> =
                    vec![None; rank];

                for i in 0..desired_len {
                    let axis = axes[i] as usize;
                    let dim = input_shape[axis];
                    let step = steps[i];
                    if step <= 0 {
                        return Err(OnnxError::InvalidShape(
                            "Slice currently requires positive step values".to_string(),
                        ));
                    }

                    let mut start = starts_norm[i];
                    let mut end = ends_norm[i];
                    if start < 0 {
                        start += dim;
                    }
                    if end == i64::MAX {
                        end = dim;
                    } else if end < 0 {
                        end += dim;
                    }
                    start = start.clamp(0, dim);
                    end = end.clamp(0, dim);

                    let size = if end <= start {
                        0
                    } else {
                        (end - start + step - 1) / step
                    };

                    // If this end value came from a dynamic dimension, mark the size as dynamic
                    if let Some(dims) = ends_dims {
                        if let Some(rustnn::graph::Dimension::Dynamic(dd)) = dims.get(i) {
                            dynamic_size_info[axis] = Some(rustnn::graph::DynamicDimension {
                                name: dd.name.clone(),
                                max_size: size as u32,
                            });
                        }
                    }

                    dense_starts[axis] = start;
                    dense_sizes[axis] = size;
                    dense_strides[axis] = step;
                }

                slice_params = Some((
                    i64_starts_as_u32(&dense_starts)?,
                    slice_sizes_from_i64(&dense_sizes, &dynamic_size_info)?,
                ));
            } else {
                return Err(OnnxError::InvalidShape(
                    "Slice on unknown-rank tensors requires known input shape for WebNN"
                        .to_string(),
                ));
            }
        } else {
            // Extract from attributes (older opset)
            let mut attr_starts: Option<Vec<i64>> = None;
            let mut attr_ends: Option<Vec<i64>> = None;
            let mut attr_axes: Option<Vec<i64>> = None;
            let mut attr_steps: Option<Vec<i64>> = None;
            for attr in node.attribute.as_slice() {
                match attr.name.as_str() {
                    "starts" => attr_starts = Some(attr.ints.clone()),
                    "ends" => attr_ends = Some(attr.ints.clone()),
                    "axes" => attr_axes = Some(attr.ints.clone()),
                    "steps" => attr_steps = Some(attr.ints.clone()),
                    _ => {}
                }
            }
            if attr_starts.is_none() || attr_ends.is_none() {
                return Err(OnnxError::InvalidShape(
                    "Slice requires static starts/ends".to_string(),
                ));
            }

            if let Some(input_shape) = context.resolve_shape(inputs[0].as_str()) {
                let rank = input_shape.len();
                let starts = attr_starts.unwrap();
                let ends = attr_ends.unwrap();
                let axes = attr_axes.unwrap_or_else(|| (0..starts.len() as i64).collect());
                let mut steps = attr_steps.unwrap_or_else(|| vec![1; starts.len()]);

                let desired_len = starts.len().max(ends.len()).max(axes.len());
                let mut starts = starts;
                let mut ends = ends;
                let mut axes = axes;
                if starts.len() < desired_len {
                    starts.resize(desired_len, 0);
                }
                if ends.len() < desired_len {
                    ends.resize(desired_len, i64::MAX);
                }
                if axes.len() < desired_len {
                    axes.resize(desired_len, 0);
                }
                if steps.len() < desired_len {
                    steps.resize(desired_len, 1);
                }

                let axes: Vec<i64> = axes
                    .iter()
                    .map(|&a| normalize_axis_best_effort(a, rank))
                    .collect();
                let mut dense_starts = vec![0i64; rank];
                let mut dense_sizes: Vec<i64> = input_shape.clone();
                let mut dense_strides = vec![1i64; rank];

                for i in 0..desired_len {
                    let axis = axes[i] as usize;
                    let dim = input_shape[axis];
                    let step = steps[i];
                    if step <= 0 {
                        return Err(OnnxError::InvalidShape(
                            "Slice currently requires positive step values".to_string(),
                        ));
                    }

                    let mut start = starts[i];
                    let mut end = ends[i];
                    if start < 0 {
                        start += dim;
                    }
                    if end == i64::MAX {
                        end = dim;
                    } else if end < 0 {
                        end += dim;
                    }
                    start = start.clamp(0, dim);
                    end = end.clamp(0, dim);

                    let size = if end <= start {
                        0
                    } else {
                        (end - start + step - 1) / step
                    };

                    dense_starts[axis] = start;
                    dense_sizes[axis] = size;
                    dense_strides[axis] = step;
                }

                slice_params = Some((
                    i64_starts_as_u32(&dense_starts)?,
                    slice_sizes_from_i64(&dense_sizes, &vec![None; rank])?,
                ));
            }
        }

        let (starts, sizes) = slice_params.ok_or_else(|| {
            OnnxError::InvalidShape(
                "Slice requires static starts/sizes for MLGraphBuilder".to_string(),
            )
        })?;
        let input = b.resolve_operand(&inputs[0])?;
        let out = slice_with_params(b, input, &output_name, &starts, &sizes)?;
        if let Some(output) = node.output.as_slice().first() {
            record_node_output(b, output, &output_name, out);
        }
        let mut result = ConversionResult::default();
        if let Some(output) = node.output.as_slice().first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(output.to_string(), *dtype);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto, TensorProto_DataType};
    use rustnn::DataType;
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
        let attr = AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    fn f32_scalar(value: f32) -> TensorProto {
        TensorProto {
            data_type: TensorProto_DataType::Float as i32,
            dims: vec![],
            raw_data: value.to_le_bytes().to_vec(),
            ..Default::default()
        }
    }

    fn i64_scalar(value: i64) -> TensorProto {
        TensorProto {
            data_type: TensorProto_DataType::Int64 as i32,
            dims: vec![],
            raw_data: value.to_le_bytes().to_vec(),
            ..Default::default()
        }
    }

    #[test]
    fn test_convert_range_float_fractional() {
        let handler = UtilityHandler;
        let node = create_test_node("Range", vec!["start", "limit", "delta"], vec!["output"]);
        let start = f32_scalar(1.0);
        let limit = f32_scalar(2.0);
        let delta = f32_scalar(0.25);
        let mut initializers: std::collections::HashMap<String, &TensorProto> =
            std::collections::HashMap::new();
        initializers.insert("start".to_string(), &start);
        initializers.insert("limit".to_string(), &limit);
        initializers.insert("delta".to_string(), &delta);
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        // Previously `delta` truncated to 0 and errored; now it builds a float32 range.
        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(result.output_types.get("output"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_range_integer_stays_int64() {
        let handler = UtilityHandler;
        let node = create_test_node("Range", vec!["start", "limit", "delta"], vec!["output"]);
        let start = i64_scalar(0);
        let limit = i64_scalar(4);
        let delta = i64_scalar(1);
        let mut initializers: std::collections::HashMap<String, &TensorProto> =
            std::collections::HashMap::new();
        initializers.insert("start".to_string(), &start);
        initializers.insert("limit".to_string(), &limit);
        initializers.insert("delta".to_string(), &delta);
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
        assert_eq!(result.output_types.get("output"), Some(&DataType::Int64));
    }

    #[test]
    fn test_utility_handler_supports() {
        let handler = UtilityHandler;
        assert!(handler.supports("Shape"));
        assert!(handler.supports("Gather"));
        assert!(handler.supports("Slice"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_shape() {
        let handler = UtilityHandler;
        let node = create_test_node("Shape", vec!["x"], vec!["shape"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_gather() {
        let handler = UtilityHandler;
        let mut node = create_test_node("Gather", vec!["data", "indices"], vec!["output"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![2, 3, 4]);
        value_shapes.insert("indices".to_string(), vec![2]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_slice() {
        let handler = UtilityHandler;
        let node = create_test_node(
            "Slice",
            vec!["x", "starts", "ends", "axes", "steps"],
            vec!["output"],
        );
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 128]);
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("starts".to_string(), vec![0]);
        const_values.insert("ends".to_string(), vec![128]);
        const_values.insert("axes".to_string(), vec![1]);
        const_values.insert("steps".to_string(), vec![1]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_constant_of_shape_prefers_dynamic_output_dims() {
        let handler = UtilityHandler;
        let mut node = create_test_node("ConstantOfShape", vec!["shape"], vec!["output"]);
        node.attribute.push(AttributeProto {
            name: "value".to_string(),
            t: Some(TensorProto {
                data_type: TensorProto_DataType::Float as i32,
                dims: vec![],
                raw_data: 0f32.to_le_bytes().to_vec(),
                ..Default::default()
            }),
            ..Default::default()
        });

        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("output".to_string(), vec![4096, 4096]);
        let mut value_shape_dims = std::collections::HashMap::new();
        value_shape_dims.insert(
            "output".to_string(),
            vec![
                rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "sequence_length".to_string(),
                    max_size: 4096,
                }),
                rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "past_sequence_length + 1".to_string(),
                    max_size: 4096,
                }),
            ],
        );
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("shape".to_string(), vec![4096, 4096]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: &value_shape_dims,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(result.output_types.get("output"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_trilu_defaults() {
        let handler = UtilityHandler;
        let node = create_test_node("Trilu", vec!["x"], vec!["y"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let mut value_types = std::collections::HashMap::new();
        value_types.insert("x".to_string(), DataType::Float32);
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
        assert_eq!(result.output_types.get("y"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_trilu_with_k_and_lower() {
        let handler = UtilityHandler;
        let mut node = create_test_node("Trilu", vec!["x", "k"], vec!["y"]);
        add_int_attribute(&mut node, "upper", 0);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("k".to_string(), vec![2]);
        let value_ids = std::collections::HashMap::new();
        let mut value_types = std::collections::HashMap::new();
        value_types.insert("x".to_string(), DataType::Float16);
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
        assert_eq!(result.output_types.get("y"), Some(&DataType::Float16));
    }
}
