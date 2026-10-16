/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

pub mod cache;
pub mod debug;
pub mod model_validation;
pub mod protos;
pub mod validation;

pub mod onnx;

pub use onnx::convert::{
    cache_onnx_model, convert_model_proto, convert_model_proto_with_backend, convert_onnx,
    convert_onnx_with_backend, export_model_proto, export_onnx, BackendChoice, ConvertOptions,
    OnnxError, UnsupportedOpEntry, ValidatedGraph,
};
pub use onnx::test_models;
pub use validation::{
    load_safetensors_inputs, validate_cached_model, validate_cached_model_with_inputs,
    validate_cached_model_with_inputs_on_backend, validate_cached_model_with_options,
    validate_cached_model_with_options_on_backend, validate_cached_model_with_overrides,
    ValidationSummary,
};
