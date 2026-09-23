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
    cache_onnx_model, convert_model_proto, convert_onnx, ConvertOptions, OnnxError,
    UnsupportedOpEntry, ValidatedGraph,
};
pub use onnx::test_models;
pub use validation::{
    validate_cached_model, validate_cached_model_with_options,
    validate_cached_model_with_overrides, ValidationSummary,
};
