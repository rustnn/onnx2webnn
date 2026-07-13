/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Helpers for constructing in-memory ONNX [`ModelProto`] fixtures in tests.

mod builder;

pub use builder::*;

/// Convenient imports for generated op fixture builders.
pub mod prelude {
    pub use super::builder::*;
    pub use crate::protos::onnx::ModelProto;
}
