/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared helpers for ONNX op conversion integration tests.

mod runner;

pub use runner::{assert_op_matches_ort, ExpectConvertOp};
