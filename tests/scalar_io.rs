/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for rank-0 ONNX graph inputs and outputs.

#[allow(dead_code)]
mod common;

use common::{assert_op_matches_ort, ExpectConvertOp};
use onnx2webnn::test_models::prelude::*;

#[test]
fn scalar_input_and_output_match_ort() {
    let model = model(
        17,
        graph(
            "scalar_identity",
            vec![f32_input("input", &[])],
            vec![f32_output("output", &[])],
            vec![node(
                "Identity",
                "scalar_identity",
                &["input"],
                &["output"],
                &[],
            )],
            vec![],
        ),
    );

    assert_op_matches_ort(model, ExpectConvertOp::Success, 17);
}
