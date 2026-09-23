#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
import unittest

# This focused test does not use ONNX.
sys.modules.setdefault("onnx", types.ModuleType("onnx"))

import generate_manifest as manifest


class LegacyValidationMetadataTests(unittest.TestCase):
    def test_validation_metadata_is_not_preserved(self):
        file = "org--repo/onnx/model.onnx"
        baselines = manifest.Baselines(
            [
                {
                    "file": file,
                    "validation": {"tier": "blocked", "reason": "legacy"},
                    "coreml_slow": "preserve unrelated metadata",
                }
            ]
        )

        self.assertEqual(
            baselines.manual_fields[(file, ())],
            {"coreml_slow": "preserve unrelated metadata"},
        )


if __name__ == "__main__":
    unittest.main()
